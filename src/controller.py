from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils, checkpoint_utils
from fairseq.models import (
	register_model,
	register_model_architecture,
)
from fairseq.models.transformer import (
	TransformerModel,
	TransformerEncoder,
	TransformerDecoder,
	base_architecture,
	Embedding,
	Linear,
)
from fairseq.modules import (
	SinusoidalPositionalEmbedding,
	TransformerDecoderLayer,
	TransformerEncoderLayer,
	LayerNorm,
	MultiheadAttention
)
import random

from .modules import ControllerDecoderLayer,MultihotNumEmbedding
from .rule_translator import RuleNumTranslator
from .dictionary import EON


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


EncoderOut = namedtuple('TransformerEncoderOut', [
    'encoder_out',  # T x B x C
    'encoder_padding_mask',  # B x T
    'encoder_embedding',  # B x T x C
    'encoder_states',  # List[T x B x C]
	'src_word_maps',
	'src_tokens',
])
def assert_valid_value(tensor):
	assert not torch.isnan(tensor).any(), torch.isnan(tensor).nonzero()
	assert not torch.isinf(tensor).any(), torch.isinf(tensor).nonzero()




class ControllerEncoder(TransformerEncoder):
	def __init__(self, args, dictionary, embed_tokens, num_embed):
		super().__init__(args, dictionary, embed_tokens)
		self.args = args
		embed_dim = args.encoder_embed_dim
		self.adapter_layers = nn.ModuleList([
			TransformerEncoderLayer(args) for i in range(1)])
		self.embed_dim = embed_dim
		self.layer_attn = MultiheadAttention(embed_dim, num_heads=8)
		self.la_ln = LayerNorm(embed_dim)
		self.activation_fn = utils.get_activation_fn(
			activation=getattr(args, 'activation_fn', 'relu'))
		self.embed_proj = nn.Linear(embed_dim, embed_dim, bias=True)
		self.label_embedding=nn.Embedding(2, embed_dim)
		self.num_embed = num_embed
		self.ce_ln = LayerNorm(embed_dim)
		self.layer_weights = nn.Parameter(torch.zeros(args.encoder_layers))
		self.la_fc1 = Linear(2*embed_dim, 4*embed_dim)
		self.la_fc2 = Linear(4*embed_dim,embed_dim)

	def forward_controller_embedding(self, src_tokens):
		# embed tokens and positions
		x = embed = self.embed_scale * self.controller_embed(src_tokens)
		if self.embed_positions is not None:
			x = embed + self.embed_positions(src_tokens)
		if self.layernorm_embedding:
			x = self.ce_ln(x)
		x = F.dropout(x, p=self.dropout, training=self.training)
		return x, embed

	def forward(self, src_tokens, src_word_maps, src_lengths,src_labels=None, cls_input=None, attn_mask=None, **unused):
		"""
		Args:
			src_tokens (LongTensor): tokens in the source language of shape
				`(batch, src_len)`
			src_lengths (torch.LongTensor): lengths of each source sentence of
				shape `(batch)`
			return_all_hiddens (bool, optional): also return all of the
				intermediate hidden states (default: False).

		Returns:
			namedtuple:
				- **encoder_out** (Tensor): the last encoder layer's output of
				  shape `(src_len, batch, embed_dim)`
				- **encoder_padding_mask** (ByteTensor): the positions of
				  padding elements of shape `(batch, src_len)`
				- **encoder_embedding** (Tensor): the (scaled) embedding lookup
				  of shape `(batch, src_len, embed_dim)`
				- **encoder_states** (List[Tensor]): all intermediate
				  hidden states of shape `(src_len, batch, embed_dim)`.
				  Only populated if *return_all_hiddens* is True.
		"""
		bsz = src_tokens.size(0)
		x, encoder_embedding = self.forward_embedding(src_tokens)

		src_mhot = self.dictionary.get_multihot_sequence(src_tokens)
		controller_embed = self.num_embed(src_mhot)
		controller_embed = controller_embed.transpose(0, 1)
		
		assert_valid_value(controller_embed)
		query = self.embed_proj(encoder_embedding)
		x = x.transpose(0, 1)
		embed = x
		# compute padding mask
		encoder_padding_mask = src_tokens.eq(self.padding_idx)
		if not encoder_padding_mask.any():
			encoder_padding_mask = None

		encoder_states = []
		# encoder layers
		for layer in self.layers:
			# add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
			x = layer(x, encoder_padding_mask, attn_mask=attn_mask)
			encoder_states.append(x)

		if self.layer_norm:
			x = self.layer_norm(x)
			encoder_states[-1] = x
		inner_states = torch.stack(encoder_states).contiguous()
		inner_states = inner_states.view(inner_states.size(0), -1)
		if src_labels is None:
			src_labels=torch.zeros_like(src_tokens)
		label_embed=self.label_embedding(src_labels)
		label_embed=label_embed.transpose(0,1)
		final_x = controller_embed
		final_x = torch.cat([x,label_embed], dim=-1)

		final_x = self.la_fc1(final_x)#+controller_embed
		final_x=self.activation_fn(final_x)
		final_x=self.la_fc2(final_x)

		assert_valid_value(final_x)
		final_x = self.la_ln(final_x)
		final_x=F.dropout(final_x,p=self.args.dropout,training=self.training)
		if encoder_states is None:
			encoder_states = [final_x]
		assert x.size(1)==bsz
		encoder_states.append(final_x)
		return EncoderOut(
			encoder_out=x,  # T x B x C
			encoder_padding_mask=encoder_padding_mask,  # B x T
			encoder_embedding=encoder_embedding,  # B x T x C
			encoder_states=encoder_states,  # List[T x B x C]
			src_word_maps=src_word_maps,
			src_tokens=src_tokens,
		)

	def reorder_encoder_out(self, encoder_out, new_order):
		"""
		Reorder encoder output according to *new_order*.

		Args:
			encoder_out: output from the ``forward()`` method
			new_order (LongTensor): desired order

		Returns:
			*encoder_out* rearranged according to *new_order*
		"""
		if encoder_out.encoder_out is not None:
			encoder_out = encoder_out._replace(
				encoder_out=encoder_out.encoder_out.index_select(1, new_order)
			)
		if encoder_out.encoder_padding_mask is not None:
			encoder_out = encoder_out._replace(
				encoder_padding_mask=encoder_out.encoder_padding_mask.index_select(0, new_order)
			)
		if encoder_out.encoder_embedding is not None:
			encoder_out = encoder_out._replace(
				encoder_embedding=encoder_out.encoder_embedding.index_select(0, new_order)
			)
		if encoder_out.encoder_states is not None:
			for idx, state in enumerate(encoder_out.encoder_states):
				encoder_out.encoder_states[idx] = state.index_select(1, new_order)
		if encoder_out.src_word_maps is not None:
			encoder_out=encoder_out._replace(
				src_word_maps=encoder_out.src_word_maps.index_select(0,new_order)
			)
		return encoder_out

	def set_controller_trainable(self):
		for param in self.adapter_layers.parameters():
			param.requires_grad = True
		for param in self.label_embedding.parameters():
			param.requires_grad = True
		for param in self.layer_attn.parameters():
			param.requires_grad = True
		for param in self.embed_proj.parameters():
			param.requires_grad = True
		for param in self.la_ln.parameters():
			param.requires_grad = True
		self.layer_weights.requires_grad = True
		for param in self.ce_ln.parameters():
			param.requires_grad = True
		for param in self.la_fc1.parameters():
			param.requires_grad = True
		for param in self.la_fc2.parameters():
    			param.requires_grad = True


@register_model('controller')
class Controller(TransformerModel):
	def __init__(self, args, encoder, project_out, fig_embedding, decoder,num_translator, bos_idx):
		super().__init__(args, encoder, decoder)
		self.mode = 0
		self.project_out = project_out
		self.fig_embedding = fig_embedding
		self.bos_idx = bos_idx
		self.ld = args.local_distance
		self.nt_module=args.nt_module
		self.num_translator=num_translator
		if getattr(args, 'pretrained_model', None) is not None:
			pre_state = checkpoint_utils.load_checkpoint_to_cpu(args.pretrained_model)['model']

			self.load_state_dict(state_dict=pre_state, strict=False)

	@staticmethod
	def add_args(parser):
		TransformerModel.add_args(parser)
		parser.add_argument('--mtmodel-path', type=str, metavar='STR')
		parser.add_argument('--pretrained-model', type=str, metavar='STR')
		parser.add_argument('--fix-encoder', action='store_true', default=False)
		parser.add_argument('--fix-embedding', action='store_true', default=False)
		parser.add_argument('--fix-decoder', action='store_true', default=False)
		parser.add_argument('--fix-enc-proj', action='store_true', default=False)
		parser.add_argument('--local-distance', type=int, default=1)
		parser.add_argument('--nt-module',type=str,metavar='STR',default='neural')
		parser.add_argument('--controller-layer',type=int,default=1)

	@classmethod
	def build_model(cls, args, task):
		args.debugging = getattr(task.args,'debugging',False)
		# print(args.debugging)
		base_controller(args)
		assert args.mtmodel_path is not None, 'MTModel model\'s path should be specified.'
		if args.encoder_layers_to_keep:
			args.encoder_layers = len(args.encoder_layers_to_keep.split(','))
		if args.decoder_layers_to_keep:
			args.encoder_layers = len(args.decoder_layers_to_keep.split(','))
		if getattr(args, 'max_source_positions', None) is None:
			args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
		if getattr(args, 'max_target_positions', None) is None:
			args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
		src_dict, tgt_dict, num_dict = task.source_dictionary, task.target_dictionary, task.number_dictionary

		def build_embedding(dictionary, embed_dim, path=None):
			num_embeddings = len(dictionary)
			padding_idx = dictionary.pad()
			emb = Embedding(num_embeddings, embed_dim, padding_idx)
			# if provided, load from preloaded dictionaries
			if path:
				embed_dict = utils.parse_embedding(path)
				utils.load_embedding(embed_dict, dictionary, emb)
			return emb

		if args.share_all_embeddings:
			if src_dict != tgt_dict:
				raise ValueError('--shared-all-embeddings requires a joined dictionary')
			if args.encoder_embed_dim != args.decoder_embed_dim:
				raise ValueError(
					'--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim'
				)
			if args.decoder_embed_path and (
					args.decoder_embed_path != args.encoder_embed_path
			):
				raise ValueError(
					'--share-all-embeddings not compatible with --decoder-embed-path'
				)
			encoder_embed_tokens = build_embedding(
				src_dict, args.encoder_embed_dim, args.encoder_embed_path
			)
			decoder_embed_tokens = encoder_embed_tokens
			args.share_decoder_input_output_embed = True
		else:
			encoder_embed_tokens = build_embedding(
				src_dict, args.encoder_embed_dim, args.encoder_embed_path
			)
			decoder_embed_tokens = build_embedding(
				tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
			)
		assert args.encoder_embed_dim == args.decoder_embed_dim
		embed_dim = args.encoder_embed_dim
		# multihot_num_embed = nn.Parameter(torch.zeros(embed_dim, 33)).float()
		# nn.init.normal_(multihot_num_embed, mean=0, std=embed_dim ** -0.5)
		multihot_num_embed = MultihotNumEmbedding(embed_dim)
		encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens, multihot_num_embed)

		project_out = Linear(args.encoder_embed_dim, 1)

		if getattr(args, 'regression', False):
			out_size = 1
		decoder = cls.build_labeling_decoder(args, tgt_dict, num_dict, decoder_embed_tokens, multihot_num_embed,args.nt_module)
		if args.nt_module=='rule' and num_dict is not None:
			num_translator=RuleNumTranslator(args,src_dict,num_dict)
		else:
			num_translator=None
		fig_embedding = Embedding(2, args.encoder_embed_dim, 0)
		if getattr(args, 'mtmodel_path', None) is not None:
			state = checkpoint_utils.load_checkpoint_to_cpu(args.mtmodel_path)
			encoder_state = {}
			decoder_state = {}
			embed_state = {}
			enc_str = 'encoder.'
			dec_str = 'decoder.'
			embed_str = 'decoder.embed_tokens.'
			enc_len = len(enc_str)
			dec_len = len(dec_str)
			for k, v in state['model'].items():
				if k.startswith(enc_str):
					encoder_state[k[enc_len:]] = v
				if k.startswith(dec_str):
					if embed_str in k:
						embed_state[k[len(embed_str):]] = v
					else:
						decoder_state[k[dec_len:]] = v
			encoder.load_state_dict(encoder_state, strict=False)
			decoder.load_state_dict(decoder_state, strict=False)
			embed_weight = embed_state['weight']
			decoder_embed_tokens.weight.data[:embed_weight.size(0)] = embed_weight.data
		if args.fix_encoder:
			for param in encoder.parameters():
				param.requires_grad = False
			if not args.fix_enc_proj:
				encoder.set_controller_trainable()
			# for param in self.project_out.parameters():
			# 	param.requires_grad=True
		if args.fix_decoder:
			for param in decoder.parameters():
				param.requires_grad = False
			decoder.set_nummodule_trainable()
		if args.fix_embedding:
			for param in encoder_embed_tokens.parameters():
				param.requires_grad = False
			for param in decoder_embed_tokens.parameters():
				param.requires_grad = False
		eos_idx = src_dict.eos()
		bos_idx = src_dict.bos()
		return cls(args, encoder, project_out, fig_embedding, decoder,num_translator, bos_idx)

	@classmethod
	def build_labeling_decoder(cls, args, tgt_dict, num_dict, embed_tokens, num_embed,nt_module):
		return TransformerLabelingDecoder(
			args, tgt_dict, num_dict,
			embed_tokens,
			num_embed,
			nt_module,
			no_encoder_attn=getattr(args, 'no_cross_attention', False)
		)

	def default_mode(self):
		self.decoder.mode = 0

	def controller_mode(self):
		self.decoder.mode = 1

	def number_mode(self):
		self.decoder.mode = 2

	def ensemble_mode(self):
		self.decoder.mode = 3

	def forward_encoder(self, encoder_input):
		encoder_out = self.encoder(**encoder_input)

		return encoder_out

	def project_encoder(self, src_tokens, encoder_out):
		encoder_projection = self.project_out(encoder_out.encoder_states[-1])
		encoder_projection = F.sigmoid(encoder_projection)
		encoder_projection = encoder_projection.transpose(0, 1).squeeze(-1)
		if encoder_out.encoder_padding_mask is not None:
			encoder_projection = encoder_projection.masked_fill(encoder_out.encoder_padding_mask, 0)
		fig_flag = encoder_projection  # .round().long()
		bos_mask = src_tokens.eq(self.bos_idx)
		fig_flag = fig_flag.masked_fill(bos_mask, 0)
		fig_flag = fig_flag.transpose(0, 1)
		# print(fig_flag.size())
		encoder_hid = encoder_out.encoder_states[
			-1]  # .float()*fig_flag.unsqueeze(2)).type_as(encoder_out.encoder_states[-1]) # + self.fig_embedding(fig_flag)
		encoder_out.encoder_states[-1] = encoder_hid
		encoder_out = EncoderOut(
			encoder_out=encoder_out.encoder_out,  # T x B x C
			encoder_padding_mask=encoder_out.encoder_padding_mask,  # B x T
			encoder_embedding=encoder_out.encoder_embedding,  # B x T x C
			encoder_states=encoder_out.encoder_states,  # List[T x B x C]
			src_word_maps=encoder_out.src_word_maps,
			src_tokens=encoder_out.src_tokens
		)
		return encoder_out, encoder_projection

	def get_targets(self, sample, net_output, use_regression=False):
		"""Get targets from either the sample or the net's output."""
		if getattr(self.args, "regression", False) and use_regression:
			return sample['regression_target']
		else:
			return sample['target']
	def clean_num_translator(self,incremental_state,step,gen_state):
		self.num_translator.clean(incremental_state,step,gen_state)
	def translate_num(self,src_tokens,attn,incremental_state=None,step=0,requires_label=False):
		if self.nt_module=='rule':
			return self.num_translator(src_tokens,attn,incremental_state,step,requires_label)
		return None
	def forward(self, src_tokens,  src_lengths, tgt_tokens, src_mhot=None, src_word_maps=None, output_proj=False, **kwargs):
		"""
		Run the forward pass for an encoder-decoder model.

		First feed a batch of source tokens through the encoder. Then, feed the
		encoder output and previous decoder outputs (i.e., teacher forcing) to
		the decoder to produce the next outputs::

			encoder_out = self.encoder(src_tokens, src_lengths)
			return self.decoder(prev_output_tokens, encoder_out)

		Args:
			src_tokens (LongTensor): tokens in the source language of shape
				`(batch, src_len)`
			src_lengths (LongTensor): source sentence lengths of shape `(batch)`
			prev_output_tokens (LongTensor): previous decoder outputs of shape
				`(batch, tgt_len)`, for teacher forcing

		Returns:
			tuple:
				- the decoder's output of shape `(batch, tgt_len, vocab)`
				- a dictionary with any model-specific outputs
		"""
		encoder_input = kwargs.copy()
		encoder_input['src_tokens'] = src_tokens
		encoder_input['src_mhot'] = src_mhot
		encoder_input['src_word_maps'] = src_word_maps
		encoder_input['src_lengths'] = src_lengths
		encoder_out = self.forward_encoder(encoder_input)
		encoder_out, encoder_projection = self.project_encoder(src_tokens, encoder_out)
		x,controller_x,number_x,value_x,extra = self.decoder(tgt_tokens, encoder_out=encoder_out, **kwargs)
		if self.nt_module=='rule' and self.num_translator is not None:
			number_x=self.translate_num(src_tokens,extra['attn'])
		if output_proj:
			return x,controller_x,number_x,value_x,extra, encoder_projection
		else:
			return x,controller_x,number_x,value_x,extra

	@classmethod
	def build_encoder(cls, args, src_dict, embed_tokens, num_embed):
		return ControllerEncoder(args, src_dict, embed_tokens, num_embed)
	def reorder_incremental_state(self,incremental_state,reorder_state):
		self.decoder.reorder_incremental_state_scripting(incremental_state,reorder_state)
		if self.num_translator is not None:
			self.num_translator.reorder_incremental_state(incremental_state,reorder_state)

class RealController(nn.Module):
	def __init__(self,args,no_encoder_attn=False):
		super().__init__()
		self.args=args
		self.embed_dim=args.decoder_embed_dim
		self.output_embed_dim=args.decoder_output_dim
		self.controller_layer = nn.ModuleList([
			TransformerDecoderLayer(args, no_encoder_attn)
			for _ in range(2)])
		# self.controller_layer=TransformerDecoderLayer(args,no_encoder_attn)
		self.controller_out = nn.Parameter(torch.Tensor(2, self.output_embed_dim))
		self.controller_embed_proj = nn.Parameter(torch.Tensor(self.embed_dim, self.embed_dim))
		self.controller_attn = MultiheadAttention(self.embed_dim, num_heads=8)
		self.controller_layer_weights = nn.Parameter(torch.zeros(args.decoder_layers))
		self.controller_layer_norm = LayerNorm(self.embed_dim)
		nn.init.normal_(self.controller_out, mean=0, std=self.output_embed_dim ** -0.5)
		nn.init.normal_(self.controller_embed_proj, mean=0, std=self.output_embed_dim ** -0.5)
	def forward(self,x,encoder_out,self_attn_mask,self_attn_padding_mask,incremental_state=None,requires_attn=False):
		src_tokens=encoder_out.src_tokens
		encoder_state=encoder_out.encoder_states[-1]
		encoder_padding_mask = encoder_out.encoder_padding_mask if encoder_out is not None else None
		for layer in self.controller_layer:
			controller_x, layer_attn,_ = layer(
				x,
				encoder_state,
				encoder_padding_mask,
				incremental_state,
				self_attn_mask=self_attn_mask,
				self_attn_padding_mask=self_attn_padding_mask,
				need_attn=True,
				need_head_weights=False,
				 )
		attn=layer_attn
		if getattr(self.args,'debugging',False) and not self.training:
			for i in range(attn.size(1)):
				print(attn[:,i,:])
				print(src_tokens[:,attn[:,i,:].argmax().item()])
		controller_x = controller_x.transpose(0, 1)
		controller_x = F.linear(controller_x, self.controller_out)
		assert not torch.isnan(controller_x).any()
		assert not torch.isinf(controller_x).any()
		if requires_attn:
			return controller_x,attn
		return controller_x

class NumeralTranslator(nn.Module):
	def __init__(self,args,dictionary,num_dictionary,num_embed,no_encoder_attn):
		super().__init__()
		self.args=args
		self.dictionary=dictionary
		self.num_dictionary=num_dictionary
		self.embed_dim=args.decoder_embed_dim
		self.output_embed_dim=args.decoder_output_dim
		num_size = 232
		if num_dictionary is not None:
			num_size = len(num_dictionary)

		self.number_layers = nn.ModuleList([])
		self.number_layers.extend([
			                          TransformerDecoderLayer(args, no_encoder_attn)
			                          for _ in range(2)] + [ControllerDecoderLayer(args, no_encoder_attn)]
		                          )
		# multihot number embedding
		self.num_embed = num_embed
		self.number_fc = Linear(3 * self.embed_dim, self.embed_dim)
		self.number_out = nn.Parameter(torch.Tensor(num_size, self.output_embed_dim)).float()
		nn.init.normal_(self.number_out, mean=0, std=self.output_embed_dim ** -0.5)
		self.value_out = nn.Parameter(torch.Tensor(num_size, self.output_embed_dim)).float()
		nn.init.normal_(self.value_out, mean=0, std=self.output_embed_dim ** -0.5)
		self.ne_ln = LayerNorm(self.embed_dim)
		self.number_embed_proj = nn.Parameter(torch.Tensor(self.embed_dim, self.embed_dim)).float()
		nn.init.normal_(self.number_embed_proj, mean=0, std=self.embed_dim ** -0.5)
		self.number_attn = MultiheadAttention(self.embed_dim, num_heads=8)
		self.number_layer_weights = nn.Parameter(torch.zeros(args.decoder_layers)).float()
		self.adaptive_softmax = None
		self.number_layer_norm = LayerNorm(self.embed_dim)
	def forward(self,prev_output_tokens,
			encoder_out,
	        decoder_embed,
			decoder_out,
	        self_attn_mask,
	        self_attn_padding_mask,
			incremental_state=None,
			**extra_args):
		tgt_mhot = self.dictionary.get_multihot_sequence(prev_output_tokens)
		number_embed = self.num_embed(tgt_mhot)
		number_embed = number_embed.transpose(0, 1)
		encoder_state = None
		if encoder_out is not None:
			encoder_state = encoder_out.encoder_states[-1]
		noised_x = F.dropout(decoder_out, p=0.9, training=self.training)
		number_x = torch.cat([decoder_embed, noised_x, number_embed], dim=-1)
		number_x = self.number_fc(number_x)  # +number_embed
		assert not torch.isnan(number_x).any()
		for layer in self.number_layers[:2]:
			number_x, layer_attn = layer(
				number_x,
				encoder_state,
				encoder_out.encoder_padding_mask if encoder_out is not None else None,
				incremental_state,
				self_attn_mask=self_attn_mask,
				self_attn_padding_mask=self_attn_padding_mask,
				need_attn=True,
				need_head_weights=False,
			)
		number_x, layer_attn = self.number_layers[-1](
			number_x,
			encoder_state,
			encoder_out.encoder_padding_mask if encoder_out is not None else None,
			incremental_state,
			src_word_maps=encoder_out.src_word_maps,
			self_attn_mask=self_attn_mask,
			self_attn_padding_mask=self_attn_padding_mask,
			need_attn=True,
			need_head_weights=False,
		)
		if self.number_layer_norm:
			number_x = self.number_layer_norm(number_x)
		number_x = number_x.transpose(0, 1)
		assert not torch.isnan(number_x).any()
		number_x = F.linear(number_x, self.number_out)
		return number_x



class TransformerLabelingDecoder(TransformerDecoder):
	"""
	Transformer decoder consisting of *args.decoder_layers* layers. Each layer
	is a :class:`TransformerDecoderLayer`.

	Args:
		args (argparse.Namespace): parsed command-line arguments
		dictionary (~fairseq.data.Dictionary): decoding dictionary
		embed_tokens (torch.nn.Embedding): output embedding
		no_encoder_attn (bool, optional): whether to attend to encoder outputs
			(default: False).
	"""

	def __init__(self, args, dictionary, num_dictionary, embed_tokens, num_embed,nt_module, no_encoder_attn=False):
		super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
		self.args = args
		self.padding_idx = embed_tokens.padding_idx

		self.controller_layer = RealController(args, no_encoder_attn)
		self.numeral_layer=None
		if nt_module=='neural':
			self.numeral_layer=NumeralTranslator(args,dictionary,num_dictionary,num_embed,no_encoder_attn)

		self.mode = 0

		self.activation_fn = utils.get_activation_fn(
			activation=getattr(args, 'activation_fn', 'relu')
		)

	def set_nummodule_trainable(self):
		for param in self.controller_layer.parameters():
			param.requires_grad=True
		if self.numeral_layer is not None:
			for param in self.numeral_layer.parameters():
				param.requires_grad=True
		# self.value_out.requires_grad = True

	def forward(
			self,
			tgt_tokens,
			encoder_out=None,
			incremental_state=None,
			features_only=False,
			**extra_args
	):
		"""
		Args:
			prev_output_tokens (LongTensor): previous decoder outputs of shape
				`(batch, tgt_len)`, for teacher forcing
			encoder_out (optional): output from the encoder, used for
				encoder-side attention
			incremental_state (dict): dictionary used for storing state during
				:ref:`Incremental decoding`
			features_only (bool, optional): only return features without
				applying output layer (default: False).

		Returns:
			tuple:
				- the decoder's output of shape `(batch, tgt_len, vocab)`
				- a dictionary with any model-specific outputs
		"""
		x, controller_x, number_x, extra = self.extract_features(
			tgt_tokens,
			encoder_out=encoder_out,
			incremental_state=incremental_state,
			**extra_args
		)
		value_x = number_x
		if not features_only:
			x= self.output_layer(x)
		if self.mode == 1:
			return controller_x, extra
		return x, controller_x, number_x, value_x, extra

	def extract_features(
			self,
			prev_output_tokens,
			encoder_out=None,
			incremental_state=None,
			full_context_alignment=False,
			alignment_layer=None,
			alignment_heads=None,
			**unused,
	):
		"""
		Similar to *forward* but only return features.

		Includes several features from "Jointly Learning to Align and
		Translate with Transformer Models" (Garg et al., EMNLP 2019).

		Args:
			full_context_alignment (bool, optional): don't apply
				auto-regressive mask to self-attention (default: False).
			alignment_layer (int, optional): return mean alignment over
				heads at this layer (default: last layer).
			alignment_heads (int, optional): only average alignment over
				this many heads (default: all heads).

		Returns:
			tuple:
				- the decoder's features of shape `(batch, tgt_len, embed_dim)`
				- a dictionary with any model-specific outputs
		"""
		# torch.autograd.set_detect_anomaly(True)
		layer_num = len(self.layers)
		if alignment_layer is None:
			alignment_layer = layer_num - 1

		# embed positions
		assert prev_output_tokens is not None
		positions = self.embed_positions(
			prev_output_tokens,
			incremental_state=incremental_state,
		) if self.embed_positions is not None else None

		if incremental_state is not None:
			prev_output_tokens = prev_output_tokens[:, -1:]
			if positions is not None:
				positions = positions[:, -1:]
		x = self.embed_scale * self.embed_tokens(prev_output_tokens)
		if self.quant_noise is not None:
			x=self.quant_noise(x)

		if self.project_in_dim is not None:
			x = self.project_in_dim(x)

		if positions is not None:
			x = x + positions

		if self.layernorm_embedding:
			x = self.layernorm_embedding(x)

		x = self.dropout_module(x)

		# B x T x C -> T x B x C
		x = x.transpose(0, 1)
		embed = x


		# controller_embed=controller_embed.transpose(0,1)
		self_attn_padding_mask = None
		# print(prev_output_tokens)
		if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
			self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

		# decoder layers
		attn = None
		inner_states = []
		for idx, layer in enumerate(self.layers):
			encoder_state = None
			if encoder_out is not None:
				encoder_state = encoder_out.encoder_out

			if incremental_state is None and not full_context_alignment:
				self_attn_mask = self.buffered_future_mask(x)
			else:
				self_attn_mask = None

			# add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
			x, layer_attn,_ = layer(
				x,
				encoder_state,
				encoder_out.encoder_padding_mask if encoder_out is not None else None,
				incremental_state,
				self_attn_mask=self_attn_mask,
				self_attn_padding_mask=self_attn_padding_mask,
				need_attn=True,
				need_head_weights=(idx == alignment_layer),
			)

			inner_states.append(x)
			if layer_attn is not None and idx == alignment_layer:
				attn = layer_attn
				layer_attn = layer_attn.mean(dim=0)
		if getattr(self.args,'debugging',False) and not self.training:
			print('encoder attn',layer_attn)
		ts_inner_states = torch.stack(inner_states, dim=0).contiguous()
		# print(ts_inner_states.size())
		ts_inner_states = ts_inner_states.view(len(inner_states), x.size(0) * x.size(1), -1)
		controller_attn=None
		controller_x,controller_attn=self.controller_layer(x,encoder_out,self_attn_mask,self_attn_padding_mask,incremental_state,True)
		number_x=None
		if self.numeral_layer is not None:
			number_x=self.numeral_layer(prev_output_tokens=prev_output_tokens,
			                            encoder_out=encoder_out,
			                            decoder_embed=embed,
			                            decoder_out=x,
			                            self_attn_mask=self_attn_mask,
			                            self_attn_padding_mask=self_attn_padding_mask,
			                            incremental_state=incremental_state)


		if layer_attn is not None: # and alignment_layer == layer_num + 1:
			attn = layer_attn
		assert len(attn.size())==3
		if attn is not None:
			if alignment_heads is not None:
				attn = attn[:alignment_heads]

				# average probabilities over heads
				attn = attn.mean(dim=0)

		if self.layer_norm:
			x = self.layer_norm(x)


		# T x B x C -> B x T x C
		x = x.transpose(0, 1)

		if self.project_out_dim is not None:
			x = self.project_out_dim(x)


		return x, controller_x, number_x, {'attn': attn, 'inner_states': inner_states,'controller_attn':controller_attn}

	def output_layer(self, features,**kwargs):
		"""Project features to the vocabulary size."""

		if self.adaptive_softmax is None:
			# project back to size of vocabulary
			if self.share_input_output_embed:
				x = F.linear(features, self.embed_tokens.weight)
			else:
				x = F.linear(features, self.embed_out)

			return x
		else:
			return features

	def upgrade_state_dict_named(self, state_dict, name):
		"""Upgrade a (possibly old) state dict for new versions of fairseq."""
		if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
			weights_key = '{}.embed_positions.weights'.format(name)
			if weights_key in state_dict:
				del state_dict[weights_key]
			state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

		for i in range(len(self.layers)):
			# update layer norms
			layer_norm_map = {
				'0': 'self_attn_layer_norm',
				'1': 'encoder_attn_layer_norm',
				'2': 'final_layer_norm'
			}
			for old, new in layer_norm_map.items():
				for m in ('weight', 'bias'):
					k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
					if k in state_dict:
						state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
						del state_dict[k]

		version_key = '{}.version'.format(name)
		if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
			# earlier checkpoints did not normalize after the stack of layers
			self.layer_norm = None
			self.normalize = False
			state_dict[version_key] = torch.Tensor([1])

		return state_dict


@register_model_architecture('controller', 'controller')
def base_controller(args):
	base_architecture(args)
	args.debugging = getattr(args, 'debugging', False)
