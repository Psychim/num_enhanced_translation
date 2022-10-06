from fairseq.modules import TransformerDecoderLayer
import torch
from torch.nn import functional as F
import torch.nn as nn

from .attention import DebugMultiheadAttention, DLayersEncoderAttention


class ControllerDecoderLayer(TransformerDecoderLayer):
	"""Decoder layer block.

	In the original paper each operation (multi-head attention, encoder
	attention or FFN) is postprocessed with: `dropout -> add residual ->
	layernorm`. In the tensor2tensor code they suggest that learning is more
	robust when preprocessing each layer with layernorm and postprocessing with:
	`dropout -> add residual`. We default to the approach in the paper, but the
	tensor2tensor approach can be enabled by setting
	*args.decoder_normalize_before* to ``True``.

	Args:
		args (argparse.Namespace): parsed command-line arguments
		no_encoder_attn (bool, optional): whether to attend to encoder outputs
			(default: False).
	"""

	def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
		super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)
		self.self_attn = DebugMultiheadAttention(
			embed_dim=self.embed_dim,
			num_heads=args.decoder_attention_heads,
			dropout=args.attention_dropout,
			add_bias_kv=add_bias_kv,
			add_zero_attn=add_zero_attn,
			self_attention=not self.cross_self_attention,
		)
		self.dlayer_encoder_attn= DLayersEncoderAttention(args)
	def forward(
			self,
			x,
			encoder_out=None,
			encoder_padding_mask=None,
			incremental_state=None,
			prev_self_attn_state=None,
			prev_attn_state=None,
			self_attn_mask=None,
			self_attn_padding_mask=None,
			need_attn=False,
			need_head_weights=False,
			src_word_maps=None,
	):
		"""
		Args:
			x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
			encoder_padding_mask (ByteTensor, optional): binary
				ByteTensor of shape `(batch, src_len)` where padding
				elements are indicated by ``1``.
			need_attn (bool, optional): return attention weights
			need_head_weights (bool, optional): return attention weights
				for each head (default: return average over heads).

		Returns:
			encoded output of shape `(seq_len, batch, embed_dim)`
		"""
		if need_head_weights:
			need_attn = True

		residual = x
		x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
		if prev_self_attn_state is not None:
			if incremental_state is None:
				incremental_state = {}
			prev_key, prev_value = prev_self_attn_state[:2]
			saved_state = {"prev_key": prev_key, "prev_value": prev_value}
			if len(prev_self_attn_state) >= 3:
				saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
			self.self_attn._set_input_buffer(incremental_state, saved_state)

		if self.cross_self_attention and not (
				incremental_state is not None and "prev_key" in self.self_attn._get_input_buffer(incremental_state)):
			if self_attn_mask is not None:
				self_attn_mask = torch.cat((x.new(x.size(0), encoder_out.size(0)).zero_(), self_attn_mask), dim=1)
			if self_attn_padding_mask is not None:
				if encoder_padding_mask is None:
					encoder_padding_mask = self_attn_padding_mask.new(encoder_out.size(1), encoder_out.size(0)).zero_()
				self_attn_padding_mask = torch.cat((encoder_padding_mask, self_attn_padding_mask), dim=1)
			y = torch.cat((encoder_out, x), dim=0)
		else:
			y = x

		x1, attn_weights = self.self_attn(
			query=x,
			key=y,
			value=y,
			key_padding_mask=self_attn_padding_mask,
			incremental_state=incremental_state,
			need_weights=True,
			attn_mask=self_attn_mask,
		)
		x1 = F.dropout(x1, p=self.dropout, training=self.training)
		# x = residual + x

		if self.encoder_attn is not None:
			# residual = x
			# x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
			if prev_attn_state is not None:
				if incremental_state is None:
					incremental_state = {}
				prev_key, prev_value = prev_attn_state[:2]
				saved_state = {"prev_key": prev_key, "prev_value": prev_value}
				if len(prev_attn_state) >= 3:
					saved_state["prev_key_padding_mask"] = prev_attn_state[2]
				self.encoder_attn._set_input_buffer(incremental_state, saved_state)

			x2, attn = self.encoder_attn(
				query=x,
				key=encoder_out,
				value=encoder_out,
				key_padding_mask=encoder_padding_mask,
				incremental_state=incremental_state,
				static_kv=True,
				need_weights=need_attn or (not self.training and self.need_attn),
				need_head_weights=need_head_weights,
			)
			x2 = F.dropout(x2, p=self.dropout, training=self.training)
			# x = residual + x+x2
			x = x1 + residual + x2
		else:
			x = x1 + residual
		x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

		residual = x
		x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
		x = self.activation_fn(self.fc1(x))
		x = F.dropout(x, p=self.activation_dropout, training=self.training)
		x = self.fc2(x)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = residual + x
		x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
		if self.onnx_trace and incremental_state is not None:
			saved_state = self.self_attn._get_input_buffer(incremental_state)
			if self_attn_padding_mask is not None:
				self_attn_state = saved_state["prev_key"], saved_state["prev_value"], saved_state[
					"prev_key_padding_mask"]
			else:
				self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
			return x, attn, self_attn_state
		return x, attn

def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m

class MultihotNumEmbedding(nn.Module):

	def __init__(self, embed_dim, lstm_layers=2, dropout=0.1):
		super().__init__()
		self.val_embed = Embedding(2, embed_dim)
		self.exp_embed = Embedding(2, embed_dim)
		self.sign_embed = Embedding(2, embed_dim)
		self.type_embed = Embedding(2, embed_dim)
		self.embed_rnn = nn.GRU(
			input_size=embed_dim,
			hidden_size=embed_dim,
			num_layers=lstm_layers,
			dropout=dropout if lstm_layers > 1 else 0,
			batch_first=True,
			bidirectional=False,
		)

	def forward(self, multihot_num):
		bsz, src_len, hot_size = multihot_num.size()
		x_type = multihot_num[:, :, 0:1]
		x_sign = multihot_num[:, :, 1:2]
		x_exp = multihot_num[:, :, 2:10]
		x_val = multihot_num[:, :, 10:]
		x_type = self.type_embed(x_type)
		x_sign = self.sign_embed(x_sign)
		x_exp = self.exp_embed(x_exp)
		x_val = self.val_embed(x_val)
		x = torch.cat([x_type, x_sign, x_exp, x_val], dim=2).contiguous()
		# print(x)
		# print(x.size())
		x = x.view(bsz * src_len, hot_size, -1)
		_,x = self.embed_rnn(x)
		# print(x_all.mean(dim=1))
		x = x.view(2,bsz, src_len, -1)
		# print(x.size())
		x = torch.select(x,dim=0,index=x.size(0)-1)
		# print(x.size())
		return x