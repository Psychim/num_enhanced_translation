from fairseq import utils
from fairseq.modules import MultiheadAttention
import torch
import torch.nn as nn
from torch.nn import functional as F


class DLayersEncoderAttention(nn.Module):
	def __init__(self,args):
		super().__init__()
		self.args=args
		self.num_heads = 1
		self.encoder_sub_attn=MultiheadAttention(
			args.decoder_embed_dim,
			self.num_heads,
			kdim=getattr(args, 'encoder_embed_dim', None),
			vdim=getattr(args, 'encoder_embed_dim', None),
			dropout=args.attention_dropout,
			encoder_decoder_attention=True,
		)
		self.encoder_attn=DebugMultiheadAttention(
			args.decoder_embed_dim,
			args.decoder_attention_heads,
			kdim=getattr(args, 'encoder_embed_dim', None),
			vdim=getattr(args, 'encoder_embed_dim', None),
			dropout=args.attention_dropout,
			encoder_decoder_attention=True,
		)
		self.dropout=getattr(args,'attention_dropout',0.1)

	def forward(self,query,encoder_out,src_word_maps,incremental_state,need_attn,need_head_weights):
		tgt_len=query.size(0)
		src_word_maps = src_word_maps.contiguous()
		bsz,src_len,word_len=src_word_maps.size()
		embed_dim=encoder_out.size(-1)
		idx=src_word_maps.view(bsz,src_len*word_len,1)
		idx=idx.repeat(1,1,embed_dim).transpose(0,1)
		assert(idx.max()<encoder_out.size(0))
		x=torch.cat([encoder_out,encoder_out.new_zeros(1,bsz,embed_dim)],dim=0)
		idx[idx.eq(-1)]=x.size(0)-1
		x = torch.gather( x , dim=0 , index=idx).transpose(0,1)
		x=x.view(bsz,src_len,word_len,x.size(-1))
		x=x.view(bsz,src_len*word_len,-1).transpose(0,1)
		padding_mask=src_word_maps.view(bsz,src_len*word_len).eq(-1)
		assert not torch.isnan(query).any()
		assert not torch.isnan(x).any()
		attn_weights, v = self.encoder_sub_attn(
			query=query,
			key=x,
			value=x,
			key_padding_mask=padding_mask,
			incremental_state=incremental_state,
			static_kv=True,
			before_softmax=True,
		)
		assert not torch.isnan(attn_weights).any()
		assert not torch.isnan(v).any()
		attn_weights=attn_weights.contiguous().view(bsz*self.num_heads,tgt_len,src_len,word_len)
		# if getattr(self.args,'debugging',False):
		# 	print(attn_weights)
		attn_weights_float = utils.softmax(attn_weights.float(), dim=-1)
		# if getattr(self.args,'debugging',False):
		# 	print(attn_weights_float)
		attn_weights_float=attn_weights_float.transpose(1,2).contiguous().view(bsz*self.num_heads*src_len,tgt_len,word_len)
		v=v.contiguous().view(bsz*self.num_heads*src_len,word_len,-1)

		attn_probs = attn_weights_float.type_as(attn_weights)
		attn_probs[torch.all(torch.isnan(attn_probs),dim=-1),:]=0

		v = torch.bmm(attn_probs, v)
		assert not torch.isnan(v).any()
		# v [bsz*self.num_heads*src_len,tgt_len,head_dim]
		v=v.transpose(0,1).contiguous().view(tgt_len,bsz,self.num_heads,src_len,-1)
		v=v.transpose(2,3).contiguous().view(tgt_len,bsz,src_len,-1).transpose(1,2)
		query=query.view(1,tgt_len*bsz,-1)
		v=v.contiguous().view(tgt_len*bsz,src_len,-1).transpose(0,1)
		padding_mask=padding_mask.view(bsz,src_len,word_len)
		encoder_padding_mask=torch.all(padding_mask,dim=-1)
		encoder_padding_mask=encoder_padding_mask.repeat(tgt_len,1,1)
		assert list(encoder_padding_mask.size())==[tgt_len,bsz,src_len]
		encoder_padding_mask=encoder_padding_mask.view(tgt_len*bsz,src_len)
		# print(query.size(),v.size())
		res,attn=self.encoder_attn(
				query=query,
				key=v,
				value=v,
				key_padding_mask=encoder_padding_mask,
				incremental_state=incremental_state,
				need_weights=need_attn or (not self.training and self.need_attn),
				need_head_weights=need_head_weights,
			)
		# print(attn.size())
		res=res.contiguous().view(tgt_len,bsz,-1)
		assert not torch.isnan(res).any()
		attn=attn.contiguous().view(tgt_len,bsz,src_len).transpose(0,1)
		return res,attn


class DebugMultiheadAttention(MultiheadAttention):
	def forward(
			self,
			query, key, value,
			key_padding_mask=None,
			incremental_state=None,
			need_weights=True,
			static_kv=False,
			attn_mask=None,
			before_softmax=False,
			need_head_weights=False,
	):
		"""Input shape: Time x Batch x Channel

		Args:
			key_padding_mask (ByteTensor, optional): mask to exclude
				keys that are pads, of shape `(batch, src_len)`, where
				padding elements are indicated by 1s.
			need_weights (bool, optional): return the attention weights,
				averaged over heads (default: False).
			attn_mask (ByteTensor, optional): typically used to
				implement causal attention, where the mask prevents the
				attention from looking forward in time (default: None).
			before_softmax (bool, optional): return the raw attention
				weights and values before the attention softmax.
			need_head_weights (bool, optional): return the attention
				weights for each head. Implies *need_weights*. Default:
				return the average attention weights over all heads.
		"""
		if need_head_weights:
			need_weights = True
		tgt_len, bsz, embed_dim = query.size()
		assert embed_dim == self.embed_dim
		assert list(query.size()) == [tgt_len, bsz, embed_dim]

		if self.enable_torch_version and not self.onnx_trace and incremental_state is None and not static_kv:
			return F.multi_head_attention_forward(query, key, value,
			                                      self.embed_dim, self.num_heads,
			                                      torch.empty([0]),
			                                      torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
			                                      self.bias_k, self.bias_v,
			                                      self.add_zero_attn, self.dropout,
			                                      self.out_proj.weight, self.out_proj.bias,
			                                      self.training, key_padding_mask, need_weights,
			                                      attn_mask, use_separate_proj_weight=True,
			                                      q_proj_weight=self.q_proj.weight,
			                                      k_proj_weight=self.k_proj.weight,
			                                      v_proj_weight=self.v_proj.weight)

		if incremental_state is not None:
			saved_state = self._get_input_buffer(incremental_state)
			if 'prev_key' in saved_state:
				# previous time steps are cached - no need to recompute
				# key and value if they are static
				if static_kv:
					assert self.encoder_decoder_attention and not self.self_attention
					key = value = None
		else:
			saved_state = None

		if self.self_attention:
			q = self.q_proj(query)
			k = self.k_proj(query)
			v = self.v_proj(query)
		elif self.encoder_decoder_attention:
			# encoder-decoder attention
			q = self.q_proj(query)
			if key is None:
				assert value is None
				k = v = None
			else:
				k = self.k_proj(key)
				v = self.v_proj(key)

		else:
			q = self.q_proj(query)
			k = self.k_proj(key)
			v = self.v_proj(value)
		q *= self.scaling

		if self.bias_k is not None:
			assert self.bias_v is not None
			k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
			v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
			if attn_mask is not None:
				attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
			if key_padding_mask is not None:
				key_padding_mask = torch.cat(
					[key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

		q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
		if k is not None:
			k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
		if v is not None:
			v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

		if saved_state is not None:
			# saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
			if 'prev_key' in saved_state:
				prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
				if static_kv:
					k = prev_key
			if 'prev_value' in saved_state:
				prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
				if static_kv:
					v = prev_value
			key_padding_mask = self._append_prev_key_padding_mask(
				key_padding_mask=key_padding_mask,
				prev_key_padding_mask=saved_state.get('prev_key_padding_mask', None),
				batch_size=bsz,
				src_len=k.size(1),
				static_kv=static_kv,
			)

			saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
			saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)
			saved_state['prev_key_padding_mask'] = key_padding_mask

			self._set_input_buffer(incremental_state, saved_state)

		src_len = k.size(1)

		# This is part of a workaround to get around fork/join parallelism
		# not supporting Optional types.
		if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
			key_padding_mask = None

		if key_padding_mask is not None:
			assert key_padding_mask.size(0) == bsz
			assert key_padding_mask.size(1) == src_len

		if self.add_zero_attn:
			src_len += 1
			k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
			v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
			if attn_mask is not None:
				attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
			if key_padding_mask is not None:
				key_padding_mask = torch.cat(
					[key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)
		attn_weights = torch.bmm(q, k.transpose(1, 2))
		attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
		assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

		if attn_mask is not None:
			attn_mask = attn_mask.unsqueeze(0)
			if self.onnx_trace:
				attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
			attn_weights += attn_mask

		if key_padding_mask is not None:
			# don't attend to padding symbols
			attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
			attn_weights = attn_weights.masked_fill(
				key_padding_mask.unsqueeze(1).unsqueeze(2),
				float('-inf'),
			)
			attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

		if before_softmax:
			return attn_weights, v

		attn_weights_float = utils.softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
		attn_weights = attn_weights_float.type_as(attn_weights)
		attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
		attn = torch.bmm(attn_probs, v)
		assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
		if (self.onnx_trace and attn.size(1) == 1):
			# when ONNX tracing a single decoder step (sequence length == 1)
			# the transpose is a no-op copy before view, thus unnecessary
			attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
		else:
			attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
		attn = self.out_proj(attn)

		if need_weights:
			attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
			if not need_head_weights:
				# average attention weights over heads
				attn_weights = attn_weights.mean(dim=0)
		else:
			attn_weights = None
		return attn, attn_weights

	@staticmethod
	def _append_prev_key_padding_mask(
			key_padding_mask,
			prev_key_padding_mask,
			batch_size,
			src_len,
			static_kv,
	):
		# saved key padding masks have shape (bsz, seq_len)
		if prev_key_padding_mask is not None and static_kv:
			key_padding_mask = prev_key_padding_mask
		# During incremental decoding, as the padding token enters and
		# leaves the frame, there will be a time when prev or current
		# is None
		elif prev_key_padding_mask is not None:
			filler = torch.zeros(batch_size, src_len - prev_key_padding_mask.size(1)).bool()
			if prev_key_padding_mask.is_cuda:
				filler = filler.cuda()
			key_padding_mask = torch.cat((prev_key_padding_mask, filler), dim=1)
		elif key_padding_mask is not None:
			filler = torch.zeros(batch_size, src_len - key_padding_mask.size(1)).bool()
			if key_padding_mask.is_cuda:
				filler = filler.cuda()
			key_padding_mask = torch.cat((filler, key_padding_mask), dim=1)
		return key_padding_mask