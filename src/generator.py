import math
from typing import Dict, List, Optional
from torch import Tensor
from fairseq import search
import torch
from torch.nn import functional as F
from fairseq.sequence_generator import SequenceGenerator
from .dictionary import CombinedDictionary

MASK = '<mas>'
PREBOS = '<pbos>'
EON='[EON]'

class NumeralTranslationGenerator(SequenceGenerator):
	def __init__(
			self,
			models,
			tgt_dict,
			num_dict,
			beam_size=1,
			max_len_a=0,
			max_len_b=200,
			min_len=1,
			normalize_scores=True,
			len_penalty=1.,
			unk_penalty=0.,
			temperature=1.,
			match_source_len=False,
			no_repeat_ngram_size=0,
			search_strategy=None,
			eos=None,
			symbols_to_strip_from_output=None,
			lm_model=None,
			lm_weight=-1.0,
			debugging=False
	):
		"""Generates translations of a given source sentence.

		Args:
			tgt_dict (~fairseq.data.Dictionary): target dictionary
			beam_size (int, optional): beam width (default: 1)
			max_len_a/b (int, optional): generate sequences of maximum length
				ax + b, where x is the source length
			min_len (int, optional): the minimum length of the generated output
				(not including end-of-sentence)
			normalize_scores (bool, optional): normalize scores by the length
				of the output (default: True)
			len_penalty (float, optional): length penalty, where <1.0 favors
				shorter, >1.0 favors longer sentences (default: 1.0)
			unk_penalty (float, optional): unknown word penalty, where <0
				produces more unks, >0 produces fewer (default: 0.0)
			retain_dropout (bool, optional): use dropout when generating
				(default: False)
			sampling (bool, optional): sample outputs instead of beam search
				(default: False)
			sampling_topk (int, optional): only sample among the top-k choices
				at each step (default: -1)
			sampling_topp (float, optional): only sample among the smallest set
				of words whose cumulative probability mass exceeds p
				at each step (default: -1.0)
			temperature (float, optional): temperature, where values
				>1.0 produce more uniform samples and values <1.0 produce
				sharper samples (default: 1.0)
			diverse_beam_groups/strength (float, optional): parameters for
				Diverse Beam Search sampling
			match_source_len (bool, optional): outputs should match the source
				length (default: False)
		"""

		super().__init__(models,tgt_dict,beam_size,max_len_a,max_len_b,min_len,normalize_scores,len_penalty,unk_penalty,
		                 temperature,match_source_len,no_repeat_ngram_size,search_strategy,eos,symbols_to_strip_from_output,
		                 lm_model,lm_weight)
		self.num_dict=num_dict
		self.all_dict= CombinedDictionary(tgt_dict, num_dict)
		# print('Done')
		self.vocab_size = len(self.all_dict)
		self.pad = self.all_dict.pad()
		self.unk = self.all_dict.unk()
		self.eos = self.all_dict.eos()
		self.bos= self.all_dict.bos()
		self.eon=self.all_dict.index(EON)
		self.num_eon=self.num_dict.index(EON)
		self.num_eos=self.num_dict.eos()
		self.debugging=debugging

		self.search = search.BeamSearch(self.all_dict) if search_strategy is None else search_strategy

		self.should_set_src_lengths = (
				hasattr(self.search, "needs_src_lengths") and self.search.needs_src_lengths
		)

	@torch.no_grad()
	def generate(self, models, sample, **kwargs):
		"""Generate a batch of translations.

		Args:
			models (List[~fairseq.models.FairseqModel]): ensemble of models
			sample (dict): batch
			prefix_tokens (torch.LongTensor, optional): force decoder to begin
				with these tokens
			bos_token (int, optional): beginning of sentence token
				(default: self.eos)
		"""
		model = models[0]
		return self._generate(model, sample, **kwargs)

	@torch.no_grad()
	def _generate(
			self,
			model,
			sample,
			prefix_tokens=None,
			constraints: Optional[Tensor] = None,
			bos_token=None,
			**kwargs
	):
		def throw_finalized(step, cand_scores):
			scores = cand_scores
			if self.normalize_scores:
				scores = scores / ((step + 1) ** self.len_penalty)
			unfi_sent = 0
			# print(step,'finished',finished)
			for sent,f in enumerate(finished):
				if f:
					continue
				# threshold = scores[unfi_sent].min()
				hyp = finalized[sent]
				top_scores=scores[unfi_sent].tolist()+[h['score'] for h in hyp]
				top_scores.sort(reverse=True)
				threshold=top_scores[self.beam_size-1]
				for i, h in enumerate(hyp):
					if h['score'] < threshold:
						del hyp[i]
				unfi_sent += 1
		incremental_states=torch.jit.annotate(
			List[Dict[str,Dict[str, Optional[Tensor]]]],
			torch.jit.annotate(Dict[str,Dict[str,Optional[Tensor]]],{}),

		)
		net_input= sample['net_input']
		if 'src_tokens' in net_input:
			src_tokens = net_input['src_tokens']
			src_lengths = (
				(src_tokens.ne(self.eos)&src_tokens.ne(self.pad)).long().sum(dim=1)
			)
		elif 'source' in net_input:
			src_tokens = net_input['source']
			src_lenghts = (
				net_input['padding_mask'].size(-1)-net_input['padding_mask'].sum(-1)
				if net_input['padding_mask'] is not None
				else torch.tensor(src_tokens.size(-1)).to(src_tokens)
			)
		if self.debugging and 36 not in sample['id']: #,763,849sh]
			return [[{'score':0,'tokens':torch.tensor([self.tgt_dict.eos()]),'alignment':None,'positional_scores':torch.tensor([0.])}]*self.beam_size]*len(sample['id'])

		input_size = src_tokens.size()
		# batch dimension goes first followed by source lengths
		bsz = input_size[0]
		src_len = input_size[1]
		beam_size = self.beam_size
		# print(beam_size)
		if constraints is not None and not self.search.supports_constraints:
			raise NotImplementedError(
				"Target-side constraints were provided, but search method doesn't support them"
			)
		# Initialize constraints, when active
		self.search.init_constraints(constraints, beam_size)

		max_len: int = -1
		if self.match_source_len:
			max_len = src_lengths.max().item()
		else:
			max_len = min(
				int(self.max_len_a * src_len + self.max_len_b),
				# exclude the EOS marker
				model.max_decoder_positions() - 1,
			)

		# compute the encoder output for each beam
		encoder_outs = model.forward_encoder(net_input)
		new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
		new_order = new_order.to(src_tokens.device).long()
		encoder_outs = model.encoder.reorder_encoder_out(encoder_outs, new_order)

		# initialize buffers
		scores = (
			torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
		)  # +1 for eos; pad is never chosen for scoring
		tokens = (
			torch.zeros(bsz * beam_size, max_len + 2)
				.to(src_tokens)
				.long()
				.fill_(self.pad)
		)  # +2 for eos and pad
		tokens[:, 0] = self.eos if bos_token is None else bos_token
		if self.debugging:
    			print(tokens)
		gen_state = src_tokens.new_zeros(bsz*beam_size).bool()
		attn=None

		num_map=self.all_dict.d2map.to(src_tokens.device).long()
		tgt_map=self.all_dict.d1map.to(src_tokens.device).long()
		tgt_backmap=self.all_dict.d1bmap.to(src_tokens.device).long()

		# A list that indicates candidates that should be ignored.
		# For example, suppose we're sampling and have already finalized 2/5
		# samples. Then cands_to_ignore would mark 2 positions as being ignored,
		# so that we only finalize the remaining 3 samples.
		cands_to_ignore = (
			torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
		)  # forward and backward-compatible False mask

		# list of completed sentences
		finalized = torch.jit.annotate(
			List[List[Dict[str, Tensor]]],
			[torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
		)  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

		finished = [
			False for i in range(bsz)
		]  # a boolean array indicating if the sentence at the index is finished or not
		num_remaining_sent = bsz

		# number of candidate hypos per step
		cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

		# offset arrays for converting between different indexing schemes
		bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
		# [0 , beam_size, 2*beam_size, ...,  bsz*beam_size )   [bsz,1]
		cand_offsets = torch.arange(0, cand_size).type_as(tokens)
		#[0,1,2,...,cand_size)
		reorder_state: Optional[Tensor] = None
		batch_idxs: Optional[Tensor] = None

		original_batch_idxs: Optional[Tensor] = None
		if "id" in sample and isinstance(sample["id"], Tensor):
			original_batch_idxs = sample["id"]
		else:
			original_batch_idxs = torch.arange(0, bsz).type_as(tokens)
		src_tokens = torch.index_select(src_tokens, dim=0, index=new_order.view(-1))
		# print(max_len)
		# retrieved_encoder_out=torch.load('saved_encoder_out')
		for step in range(max_len + 1):  # one extra step for EOS marker
			# print(tokens)
			# reorder decoder internal states based on the prev choice of beams
			if reorder_state is not None:
				if batch_idxs is not None:
					# update beam indices to take into account removed sentences
					corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
					reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
					original_batch_idxs=original_batch_idxs[batch_idxs]
				model.reorder_incremental_state(incremental_states,reorder_state)
				encoder_outs = model.encoder.reorder_encoder_out(encoder_outs, reorder_state)
				# print(reorder_state)
				src_tokens = torch.index_select(src_tokens, dim=0, index=reorder_state)
			if self.debugging:
				print(step,'reorder_state',reorder_state)
				print(step,'mode',gen_state)
				print(step,'tokens',tokens[:,:step+1],'\n',self.tgt_dict.string(tokens[:,:step+1]))
			lprobs,controller_probs,num_probs,value_probs, extra = model.forward_decoder(
				tokens[:, :step + 1], encoder_out=encoder_outs, temperature=self.temperature,
				incremental_state=incremental_states
			)

			lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)
			if num_probs is None:
				model.clean_num_translator(incremental_states,step,gen_state)
				num_probs=model.translate_num(src_tokens,extra['attn'],incremental_state=incremental_states,step=step)
				num_probs[num_probs != num_probs] = torch.tensor(-math.inf).to(num_probs)
			num_probs[:, :, self.num_eos] = -math.inf
			avg_attn_scores=extra['attn'].squeeze(1)
			controller_probs=F.log_softmax(controller_probs,dim=-1)
			if self.debugging:
				print(lprobs.max(),lprobs.min(),lprobs.mean())
			lprobs=F.log_softmax(lprobs.float(),dim=-1)     #[bsz,1,vocab_sz]
			if self.debugging:
				print(step,'raw controller_probs',controller_probs)
			controller_probs[gen_state , : , 1] = 0
			controller_probs[gen_state , : , 0] = -math.inf

			controller_probs[:,:,0]=torch.logaddexp(controller_probs[:,:,0],controller_probs[:,:,1]+num_probs[:,:,self.num_eon])
			assert not torch.isinf(controller_probs[:,:,0][~torch.isinf(num_probs[:,:,self.num_eon])]).any()
			num_probs[:,:,self.num_eon]=-math.inf
			transformer_probs=lprobs
			assert not torch.isinf(lprobs).any(), '%s\n%s'%(src_tokens,lprobs[33,0])
			lprobs,gen_mask=self.all_dict.combine_probs(lprobs,num_probs,controller_probs)
			lprobs[:, self.pad] = -math.inf  # never select pad
			lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty
			if step >= max_len:
				lprobs[:, :self.eos] = -math.inf
				lprobs[:, self.eos + 1:] = -math.inf
			# handle prefix tokens (possibly with different lengths)
			if prefix_tokens is not None and step < prefix_tokens.size(1) and step < max_len:
				prefix_toks = torch.index_select(tgt_map, dim=0,index= prefix_tokens[:,step].view(-1))
				prefix_toks=prefix_toks.unsqueeze(-1).repeat(1,beam_size).view(-1)

				prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
				prefix_mask = prefix_toks.ne(self.pad)
				lprobs[prefix_mask] = -math.inf
				lprobs[prefix_mask] = lprobs[prefix_mask].scatter_(
					-1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
				)
				# if prefix includes eos, then we should make sure tokens and
				# scores are the same across all beams
				eos_mask = prefix_toks.eq(self.eos)
				if eos_mask.any():
					# validate that the first beam matches the prefix
					first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[:, 0, 1:step + 1]
					eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
					target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
					assert (first_beam == target_prefix).all()

					def replicate_first_beam(tensor, mask):
						tensor = tensor.view(-1, beam_size, tensor.size(-1))
						tensor[mask] = tensor[mask][:, :1, :]
						return tensor.view(-1, tensor.size(-1))

					# copy tokens, scores and lprobs from the first beam to all beams
					tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim)
					scores = self.replicate_first_beam(scores, eos_mask_batch_dim)
					lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim)
			elif step < self.min_len:
				# minimum length constraint (does not apply if using prefix_tokens)
				lprobs[:, self.eos] = -math.inf
			# print('lprobs.argmax',lprobs.argmax(dim=-1))
			if self.no_repeat_ngram_size > 0:
				# for each beam and batch sentence, generate a list of previous ngrams
				gen_ngrams = [{} for bbsz_idx in range(bsz * beam_size)]
				for bbsz_idx in range(bsz * beam_size):
					gen_tokens = tokens[bbsz_idx].tolist()
					for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
						gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
							gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

			# Record attention scores
			if avg_attn_scores is not None:
				if attn is None:
					attn = torch.empty(
						bsz * beam_size, avg_attn_scores.size(1), max_len + 2
					).to(scores)
				attn[:, :, step + 1].copy_(avg_attn_scores)

			scores = scores.type_as(lprobs)
			eos_bbsz_idx = torch.empty(0).to(
				tokens
			)  # indices of hypothesis ending with eos (finished sentences)
			eos_scores = torch.empty(0).to(
				scores
			)  # scores of hypothesis ending with eos (finished sentences)

			self.search.set_src_lengths(src_lengths)

			if self.no_repeat_ngram_size > 0:
				def calculate_banned_tokens(bbsz_idx):
					# before decoding the next token, prevent decoding of ngrams that have already appeared
					ngram_index = tuple(tokens[bbsz_idx, step + 2 - self.no_repeat_ngram_size:step + 1].tolist())
					return gen_ngrams[bbsz_idx].get(ngram_index, [])

				if step + 2 - self.no_repeat_ngram_size >= 0:
					# no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
					banned_tokens = [calculate_banned_tokens(bbsz_idx) for bbsz_idx in range(bsz * beam_size)]
				else:
					banned_tokens = [[] for bbsz_idx in range(bsz * beam_size)]

				for bbsz_idx in range(bsz * beam_size):
					lprobs[bbsz_idx, banned_tokens[bbsz_idx]] = -math.inf

			cand_scores, cand_indices, cand_beams = self.search.step(
				step,
				lprobs.view(bsz, -1, self.vocab_size),
				scores.view(bsz, beam_size, -1)[:, :, :step],
				tokens[:,:step+1],
				original_batch_idxs,
			)
			if self.debugging:

				print(step,'cand_scores',cand_scores)
				print(step,'normed cand scores',cand_scores/((step + 1) ** self.len_penalty))
			#cand_scores [bsz, k]   lprobs[bsz_id, idx]
			#cand_indices [bsz, k]  idx%vocab_size
			#cand_beams [bsz, k]   idx//vocab_size

			# cand_bbsz_idx contains beam indices for the top candidate
			# hypotheses, with a range of values: [0, bsz*beam_size),
			# and dimensions: [bsz, cand_size]
			cand_bbsz_idx = cand_beams.add(bbsz_offsets)
			if self.debugging:
				print(step,'cand_bbsz_idx',cand_bbsz_idx)
			#cand_bbsz_idx [bsz, k] bsz_idx*beam_size+idx//vocab_size

			# finalize hypotheses that end in eos, except for blacklisted ones
			# or candidates with a score of -inf
			eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
			eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)


			# only consider eos when it's among the top beam_size indices
			eos_bbsz_idx = torch.masked_select(
				cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
			)
			# TODO throw finalized hypos with score lower than current beams
			if step<max_len-1:
				throw_finalized(step,cand_scores)
			finalized_sents: List[int] = []
			if eos_bbsz_idx.numel() > 0:
				eos_scores = torch.masked_select(
					cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
				)

				finalized_sents = self.finalize_hypos(step, eos_bbsz_idx, eos_scores,tokens,scores,finalized,finished,beam_size,attn,src_lengths,max_len)
				if self.debugging and len(finalized_sents):
					print(step,finalized_sents)
				num_remaining_sent -= len(finalized_sents)

			# print('\n'.join(['\t'.join([str(h['score'].item()) for h in hyp]) for hyp in finalized]))
			assert num_remaining_sent >= 0
			if num_remaining_sent == 0:
				# if self.debugging:
					# print(step,'break')
				break
			if self.search.stop_on_max_len and step >= max_len:
				break
			assert step < max_len
			#去掉已经finalize的batch
			if len(finalized_sents) > 0:
				new_bsz = bsz - len(finalized_sents)

				# construct batch_idxs which holds indices of batches to keep for the next pass
				batch_mask = torch.ones(bsz,dtype=torch.bool,device=cand_indices.device)
				batch_mask[finalized_sents] = False
				# TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
				batch_idxs = torch.arange(
					bsz, device=cand_indices.device
				).masked_select(batch_mask)
				self.search.prune_sentences(batch_idxs)

				eos_mask = eos_mask[batch_idxs]
				cand_beams = cand_beams[batch_idxs]
				bbsz_offsets.resize_(new_bsz, 1)
				cand_bbsz_idx = cand_beams.add(bbsz_offsets)
				cand_scores = cand_scores[batch_idxs]
				cand_indices = cand_indices[batch_idxs]
				if prefix_tokens is not None:
					prefix_tokens = prefix_tokens[batch_idxs]
				src_lengths = src_lengths[batch_idxs]
				cands_to_ignore = cands_to_ignore[batch_idxs]

				scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
				tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
				gen_mask=gen_mask.view(bsz,beam_size,-1)[batch_idxs].view(new_bsz*beam_size,-1)


				if attn is not None:
					attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
				bsz = new_bsz
			else:
				batch_idxs = None

			# Set active_mask so that values > cand_size indicate eos or
			# blacklisted hypos and values < cand_size indicate candidate
			# active hypos. After this, the min values per row are the top
			# candidate active hypos.
			eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
			active_mask = torch.add(
				eos_mask.type_as(cand_offsets) * cand_size,
				cand_offsets[: eos_mask.size(1)],
			)
			#active_mask [bsz, cand_size]

			# get the top beam_size active hypotheses, which are just the hypos
			# with the smallest values in active_mask
			new_cands_to_ignore, active_hypos = torch.topk(
				active_mask, k=beam_size, dim=1, largest=False
			)
			#new_blacklist [bsz,beam_size] beam_idx或者beam_idx+cand_size
			#active_hypos [bsz,beam_size] beam_idx

			# update blacklist to ignore any finalized hypos
			cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
			# Make sure there is at least one active item for each sentence in the batch.
			assert (~cands_to_ignore).any(dim=1).all()

			active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
			#[bsz,cand_size] -> [bsz,beam_size]

			active_scores = torch.gather(
				cand_scores, dim=1, index=active_hypos,
			)

			active_bbsz_idx = active_bbsz_idx.view(-1)   #[bsz*beam_size,]
			active_scores = active_scores.view(-1)

			# copy tokens and scores for active hypotheses
			tokens[:, : step + 1] = torch.index_select(
				tokens[:, : step + 1], dim=0, index=active_bbsz_idx
			)
			if self.debugging:
				print(active_bbsz_idx,active_hypos,cand_indices)
			gen_state=torch.index_select(
				gen_mask,dim=0,index=active_bbsz_idx,
			) #[bsz*cand_size,vocab_size]
			new_beam_indices = torch.gather(
				cand_indices, dim=1, index=active_hypos
			)
			gen_state=torch.gather(
				gen_state.view(bsz,-1,self.vocab_size),dim=-1,index=new_beam_indices.unsqueeze(-1)
			).squeeze(-1).view(-1)     #[bsz,cand_size]
			if self.debugging:
				print(step,'gen_state',gen_state)
			new_beam_indices=torch.index_select(tgt_backmap,0,new_beam_indices.view(-1)).view_as(new_beam_indices)
			# Select the next token for each of them
			tokens.view(bsz, beam_size, -1)[:, :, step + 1] = new_beam_indices
			if step > 0:
				scores[:, :step] = torch.index_select(
					scores[:, :step], dim=0, index=active_bbsz_idx
				)
			scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
				cand_scores, dim=1, index=active_hypos
			)

			# Update constraints based on which candidates were selected for the next beam
			self.search.update_constraints(active_hypos)

			# copy attention for active hypotheses
			if attn is not None:
				attn[:, :, : step + 2] = torch.index_select(
					attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
				)

			# reorder incremental state in decoder
			reorder_state = active_bbsz_idx

		# sort by score descending
		for sent in range(len(finalized)):
			scores = torch.tensor(
				[float(elem["score"].item()) for elem in finalized[sent]]
			)
			_, sorted_scores_indices = torch.sort(scores, descending=True)
			finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
			finalized[sent] = torch.jit.annotate(
				List[Dict[str, Tensor]], finalized[sent]
			)
		return finalized




class RegressionGenerator(object):
	def __init__(
			self,
			tgt_dict,
			lbl_dict,
			max_len_a=0,
			max_len_b=200,
			min_len=1,
			normalize_scores=True,
			unk_penalty=0.,
			retain_dropout=False,
	):
		"""Generates translations of a given source sentence.

		Args:
			tgt_dict (~fairseq.data.Dictionary): target dictionary
			beam_size (int, optional): beam width (default: 1)
			max_len_a/b (int, optional): generate sequences of maximum length
				ax + b, where x is the source length
			min_len (int, optional): the minimum length of the generated output
				(not including end-of-sentence)
			normalize_scores (bool, optional): normalize scores by the length
				of the output (default: True)
			len_penalty (float, optional): length penalty, where <1.0 favors
				shorter, >1.0 favors longer sentences (default: 1.0)
			unk_penalty (float, optional): unknown word penalty, where <0
				produces more unks, >0 produces fewer (default: 0.0)
			retain_dropout (bool, optional): use dropout when generating
				(default: False)
			sampling (bool, optional): sample outputs instead of beam search
				(default: False)
			sampling_topk (int, optional): only sample among the top-k choices
				at each step (default: -1)
			sampling_topp (float, optional): only sample among the smallest set
				of words whose cumulative probability mass exceeds p
				at each step (default: -1.0)
			temperature (float, optional): temperature, where values
				>1.0 produce more uniform samples and values <1.0 produce
				sharper samples (default: 1.0)
			diverse_beam_groups/strength (float, optional): parameters for
				Diverse Beam Search sampling
			match_source_len (bool, optional): outputs should match the source
				length (default: False)
		"""
		self.pad = tgt_dict.pad()
		self.bos = tgt_dict.bos()
		self.unk = tgt_dict.unk()
		self.eos = tgt_dict.eos()
		self.lbl_dict=lbl_dict
		self.tgt_dict=tgt_dict
		self.max_len_a = max_len_a
		self.max_len_b = max_len_b
		self.min_len = min_len
		self.normalize_scores = normalize_scores
		self.unk_penalty = unk_penalty
		self.retain_dropout = retain_dropout

	@torch.no_grad()
	def generate(self, models, sample, **kwargs):
		"""Generate a batch of translations.

		Args:
			models (List[~fairseq.models.FairseqModel]): ensemble of models
			sample (dict): batch
			prefix_tokens (torch.LongTensor, optional): force decoder to begin
				with these tokens
			bos_token (int, optional): beginning of sentence token
				(default: self.eos)
		"""
		# model = EnsembleModel(models)
		model=models[0]
		model.controller_mode()
		return self._generate(model, sample, **kwargs)

	@torch.no_grad()
	def _generate(
			self,
			model,
			sample,
			bos_token=None,
			**kwargs
	):
		if not self.retain_dropout:
			model.eval()

		# model.forward normally channels prev_output_tokens into the decoder
		# separately, but SequenceGenerator directly calls model.encoder
		encoder_input=sample['net_input']

		src_tokens = encoder_input['src_tokens']
		tgt_tokens = encoder_input['tgt_tokens']
		# assert tgt_tokens[0][1]==self.bos
		tgt_lengths=(tgt_tokens.ne(self.eos) & tgt_tokens.ne(self.pad)).long().sum(dim=1)
		input_size = src_tokens.size()
		# batch dimension goes first followed by source lengths
		bsz = input_size[0]
		# compute the encoder output for each beam
		encoder_outs = model.forward_encoder(encoder_input)
		encoder_outs,project_outs=model.project_encoder(src_tokens,encoder_outs)
		lprobs, attn_scores=model.forward_decoder(tgt_tokens,encoder_out=encoder_outs)
		scores=lprobs.squeeze(-1)
		hypos=torch.argmax(scores,dim=-1)
		finalized=[]
		for i in range(bsz):
			hypo=hypos[i].tolist()
			hypo=[str(x) for x in hypo]
			hypo=[self.tgt_dict.index(x) for x in hypo]
			hypo=torch.LongTensor(hypo)
			score=F.softmax(scores[i],dim=-1)[:,1]
			# project_out=sample['src_label'][i]
			project_out = project_outs[i]
			finalized.append([{
				'tokens': hypo[:tgt_lengths[i]-1],
				'score': sample['regression_target'][i].sum(),
				'attention': None,
				'alignment': None,
				'positional_scores': score,
			}])
		return finalized


class SequenceLabelingGenerator(object):
	def __init__(
			self,
			tgt_dict,
			max_len_a=0,
			max_len_b=200,
			min_len=1,
			normalize_scores=True,
			unk_penalty=0.,
			retain_dropout=False,
	):
		"""Generates translations of a given source sentence.

		Args:
			tgt_dict (~fairseq.data.Dictionary): target dictionary
			beam_size (int, optional): beam width (default: 1)
			max_len_a/b (int, optional): generate sequences of maximum length
				ax + b, where x is the source length
			min_len (int, optional): the minimum length of the generated output
				(not including end-of-sentence)
			normalize_scores (bool, optional): normalize scores by the length
				of the output (default: True)
			len_penalty (float, optional): length penalty, where <1.0 favors
				shorter, >1.0 favors longer sentences (default: 1.0)
			unk_penalty (float, optional): unknown word penalty, where <0
				produces more unks, >0 produces fewer (default: 0.0)
			retain_dropout (bool, optional): use dropout when generating
				(default: False)
			sampling (bool, optional): sample outputs instead of beam search
				(default: False)
			sampling_topk (int, optional): only sample among the top-k choices
				at each step (default: -1)
			sampling_topp (float, optional): only sample among the smallest set
				of words whose cumulative probability mass exceeds p
				at each step (default: -1.0)
			temperature (float, optional): temperature, where values
				>1.0 produce more uniform samples and values <1.0 produce
				sharper samples (default: 1.0)
			diverse_beam_groups/strength (float, optional): parameters for
				Diverse Beam Search sampling
			match_source_len (bool, optional): outputs should match the source
				length (default: False)
		"""
		self.tgt_dict=tgt_dict
		self.pad = tgt_dict.pad()
		self.bos=tgt_dict.bos()
		self.unk = tgt_dict.unk()
		self.eos = tgt_dict.eos()
		self.vocab_size = len(tgt_dict)
		# the max beam size is the dictionary size - 1, since we never select pad
		self.max_len_a = max_len_a
		self.max_len_b = max_len_b
		self.min_len = min_len
		self.normalize_scores = normalize_scores
		self.unk_penalty = unk_penalty
		self.retain_dropout = retain_dropout

	@torch.no_grad()
	def generate(self, models, sample, **kwargs):
		"""Generate a batch of translations.

		Args:
			models (List[~fairseq.models.FairseqModel]): ensemble of models
			sample (dict): batch
			prefix_tokens (torch.LongTensor, optional): force decoder to begin
				with these tokens
			bos_token (int, optional): beginning of sentence token
				(default: self.eos)
		"""
		# model = EnsembleModel(models)
		model=models[0]
		return self._generate(model, sample, **kwargs)

	@torch.no_grad()
	def _generate(
			self,
			model,
			sample,
			bos_token=None,
			**kwargs
	):
		if not self.retain_dropout:
			model.eval()

		# model.forward normally channels prev_output_tokens into the decoder
		# separately, but SequenceGenerator directly calls model.encoder
		encoder_input=sample['net_input']

		src_tokens = encoder_input['src_tokens']
		tgt_tokens = encoder_input['tgt_tokens']
		if getattr(model.args,'debugging',False) and not model.training:
			print(sample)
			print(src_tokens)
			print(self.tgt_dict.string(tgt_tokens))
		tgt_lengths=tgt_tokens.ne(self.pad).long().sum(dim=1)
		input_size = src_tokens.size()
		# batch dimension goes first followed by source lengths
		bsz = input_size[0]
		# compute the encoder output for each beam
		# encoder_outs = model.forward_encoder(encoder_input)

		_,lprobs,_,_, attn_scores=model(**encoder_input)
		scores,hypos=torch.max(lprobs,dim=2)
		if getattr(model.args,'debugging',False) and not model.training:
			print(lprobs)
		hypos=hypos.long()
		finalized=[]
		for i in range(bsz):
			if getattr(model.args,'debugging',False) and not model.training:
				print(tgt_lengths[i])
			scores_i=scores[i,:tgt_lengths[i]-1]
			hypo=hypos[i,:tgt_lengths[i]-1]
			hypo=hypo.tolist()
			hypo=' '.join(map(str,hypo))
			hypo=self.tgt_dict.encode_line(hypo,append_eos=False,add_if_not_exist=False)
			finalized.append([{
				'tokens': hypo,
				'score': tgt_lengths[i],
				'attention': None,
				'alignment': None,
				'positional_scores': scores_i,
			}])
		return finalized