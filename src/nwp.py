import os
import sys
import logging
import torch
import math
from fairseq.tasks import FairseqTask, register_task
from fairseq import options, utils
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.data import (
	data_utils,
	LanguagePairDataset
)

logger = logging.getLogger(__name__)

@register_task("next_word_predict")
class NextWordPredictTask(FairseqTask):
	@staticmethod
	def add_args(parser):
		"""Add task-specific arguments to the parser."""
		# fmt: off
		parser.add_argument('data', help='colon separated path to data directories list, \
							will be iterated upon during epochs in round-robin manner')
		parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
							help='source language')
		parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
							help='target language')
		parser.add_argument('--lazy-load', action='store_true',
							help='load the dataset lazily')
		parser.add_argument('--raw-text', action='store_true',
							help='load raw text dataset')
		parser.add_argument('--load-alignments', action='store_true',
							help='load the binarized alignments')
		parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
							help='pad the source on the left')
		parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
							help='pad the target on the left')
		parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
							help='max number of tokens in the source sequence')
		parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
							help='max number of tokens in the target sequence')
		parser.add_argument('--upsample-primary', default=1, type=int,
							help='amount to upsample primary dataset')
		parser.add_argument('--truncate-source', default=False, action='store_true',
							help='boolean to truncate source to max-source-positions')

	# fmt: on

	def __init__(self, args, src_dict, tgt_dict):
		super().__init__(args)
		self.src_dict = src_dict
		self.tgt_dict = tgt_dict

	@classmethod
	def setup_task(cls, args, **kwargs):
		"""Setup the task (e.g., load dictionaries).

		Args:
			args (argparse.Namespace): parsed command-line arguments
		"""
		args.left_pad_source = options.eval_bool(args.left_pad_source)
		args.left_pad_target = options.eval_bool(args.left_pad_target)
		if getattr(args, 'raw_text', False):
			utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
			args.dataset_impl = 'raw'
		elif getattr(args, 'lazy_load', False):
			utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
			args.dataset_impl = 'lazy'

		paths = args.data.split(':')
		assert len(paths) > 0
		# find language pair automatically
		if args.source_lang is None or args.target_lang is None:
			args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
		if args.source_lang is None or args.target_lang is None:
			raise Exception('Could not infer language pair, please provide it explicitly')

		# load dictionaries
		src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
		tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
		assert src_dict.pad() == tgt_dict.pad()
		assert src_dict.eos() == tgt_dict.eos()
		assert src_dict.unk() == tgt_dict.unk()
		print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
		print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

		return cls(args, src_dict, tgt_dict)

	def load_dataset(self, split, epoch=0, combine=False, **kwargs):
		"""Load a given dataset split.

		Args:
			split (str): name of the split (e.g., train, valid, test)
		"""
		paths = self.args.data.split(':')
		assert len(paths) > 0
		data_path = paths[epoch % len(paths)]

		# infer langcode
		src, tgt = self.args.source_lang, self.args.target_lang

		self.datasets[split] = load_langpair_dataset(
			data_path, split, src, self.src_dict, tgt, self.tgt_dict,
			combine=combine, dataset_impl=self.args.dataset_impl,
			upsample_primary=self.args.upsample_primary,
			left_pad_source=self.args.left_pad_source,
			left_pad_target=self.args.left_pad_target,
			max_source_positions=self.args.max_source_positions,
			max_target_positions=self.args.max_target_positions,
			load_alignments=self.args.load_alignments,
			truncate_source=self.args.truncate_source,
		)

	def build_dataset_for_inference(self, src_tokens, src_lengths,tgt_tokens,tgt_lengths):
		return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary,tgt_tokens,tgt_lengths,self.target_dictionary)
	def build_generator(self, args):
		return NextWordGenerator(self.target_dictionary)
	def max_positions(self):
		"""Return the max sentence length allowed by the task."""
		return (self.args.max_source_positions, self.args.max_target_positions)

	@property
	def source_dictionary(self):
		"""Return the source :class:`~fairseq.data.Dictionary`."""
		return self.src_dict

	@property
	def target_dictionary(self):
		"""Return the target :class:`~fairseq.data.Dictionary`."""
		return self.tgt_dict


class NextWordGenerator(object):
	"""Scores the target for a given source sentence."""

	def __init__(self, tgt_dict, softmax_batch=None):
		self.pad = tgt_dict.pad()
		self.eos = tgt_dict.eos()

	@torch.no_grad()
	def generate(self, models, sample, **kwargs):
		"""Score a batch of translations."""
		model=models[0]
		net_input = sample['net_input']
		model.eval()

		# model.forward normally channels prev_output_tokens into the decoder
		# separately, but SequenceGenerator directly calls model.encoder

		src_tokens = net_input['src_tokens']
		tgt_tokens = sample['target']
		tgt_lengths=(tgt_tokens.ne(self.eos) & tgt_tokens.ne(self.pad)).long().sum(dim=1)
		input_size = src_tokens.size()
		# batch dimension goes first followed by source lengths
		bsz = input_size[0]
		lprobs, attn_scores=model(**net_input)
		lprobs[:,:,self.pad]=-math.inf
		scores,hypos=torch.max(lprobs,dim=2)
		finalized=[]
		for i in range(bsz):
			scores_i=scores[i,:tgt_lengths[i]]
			finalized.append([{
				'tokens': hypos[i,:tgt_lengths[i]],
				'score': sum(scores_i),
				'attention': None,
				'alignment': None,
				'positional_scores': scores_i,
			}])
		return finalized