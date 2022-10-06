import os
import torch
from fairseq import options, utils
from fairseq.data import data_utils
from fairseq.tasks import register_task, FairseqTask

from .datasets import load_numeral_translation_dataset, load_joint_dataset, NumeralTranslationDataset, \
	load_seqlabel_dataset, SequenceLabelingDataset
from .generator import NumeralTranslationGenerator, RegressionGenerator, SequenceLabelingGenerator
from .dictionary import NumeralDictionary,MultihotNumDictionary
import logging
logger = logging.getLogger(__name__)

MASK = '<mas>'
PREBOS = '<pbos>'
EON='[EON]'



@register_task("joint_numeral_task")
class JointNumeralTask(FairseqTask):
	@staticmethod
	def add_args(parser):
		NumeralTranslationTask.add_args(parser)
		parser.add_argument('--regression', action='store_true')

	def __init__(self, args, nt_task,lbl_dict):
		super().__init__(args)
		self.args=args
		self.nt_task=nt_task
		self.lbl_dict=lbl_dict
	@classmethod
	def setup_task(cls, args, **kwargs):
		paths = args.data.split(':')
		nt_task=NumeralTranslationTask.setup_task(args,**kwargs)
		lbl_dict = cls.load_dictionary(
			os.path.join(paths[0], "dict.{}.txt".format('label'))
		)
		return cls(args,nt_task,lbl_dict)
	def load_dataset(self,split,epoch=1,combine=False,**kwargs):
		"""Load a given dataset split.

				Args:
					split (str): name of the split (e.g., train, valid, test)
				"""
		paths = self.args.data.split(':')
		assert len(paths) > 0
		data_path = paths[epoch % len(paths)]

		# infer langcode
		src, tgt = self.args.source_lang, self.args.target_lang

		self.datasets[split] = load_joint_dataset(
			data_path,
			split,
			src,
			self.nt_task.src_dict,
			tgt,
			self.nt_task.tgt_dict,
			'num',
			self.nt_task.num_dict,
			'label',
			self.lbl_dict,
			combine=combine,
			dataset_impl=self.args.dataset_impl,
			upsample_primary=self.args.upsample_primary,
			left_pad_source=self.args.left_pad_source,
			left_pad_target=self.args.left_pad_target,
			max_source_positions=self.args.max_source_positions,
			max_target_positions=self.args.max_target_positions,
			truncate_source=self.args.truncate_source,
			mask_rate=self.nt_task.mask_rate,
		)
	def build_dataset_for_inference(self, src_tokens, src_lengths):
		return NumeralTranslationDataset(
			src_tokens,
			src_lengths,
			self.src_dict,
			tgt_dict=self.tgt_dict,
			num_dict=self.num_dict,
		)

	def train_step(
			self, sample, model, criterion, optimizer, update_num, ignore_grad=False
	):
		"""
		Do forward and backward, and return the loss as computed by *criterion*
		for the given *model* and *sample*.

		Args:
			sample (dict): the mini-batch. The format is defined by the
				:class:`~fairseq.data.FairseqDataset`.
			model (~fairseq.models.BaseFairseqModel): the model
			criterion (~fairseq.criterions.FairseqCriterion): the criterion
			optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
			update_num (int): the current update
			ignore_grad (bool): multiply loss by 0 if this is set to True

		Returns:
			tuple:
				- the loss
				- the sample size, which is used as the denominator for the
				  gradient
				- logging outputs to display while training
		"""
		model.train()
		model.set_num_updates(update_num)
		# torch.autograd.set_detect_anomaly(True)
		with torch.autograd.profiler.record_function("forward"):
			loss, sample_size, logging_output = criterion(model, sample)
		if ignore_grad:
			loss *= 0
		with torch.autograd.profiler.record_function("backward"):
			optimizer.backward(loss)

			# for name, param in model.named_parameters():
			# 	if param.requires_grad and param.grad is not None:
			# 		print('name', name, 'grad', param.grad.max().item(), param.grad.min().item())
		return loss, sample_size, logging_output


	def build_generator(self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None):
		return NumeralTranslationGenerator(models,self.target_dictionary,self.number_dictionary,
		                                   beam_size=getattr(args, 'beam', 5),
		                                   max_len_a=getattr(args, 'max_len_a', 0),
		                                   max_len_b=getattr(args, 'max_len_b', 200),
		                                   min_len=getattr(args, 'min_len', 1),
		                                   normalize_scores=(not getattr(args, 'unnormalized', False)),
		                                   len_penalty=getattr(args, 'lenpen', 1),
		                                   unk_penalty=getattr(args, 'unkpen', 0),
		                                   temperature=getattr(args, 'temperature', 1.),
		                                   match_source_len=getattr(args, 'match_source_len', False),
		                                   no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
		                                   debugging=getattr(args,'debugging',False)
		                                   )

	@property
	def source_dictionary(self):
		"""Return the source :class:`~fairseq.data.Dictionary`."""
		return self.nt_task.src_dict

	@property
	def target_dictionary(self):
		"""Return the target :class:`~fairseq.data.Dictionary`."""
		return self.nt_task.tgt_dict
	@property
	def number_dictionary(self):
		return self.nt_task.num_dict


@register_task("numeral_translation")
class NumeralTranslationTask(FairseqTask):
	@staticmethod
	def add_args(parser):
		"""Add task-specific arguments to the parser."""
		# fmt: off
		parser.add_argument('data', help='colon separated path to data directories list, \
							will be iterated upon during epochs in round-robin manner; \
							however, valid and test data are always in the first directory to \
							avoid the need for repeating them in all directories')
		parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
		                    help='source language')
		parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
		                    help='target language')
		parser.add_argument('--lazy-load', action='store_true', help='load the dataset lazily')
		parser.add_argument('--raw-text', action='store_true', help='load raw text dataset')
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
		parser.add_argument('--truncate-source', action='store_true', default=False,
		                    help='truncate source to max-source-positions')
		parser.add_argument('--mask-rate', default=0., type=float)

		parser.add_argument('--debugging', action='store_true',default=False)
	def __init__(self, args, src_dict, tgt_dict, num_dict):
		super().__init__(args)
		self.src_dict = src_dict
		self.tgt_dict = tgt_dict
		self.num_dict = num_dict
		self.mask_rate = args.mask_rate

	@classmethod
	def load_num_dictionary(cls, filename):
		"""Load the dictionary from the filename

		Args:
			filename (str): the filename
		"""
		return NumeralDictionary.load(filename)
	@classmethod
	def load_multihotnum_dictionary(cls,filename):
		return MultihotNumDictionary.load(filename)
	@classmethod
	def setup_task(cls, args, **kwargs):
		"""Setup the task (e.g., load dictionaries).

		Args:
			args (argparse.Namespace): parsed command-line arguments
		"""
		args.left_pad_source = options.eval_bool(args.left_pad_source)
		args.left_pad_target = options.eval_bool(args.left_pad_target)
		if getattr(args, 'raw_text', False):
			utils.deprecation_warning('--raw-text is deprecated,please use --dataset-impl=raw')
			args.dataset_impl = 'raw'
		elif getattr(args, 'lazy_load', False):
			utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-imple=lazy')
			args.dataset_impl = 'lazy'
		paths = args.data.split(':')
		assert len(paths) > 0
		# find language pair automatically
		if args.source_lang is None or args.target_lang is None:
			args.source_lang, args.target_lang = data_utils.infer_language_pair(
				paths[0]
			)
		if args.source_lang is None or args.target_lang is None:
			raise Exception(
				"Could not infer language pair, please provide it explicitly"
			)

		# load dictionaries
		src_dict = cls.load_multihotnum_dictionary(
			os.path.join(paths[0], "dict.{}.txt".format(args.source_lang))
		)
		tgt_dict = cls.load_multihotnum_dictionary(
			os.path.join(paths[0], "dict.{}.txt".format(args.target_lang))
		)
		num_dict = cls.load_multihotnum_dictionary(
			os.path.join(paths[0], "dict.{}.txt".format('num'))
		)
		# print("Chinese")
		src_dict.build_multihot_vector()
		# num_dict.build_value_map()
		# print("English")
		num_dict.build_multihot_vector()
		# exit(0)
		tgt_dict.build_multihot_vector()

		assert src_dict.pad() == tgt_dict.pad()
		assert src_dict.eos() == tgt_dict.eos()
		assert src_dict.unk() == tgt_dict.unk()
		logger.info("[{}] dictionary: {} types".format(args.source_lang, len(src_dict)))
		logger.info("[{}] dictionary: {} types".format(args.target_lang, len(tgt_dict)))
		logger.info("[{}] dictionary: {} types".format('num', len(num_dict)))

		return cls(args, src_dict, tgt_dict, num_dict)

	def load_dataset(self, split, epoch=1, combine=False, **kwargs):
		"""Load a given dataset split.

		Args:
			split (str): name of the split (e.g., train, valid, test)
		"""
		paths = self.args.data.split(':')
		assert len(paths) > 0
		data_path = paths[epoch % len(paths)]

		# infer langcode
		src, tgt = self.args.source_lang, self.args.target_lang

		self.datasets[split] = load_numeral_translation_dataset(
			data_path,
			split,
			src,
			self.src_dict,
			tgt,
			self.tgt_dict,
			'num',
			self.num_dict,
			combine=combine,
			dataset_impl=self.args.dataset_impl,
			upsample_primary=self.args.upsample_primary,
			left_pad_source=self.args.left_pad_source,
			left_pad_target=self.args.left_pad_target,
			max_source_positions=self.args.max_source_positions,
			max_target_positions=self.args.max_target_positions,
			truncate_source=self.args.truncate_source,
			mask_rate=self.mask_rate,
		)

	def build_dataset_for_inference(self, src_tokens, src_lengths):
		return NumeralTranslationDataset(
			src_tokens,
			src_lengths,
			self.src_dict,
			tgt_dict=self.tgt_dict,
			num_dict=self.num_dict,
		)

	def build_generator(self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None):
		return NumeralTranslationGenerator(models,self.target_dictionary,self.number_dictionary,
		                                   beam_size=getattr(args, 'beam', 5),
		                                   max_len_a=getattr(args, 'max_len_a', 0),
		                                   max_len_b=getattr(args, 'max_len_b', 200),
		                                   min_len=getattr(args, 'min_len', 1),
		                                   normalize_scores=(not getattr(args, 'unnormalized', False)),
		                                   len_penalty=getattr(args, 'lenpen', 1),
		                                   unk_penalty=getattr(args, 'unkpen', 0),
		                                   temperature=getattr(args, 'temperature', 1.),
		                                   match_source_len=getattr(args, 'match_source_len', False),
		                                   no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
		                                   debugging=getattr(args, 'debugging', False)
		                                   )

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
	@property
	def number_dictionary(self):
		return self.num_dict


@register_task("sequence_labeling")
class SequenceLabelingTask(FairseqTask):
	@staticmethod
	def add_args(parser):
		"""Add task-specific arguments to the parser."""
		# fmt: off
		parser.add_argument('--debugging',action='store_true')
		parser.add_argument('data', help='colon separated path to data directories list, \
							will be iterated upon during epochs in round-robin manner; \
							however, valid and test data are always in the first directory to \
							avoid the need for repeating them in all directories')
		parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
							help='source language')
		parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
							help='target language')
		parser.add_argument('-l', '--label-lang',default=None)
		parser.add_argument('--lazy-load', action='store_true', help='load the dataset lazily')
		parser.add_argument('--raw-text', action='store_true', help='load raw text dataset')
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
		parser.add_argument('--truncate-source', action='store_true', default=False,
							help='truncate source to max-source-positions')
		parser.add_argument('--mask-rate',default=0.,type=float)
		parser.add_argument('--regression',action='store_true')
	def __init__(self, args, src_dict, tgt_dict, lbl_dict):
		super().__init__(args)
		self.src_dict = src_dict
		self.tgt_dict = tgt_dict
		self.lbl_dict = lbl_dict
		self.mask_rate=args.mask_rate

	@classmethod
	def setup_task(cls, args, **kwargs):
		"""Setup the task (e.g., load dictionaries).

		Args:
			args (argparse.Namespace): parsed command-line arguments
		"""
		args.left_pad_source = options.eval_bool(args.left_pad_source)
		args.left_pad_target = options.eval_bool(args.left_pad_target)
		if getattr(args, 'raw_text', False):
			utils.deprecation_warning('--raw-text is deprecated,please use --dataset-impl=raw')
			args.dataset_impl = 'raw'
		elif getattr(args, 'lazy_load', False):
			utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-imple=lazy')
			args.dataset_impl = 'lazy'
		paths = args.data.split(':')
		assert len(paths) > 0
		# find language pair automatically
		if args.source_lang is None or args.target_lang is None:
			args.source_lang, args.target_lang = data_utils.infer_language_pair(
				paths[0]
			)
		if args.source_lang is None or args.target_lang is None:
			raise Exception(
				"Could not infer language pair, please provide it explicitly"
			)

		# load dictionaries
		src_dict = cls.load_multihotnum_dictionary(
			os.path.join(paths[0], "dict.{}.txt".format(args.source_lang))
		)
		tgt_dict = cls.load_multihotnum_dictionary(
			os.path.join(paths[0], "dict.{}.txt".format(args.target_lang))
		)
		lbl_dict = cls.load_dictionary(
			os.path.join(paths[0], "dict.{}.txt".format(args.label_lang))
		)
		src_dict.build_multihot_vector()
		# num_dict.build_value_map()
		# print("English")
		# exit(0)
		tgt_dict.build_multihot_vector()
		assert src_dict.pad() == tgt_dict.pad()
		assert src_dict.eos() == tgt_dict.eos()
		assert src_dict.unk() == tgt_dict.unk()
		logger.info("[{}] dictionary: {} types".format(args.source_lang, len(src_dict)))
		logger.info("[{}] dictionary: {} types".format(args.target_lang, len(tgt_dict)))

		return cls(args, src_dict, tgt_dict, lbl_dict)

	def load_dataset(self, split, epoch=1, combine=False, **kwargs):
		"""Load a given dataset split.

		Args:
			split (str): name of the split (e.g., train, valid, test)
		"""
		paths = self.args.data.split(':')
		assert len(paths) > 0
		data_path = paths[epoch % len(paths)]

		# infer langcode
		src, tgt = self.args.source_lang, self.args.target_lang
		lbl=self.args.label_lang

		self.datasets[split] = load_seqlabel_dataset(
			data_path,
			split,
			src,
			self.src_dict,
			tgt,
			self.tgt_dict,
			lbl,
			self.lbl_dict,
			self.args.regression,
			combine=combine,
			dataset_impl=self.args.dataset_impl,
			upsample_primary=self.args.upsample_primary,
			left_pad_source=self.args.left_pad_source,
			left_pad_target=self.args.left_pad_target,
			max_source_positions=self.args.max_source_positions,
			max_target_positions=self.args.max_target_positions,
			truncate_source=self.args.truncate_source,
			mask_rate=self.mask_rate,
		)

	def build_dataset_for_inference(self, src_tokens, src_lengths, tgt_tokens, tgt_lengths):
		return SequenceLabelingDataset(
			src_tokens,
			src_lengths,
			self.src_dict,
			tgt_tokens,
			tgt_lengths,
			self.tgt_dict,
			lbl_dict=self.lbl_dict
		)
	def build_generator(self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None):
		if args.regression:
			return RegressionGenerator(self.target_dictionary,self.lbl_dict)
		else:
			return SequenceLabelingGenerator(self.tgt_dict)


	def max_positions(self):
		"""Return the max sentence length allowed by the task."""
		return (self.args.max_source_positions, self.args.max_target_positions)
	@classmethod
	def load_multihotnum_dictionary(cls,filename):
		return MultihotNumDictionary.load(filename)
	@property
	def source_dictionary(self):
		"""Return the source :class:`~fairseq.data.Dictionary`."""
		return self.src_dict

	@property
	def number_dictionary(self):
		"""Return the target :class:`~fairseq.data.Dictionary`."""
		return None

	@property
	def target_dictionary(self):
		return self.tgt_dict