import itertools
import os
import random

import numpy as np
from fairseq.data import indexed_dataset, data_utils, AppendTokenDataset, TruncateDataset, StripTokenDataset, \
	ConcatDataset, PrependTokenDataset, FairseqDataset
import torch

import logging

logger = logging.getLogger(__name__)

MASK = '<mas>'
PREBOS = '<pbos>'
EON = '[EON]'


def load_numeral_translation_dataset(
		data_path,
		split,
		src,
		src_dict,
		tgt,
		tgt_dict,
		num,
		num_dict,
		combine,
		dataset_impl,
		upsample_primary,
		left_pad_source,
		left_pad_target,
		max_source_positions,
		max_target_positions,
		prepend_bos=False,
		truncate_source=False,
		mask_rate=0.,
):
	def split_exists(split, src, tgt, lang, data_path):
		filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
		return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

	src_datasets = []
	tgt_datasets = []
	num_datasets = []
	for k in itertools.count():
		split_k = split + (str(k) if k > 0 else "")

		# infer langcode
		if split_exists(split_k, src, tgt, src, data_path):
			prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
			if split_exists(split_k, src, num, num, data_path):
				prefix_num = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, num))
			else:
				prefix_num = None
		else:
			if k > 0:
				break
			else:
				raise FileNotFoundError(
					"Dataset not found: {} ({})".format(split, data_path)
				)

		src_dataset = data_utils.load_indexed_dataset(
			prefix + src, src_dict, dataset_impl
		)
		if truncate_source:
			src_dataset = AppendTokenDataset(
				TruncateDataset(
					StripTokenDataset(src_dataset, src_dict.eos()),
					max_source_positions - 1,
				),
				src_dict.eos(),
			)
		src_datasets.append(src_dataset)

		tgt_dataset = data_utils.load_indexed_dataset(
			prefix + tgt, tgt_dict, dataset_impl
		)
		tgt_datasets.append(tgt_dataset)

		if prefix_num is None:
			num_dataset = None
		else:
			num_dataset = data_utils.load_indexed_dataset(
				prefix_num + num, num_dict, dataset_impl
			)
			num_datasets.append(num_dataset)
		logger.info(
			"{} {} {}-{} {} examples".format(
				data_path, split_k, src, tgt, len(src_datasets[-1])
			)
		)

		if not combine:
			break

	assert len(src_datasets) == len(tgt_datasets)

	if len(src_datasets) == 1:
		src_dataset = src_datasets[0]
		tgt_dataset = tgt_datasets[0]
		if num_dataset is not None:
			num_dataset = num_datasets[0]
	else:
		sample_ratios = [1] * len(src_datasets)
		sample_ratios[0] = upsample_primary
		src_dataset = ConcatDataset(src_datasets, sample_ratios)
		tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
		if num_dataset is not None:
			num_dataset = ConcatDataset(num_datasets, sample_ratios)

	if prepend_bos:
		assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
		src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
		tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
	return NumeralTranslationDataset(
		src_dataset,
		src_dataset.sizes,
		src_dict,
		tgt_dataset,
		tgt_dataset.sizes,
		tgt_dict,
		num_dataset,
		num_dataset.sizes if num_dataset is not None else None,
		num_dict,
		left_pad_source=left_pad_source,
		left_pad_target=left_pad_target,
		mask_rate=mask_rate
	)


def load_joint_dataset(
		data_path,
		split,
		src,
		src_dict,
		tgt,
		tgt_dict,
		num,
		num_dict,
		lbl,
		lbl_dict,
		combine,
		dataset_impl,
		upsample_primary,
		left_pad_source,
		left_pad_target,
		max_source_positions,
		max_target_positions,
		prepend_bos=False,
		truncate_source=False,
		mask_rate=0.,
):
	def split_exists(split, src, tgt, lang, data_path):
		filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
		return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

	src_datasets = []
	tgt_datasets = []
	num_datasets = []
	lbl_datasets = []
	for k in itertools.count():
		split_k = split + (str(k) if k > 0 else "")

		# infer langcode
		if split_exists(split_k, src, tgt, src, data_path) and split_exists(split_k, src, num, num, data_path):
			prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
			prefix_num = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, num))
			prefix_lbl = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, lbl))
		# print(prefix_lbl)
		else:
			if k > 0:
				break
			else:
				raise FileNotFoundError(
					"Dataset not found: {} ({})".format(split, data_path)
				)

		src_dataset = data_utils.load_indexed_dataset(
			prefix + src, src_dict, dataset_impl
		)
		if truncate_source:
			src_dataset = AppendTokenDataset(
				TruncateDataset(
					StripTokenDataset(src_dataset, src_dict.eos()),
					max_source_positions - 1,
				),
				src_dict.eos(),
			)
		src_datasets.append(src_dataset)

		tgt_dataset = data_utils.load_indexed_dataset(
			prefix + tgt, tgt_dict, dataset_impl
		)
		tgt_datasets.append(tgt_dataset)
		num_dataset = data_utils.load_indexed_dataset(
			prefix_num + num, num_dict, dataset_impl
		)
		num_datasets.append(num_dataset)
		lbl_dataset = data_utils.load_indexed_dataset(
			prefix_lbl + lbl, lbl_dict, dataset_impl
		)
		lbl_datasets.append(lbl_dataset)
		# print(lbl_dataset)
		logger.info(
			"{} {} {}-{} {} examples".format(
				data_path, split_k, src, tgt, len(src_datasets[-1])
			)
		)

		if not combine:
			break

	assert len(src_datasets) == len(tgt_datasets)

	if len(src_datasets) == 1:
		src_dataset = src_datasets[0]
		tgt_dataset = tgt_datasets[0]
		num_dataset = num_datasets[0]
		lbl_dataset = lbl_datasets[0]
	else:
		sample_ratios = [1] * len(src_datasets)
		sample_ratios[0] = upsample_primary
		src_dataset = ConcatDataset(src_datasets, sample_ratios)
		tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
		num_dataset = ConcatDataset(num_datasets, sample_ratios)
		lbl_dataset = ConcatDataset(lbl_datasets, sample_ratios)
	# print(lbl_dataset[0])
	if prepend_bos:
		assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
		src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
		tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
	return JointTranslationDataset(
		src_dataset,
		src_dataset.sizes,
		src_dict,
		tgt_dataset,
		tgt_dataset.sizes,
		tgt_dict,
		num_dataset,
		num_dataset.sizes,
		num_dict,
		lbl=lbl_dataset,
		lbl_sizes=lbl_dataset.sizes if lbl_dataset is not None else None,
		lbl_dict=lbl_dict,
		left_pad_source=left_pad_source,
		left_pad_target=left_pad_target,
		mask_rate=mask_rate
	)


def collate(
		samples,
		pad_idx,
		eos_idx,
		eon_idx,
		left_pad_source=True,
		left_pad_target=False,
):
	if len(samples) == 0:
		return {}

	def merge(key, left_pad, pad_idx=pad_idx, move_eos_to_beginning=False):
		return data_utils.collate_tokens(
			[s[key] for s in samples],
			pad_idx,
			eos_idx,
			left_pad,
			move_eos_to_beginning,
		)
	id = torch.LongTensor([s["id"] for s in samples])
	src_tokens = merge(
		"source",
		left_pad=left_pad_source,
	)
	src_lengths = torch.LongTensor(
		[s["source"].ne(pad_idx).long().numel() for s in samples]
	)

	src_lengths, sort_order = src_lengths.sort(descending=True)
	id = id.index_select(0, sort_order)
	src_tokens = src_tokens.index_select(0, sort_order)
	src_word_maps=None
	src_labels = None
	if samples[0].get('src_label', None) is not None:
		src_labels = merge('src_label', left_pad=left_pad_source, pad_idx=0)
		src_labels = src_labels.index_select(0, sort_order)
	align=None
	if samples[0].get('align', None) is not None:
		align=collate_2d_tokens(
			[s["align"] for s in samples],
			0,
			None,
			left_pad_source,
			False
		)
		align=align.index_select(0,sort_order)
	ntokens = src_lengths.sum().item()
	tgt_tokens = None
	tgt_lengths = None
	if samples[0].get('target', None) is not None:
		target = merge(
			"target",
			left_pad=left_pad_target,
			move_eos_to_beginning=True
		)
		tgt_tokens = target.index_select(0, sort_order)
		tgt_lengths = torch.LongTensor(
			[s["target"].numel() for s in samples]
		).index_select(0, sort_order)
	num = None
	if samples[0].get('num', None) is not None:
		num = merge("num",
					pad_idx=eon_idx,
		            left_pad=left_pad_target)
		num = num.index_select(0, sort_order)
	batch = {
		"id": id,
		"nsentences": len(samples),
		"ntokens": ntokens,
		"net_input": {
			"src_tokens": src_tokens,
			"src_labels":src_labels,
			"src_lengths": src_lengths,
			"src_word_maps":src_word_maps,
			"tgt_tokens": tgt_tokens,
			# "tgt_lengths": tgt_lengths,

		},
		"target": num,
		"align": align,
	}
	return batch


class NumeralTranslationDataset(FairseqDataset):
	def __init__(self, src, src_sizes, src_dict, tgt=None, tgt_sizes=None, tgt_dict=None,
	             num=None, num_sizes=None, num_dict=None,
	             left_pad_source=True, left_pad_target=False,
	             max_source_positions=1024, max_target_positions=1024,
	             shuffle=True, input_feeding=True,
	             remove_eos_from_source=False, append_eos_to_target=False,
	             mask_rate=0.,
	             append_pbos=False,
	             ):
		assert src_dict.pad() == tgt_dict.pad()
		assert src_dict.eos() == tgt_dict.eos()
		assert src_dict.unk() == tgt_dict.unk()
		assert len(src) == len(tgt), "Source and target must contain the same number of examples"
		if tgt is not None:
			assert len(src) == len(tgt), "Source and label must contain the same number of examples"
		self.src = src
		self.tgt = tgt
		self.num = num
		self.src_sizes = np.array(src_sizes)
		self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
		self.num_sizes = np.array(num_sizes) if num_sizes is not None else None
		self.src_dict = src_dict
		self.tgt_dict = tgt_dict
		self.num_dict = num_dict
		self.left_pad_source = left_pad_source
		self.left_pad_target = left_pad_target
		self.shuffle = shuffle
		self.max_source_positions = max_source_positions
		self.max_target_positions = max_target_positions
		self.input_feeding = input_feeding
		self.remove_eos_from_source = remove_eos_from_source
		self.append_eos_to_target = append_eos_to_target
		self.mask_rate = mask_rate
		self.append_pbos = append_pbos

	# self.append_bos=append_bos
	def __getitem__(self, index):
		def recover_str(itm, dct):
			s = [dct[c] for c in itm]
			for i, c in enumerate(s):
				if c.endswith('@@'):
					s[i] = c[:-2]
			return s
		tgt_item = self.tgt[index] if self.tgt is not None else None
		src_item = self.src[index]
		num_item = self.num[index] if self.num is not None else None
		eon = self.num_dict.index(EON)
		if tgt_item is not None:
			eos = self.tgt_dict.eos()
			if tgt_item[-1] != eos:
				tgt_item = torch.cat([tgt_item, torch.LongTensor([eos])])
				if num_item is not None:
					num_item = torch.cat([num_item, torch.LongTensor([eon])])
		bos = self.tgt_dict.bos()
		if num_item is not None:
			num_eos = self.num_dict.eos()
			if num_item[-1] == num_eos:
				num_item[-1] = eon
		bos = self.src_dict.bos()
		eos = self.src_dict.eos()
		if src_item[-1] != eos:
			src_item = torch.cat([src_item, torch.LongTensor([eos])])
		if self.remove_eos_from_source:
			eos = self.src_dict.eos()
			if src_item[-1] == eos:
				src_item = src_item[:-1]
		src_word_item = [[]]
		for j, idx in enumerate(src_item):
			src_word_item[-1].append(j)
			if not self.src_dict[idx].endswith('@@'):
				src_word_item.append([])
		src_word_item = src_word_item[:-1]
		max_len = max(map(len, src_word_item))
		for x in src_word_item:
			while len(x) < max_len:
				x.append(-1)
		src_word_item = torch.tensor(src_word_item, dtype=src_item.dtype, device=src_item.device)
		align_item=None
		if num_item is not None:
			src=recover_str(src_item,self.src_dict)
			num_tgt=recover_str(num_item,self.num_dict)
			align_item=src_item.new(len(num_tgt),len(src)).fill_(0)
			for i in range(len(num_tgt)):
				if num_item[i] ==eon:
					continue
				for j in range(len(src)):
					if src[j] == num_tgt[i]:
						align_item[i][j]=1
		poswise_lbls = torch.zeros_like(src_item)

		def check_num(s):
			num_list = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '1', '2', '3', '4', '5', '6', '7', '8', '9',
			            '0', '两','零','○','百','千','万','亿','兆']
			for num in num_list:
				if num in s:
					return True
			return False

		for i in range(src_item.size(0)):
			if check_num(self.src_dict[src_item[i]]):
				poswise_lbls[i] = 1
		example = {
			"id": index,
			"source": src_item,
			"src_label":poswise_lbls,
			"source_word": src_word_item,
			"target": tgt_item,
			"num": num_item,
			"align": align_item,
		}
		# print(example,self.num_dict.string(num_item))
		return example

	def __len__(self):
		return len(self.src)

	def collater(self, samples, pad_to_length=None):
		res = collate(
			samples,
			pad_idx=self.src_dict.pad(),
			eos_idx=self.src_dict.eos(),
			eon_idx=self.num_dict.index(EON),
			left_pad_source=self.left_pad_source,
			left_pad_target=self.left_pad_target,
		)
		return res

	def num_tokens(self, index):
		return max(
			self.src_sizes[index],
			self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
			self.num_sizes[index] if self.num_sizes is not None else 0)

	def size(self, index):
		return (
			self.src_sizes[index],
			self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
			self.num_sizes[index] if self.num_sizes is not None else 0)

	def ordered_indices(self):
		if self.shuffle:
			indices = np.random.permutation(len(self)).astype(np.int64)
		else:
			indices = np.arange(len(self), dtype=np.int64)
		indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
		return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

	@property
	def supports_prefetch(self):
		return getattr(self.src, "supports_prefetch", False) and \
		       (self.tgt is None or getattr(self.tgt, "supports_prefetch", False)) and \
		       (self.num is None or getattr(self.num, "support_prefetch", False))

	def prefetch(self, indices):
		self.src.prefetch(indices)
		self.tgt.prefetch(indices)
		self.num.prefetch(indices)


def collate_2d_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
	"""Convert a list of 2d tensors into a padded 3d tensor."""
	size0 = max(v.size(0) for v in values)
	size1 = max(v.size(1) for v in values)
	res = values[0].new(len(values), size0, size1).fill_(pad_idx)

	def copy_tensor(src, dst):
			dst[:src.size(0),:src.size(1)].copy_(src)

	for i, v in enumerate(values):
		copy_tensor(v, res[i][size0 - len(v):] if left_pad else res[i][:len(v)])
	return res


def joint_collate(
		samples,
		pad_idx,
		eos_idx,
		eon_idx,
		left_pad_source=True,
		left_pad_target=False,
):
	if len(samples) == 0:
		return {}

	def merge(key, left_pad, pad_idx=pad_idx, move_eos_to_beginning=False):
		return data_utils.collate_tokens(
			[s[key] for s in samples],
			pad_idx,
			eos_idx,
			left_pad,
			move_eos_to_beginning,
		)
	id = torch.LongTensor([s["id"] for s in samples])
	src_tokens = merge(
		"source",
		left_pad=left_pad_source,
	)
	src_lengths = torch.LongTensor(
		[s["source"].numel() for s in samples]
	)
	src_lengths, sort_order = src_lengths.sort(descending=True)
	id = id.index_select(0, sort_order)
	src_tokens = src_tokens.index_select(0, sort_order)
	ntokens = src_lengths.sum().item()
	src_word_maps=None
	src_labels = None
	if samples[0].get('src_label', None) is not None:
		src_labels = merge('src_label', left_pad=left_pad_source, pad_idx=0)
		src_labels = src_labels.index_select(0, sort_order)
	align=None
	if samples[0].get('align', None) is not None:
		align=collate_2d_tokens(
			[s["align"] for s in samples],
			0,
			None,
			left_pad_source,
			False
		)
		align=align.index_select(0,sort_order)
	# print(src_word_maps)
	tgt_tokens = None
	gtarget = None
	tgt_lengths = None
	if samples[0].get('target', None) is not None:
		target = merge(
			"target",
			left_pad=left_pad_target,
			move_eos_to_beginning=True,
		)
		tgt_tokens = target.index_select(0, sort_order)
		tgt_lengths = torch.LongTensor(
			[s["target"].numel() for s in samples]
		).index_select(0, sort_order)
		gtarget = merge("target", left_pad=left_pad_target)
		gtarget = gtarget.index_select(0, sort_order)
	num = None
	if samples[0].get('num', None) is not None:
		num = merge("num",
					pad_idx=eon_idx,
		            left_pad=left_pad_target)
					
		num = num.index_select(0, sort_order)
	lbl = None
	if samples[0].get("label", None) is not None:
		lbl = merge("label", left_pad=left_pad_target, pad_idx=0)
		lbl = lbl.index_select(0, sort_order)
	batch = {
		"id": id,
		"nsentences": len(samples),
		"ntokens": ntokens,
		"net_input": {
			"src_tokens": src_tokens,
			"src_labels": src_labels,
			"src_word_maps": src_word_maps,
			"src_lengths": src_lengths,
			"tgt_tokens": tgt_tokens,
			# "tgt_lengths": tgt_lengths,

		},
		"target": num,
		"regression_target": lbl,
		"general_target": gtarget,
		"align": align,
	}
	return batch


class JointTranslationDataset(NumeralTranslationDataset):
	def __init__(self, src, src_sizes, src_dict, tgt=None, tgt_sizes=None, tgt_dict=None,
	             num=None, num_sizes=None, num_dict=None,
	             lbl=None, lbl_sizes=None, lbl_dict=None,
	             left_pad_source=True, left_pad_target=False,
	             max_source_positions=1024, max_target_positions=1024,
	             shuffle=True, input_feeding=True,
	             remove_eos_from_source=False, append_eos_to_target=False,
	             mask_rate=0.,
	             append_pbos=False,
	             ):
		super().__init__(src, src_sizes, src_dict, tgt, tgt_sizes, tgt_dict, num, num_sizes, num_dict, left_pad_source,
		                 left_pad_target,
		                 max_source_positions, max_target_positions, shuffle, input_feeding, remove_eos_from_source,
		                 append_eos_to_target, mask_rate, append_pbos)
		self.lbl = lbl
		# print(self.lbl)
		self.lbl_sizes = np.array(lbl_sizes) if lbl_sizes is not None else None
		self.lbl_dict = lbl_dict

	def __getitem__(self, index):
		example = super().__getitem__(index)
		lpad = 0
		lbl_item = self.lbl[index] if self.lbl is not None else None
		if lbl_item is not None:
			# lbl_item = torch.cat([lbl_item, torch.LongTensor([lpad])])
			lbl_item = lbl_item.tolist()
			lbl_item = map(lambda i: self.lbl_dict[i], lbl_item)
			lbl_item = map(lambda x: int(x) if x.isdigit() else 0, lbl_item)
			lbl_item = torch.LongTensor(list(lbl_item))

		example["label"] = lbl_item
		# print(lbl_item.size(),example['target'].size())
		assert (lbl_item.size() == example['target'].size()), "%s %s" % (
		lbl_item, self.tgt_dict.string(example['target']))
		return example

	def collater(self, samples, pad_to_length=None):
		res = joint_collate(
			samples,
			pad_idx=self.src_dict.pad(),
			eos_idx=self.src_dict.eos(),
			eon_idx=self.num_dict.index(EON),
			left_pad_source=self.left_pad_source,
			left_pad_target=self.left_pad_target,
		)
		return res

	def num_tokens(self, index):
		return max(
			self.src_sizes[index],
			self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
			self.num_sizes[index] if self.num_sizes is not None else 0,
			self.lbl_sizes[index] if self.lbl_sizes is not None else 0)

	def size(self, index):
		return (
			self.src_sizes[index],
			self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
			self.num_sizes[index] if self.num_sizes is not None else 0,
			self.lbl_sizes[index] if self.lbl_sizes is not None else 0)

	@property
	def supports_prefetch(self):
		return getattr(self.src, "supports_prefetch", False) and \
		       (self.tgt is None or getattr(self.tgt, "supports_prefetch", False)) and \
		       (self.num is None or getattr(self.num, "support_prefetch", False)) and \
		       (self.lbl is None or getattr(self.lbl, "support_prefetch", False))

	def prefetch(self, indices):
		self.src.prefetch(indices)
		self.tgt.prefetch(indices)
		self.num.prefetch(indices)
		self.lbl.prefetch(indices)


def load_seqlabel_dataset(
		data_path,
		split,
		src,
		src_dict,
		tgt,
		tgt_dict,
		lbl,
		lbl_dict,
		is_regression,
		combine,
		dataset_impl,
		upsample_primary,
		left_pad_source,
		left_pad_target,
		max_source_positions,
		max_target_positions,
		prepend_bos=False,
		truncate_source=False,
		mask_rate=0.,
):
	def split_exists(split, src, tgt, lang, data_path):
		filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
		return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

	src_datasets = []
	tgt_datasets = []
	lbl_datasets = []
	for k in itertools.count():
		split_k = split + (str(k) if k > 0 else "")

		# infer langcode
		if split_exists(split_k, src, tgt, src, data_path):
			prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
			
		elif split_exists(split_k, tgt, src, src, data_path):
			prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
		else:
			if k > 0:
				break
			else:
				raise FileNotFoundError(
					"Dataset not found: {} ({})".format(split, data_path)
				)

		src_dataset = data_utils.load_indexed_dataset(
			prefix + src, src_dict, dataset_impl
		)
		if truncate_source:
			src_dataset = AppendTokenDataset(
				TruncateDataset(
					StripTokenDataset(src_dataset, src_dict.eos()),
					max_source_positions - 1,
				),
				src_dict.eos(),
			)
		src_datasets.append(src_dataset)

		tgt_dataset = data_utils.load_indexed_dataset(
			prefix + tgt, tgt_dict, dataset_impl
		)
		tgt_datasets.append(tgt_dataset)
		if split_exists(split_k, src, lbl, lbl, data_path):
			prefix_label = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, lbl))
			lbl_dataset = data_utils.load_indexed_dataset(
				prefix_label + lbl, lbl_dict, dataset_impl
			)
			lbl_datasets.append(lbl_dataset)
		logger.info(
			"{} {} {}-{} {} examples".format(
				data_path, split_k, src, tgt, len(src_datasets[-1])
			)
		)

		if not combine:
			break

	assert len(src_datasets) == len(tgt_datasets)
	lbl_dataset=None
	if len(src_datasets) == 1:
		src_dataset = src_datasets[0]
		tgt_dataset = tgt_datasets[0]
		if lbl_datasets:
			lbl_dataset = lbl_datasets[0]
	else:
		sample_ratios = [1] * len(src_datasets)
		sample_ratios[0] = upsample_primary
		src_dataset = ConcatDataset(src_datasets, sample_ratios)
		tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
		if lbl_datasets:
			lbl_dataset = ConcatDataset(lbl_datasets, sample_ratios)

	if prepend_bos:
		assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
		src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
		tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
	return SequenceLabelingDataset(
		src_dataset,
		src_dataset.sizes,
		src_dict,
		tgt_dataset,
		tgt_dataset.sizes,
		tgt_dict,
		lbl_dataset,
		lbl_dataset.sizes if lbl_dataset else None,
		lbl_dict,
		is_regression,
		left_pad_source=left_pad_source,
		left_pad_target=left_pad_target,
		mask_rate=mask_rate
	)


def labelling_collate(
		samples,
		pad_idx,
		eos_idx,
		left_pad_source=True,
		left_pad_target=False,
):
	if len(samples) == 0:
		return {}

	def merge(key, left_pad, pad_idx=pad_idx, move_eos_to_beginning=False):
		return data_utils.collate_tokens(
			[s[key] for s in samples],
			pad_idx,
			eos_idx,
			left_pad,
			move_eos_to_beginning,
		)
	id = torch.LongTensor([s["id"] for s in samples])
	src_tokens = merge(
		"source",
		left_pad=left_pad_source,
	)
	src_lengths = torch.LongTensor(
		[s["source"].numel() for s in samples]
	)
	src_lengths, sort_order = src_lengths.sort(descending=True)
	id = id.index_select(0, sort_order)
	src_tokens = src_tokens.index_select(0, sort_order)
	target = merge(
		"target",
		left_pad=left_pad_target,
		move_eos_to_beginning=True
	)
	tgt_tokens = target.index_select(0, sort_order)
	tgt_lengths = torch.LongTensor(
		[s["target"].numel() for s in samples]
	).index_select(0, sort_order)
	ntokens = tgt_lengths.sum().item()
	label = None
	if samples[0].get('label', None) is not None:
		label = merge(
			"label",
			left_pad=left_pad_target,
		)
		label = label.index_select(0, sort_order)
	r_label = None
	if samples[0].get('regression_label', None) is not None:
		r_label = merge("regression_label",
		                left_pad=left_pad_target, pad_idx=0)
		r_label = r_label.index_select(0, sort_order)
	src_label = None
	if samples[0].get('src_label', None) is not None:
		src_label = merge('src_label', left_pad=left_pad_source, pad_idx=0)
		src_label = src_label.index_select(0, sort_order)

	batch = {
		"id": id,
		"nsentences": len(samples),
		"ntokens": ntokens,
		"net_input": {
			"src_tokens": src_tokens,
			"src_lengths": src_lengths,
			"tgt_tokens": tgt_tokens,
			# "tgt_lengths":tgt_lengths,
		},
		"target": label,
		"regression_target": r_label,
		"src_label": src_label
	}
	return batch


class SequenceLabelingDataset(FairseqDataset):
	def __init__(self, src, src_sizes, src_dict, tgt, tgt_sizes, tgt_dict, lbl=None, lbl_sizes=None, lbl_dict=None,
	             is_regression=False,
	             left_pad_source=True, left_pad_target=False,
	             max_source_positions=1024, max_target_positions=1024,
	             shuffle=True, input_feeding=True,
	             remove_eos_from_source=False, append_eos_to_target=False,
	             mask_rate=0.,
	             append_bos=True,
	             ):
		assert src_dict.pad() == tgt_dict.pad()
		assert src_dict.pad() == lbl_dict.pad()
		assert src_dict.eos() == tgt_dict.eos()
		assert src_dict.unk() == tgt_dict.unk()
		assert len(src) == len(tgt), "Source and target must contain the same number of examples"
		if lbl is not None:
			assert len(src) == len(lbl), "Source and label must contain the same number of examples"
		self.src = src
		self.tgt = tgt
		self.lbl = lbl
		self.src_sizes = np.array(src_sizes)
		self.tgt_sizes = np.array(tgt_sizes)
		self.lbl_sizes = np.array(lbl_sizes) if lbl_sizes is not None else None
		self.src_dict = src_dict
		self.tgt_dict = tgt_dict
		self.lbl_dict = lbl_dict
		self.left_pad_source = left_pad_source
		self.left_pad_target = left_pad_target
		self.shuffle = shuffle
		self.max_source_positions = max_source_positions
		self.max_target_positions = max_target_positions
		self.input_feeding = input_feeding
		self.remove_eos_from_source = remove_eos_from_source
		self.append_eos_to_target = append_eos_to_target
		self.mask_rate = mask_rate
		self.is_regression = is_regression
		self.append_bos = append_bos

	# self.append_bos=append_bos
	def __getitem__(self, index):
		tgt_item = self.tgt[index]
		for i in range(len(tgt_item)):
			if random.random() <= self.mask_rate:
				tgt_item[i] = self.tgt_dict.unk_index
		src_item = self.src[index]
		lbl_item = self.lbl[index] if self.lbl is not None else None
		r_lbl_item = None
		lpad = self.lbl_dict.pad()
		if self.append_eos_to_target:
			eos = self.tgt_dict.eos()
			if self.tgt[index][-1] != eos:
				tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])
				if lbl_item is not None:
					lbl_item = torch.cat([self.lbl[index], torch.LongTensor([lpad])])
		if lbl_item is not None:
			if self.is_regression:
				r_lbl_item = lbl_item.tolist()
				r_lbl_item = map(lambda i: self.lbl_dict[i], r_lbl_item)
				r_lbl_item = map(lambda x: int(x) if x.isdigit() else 0, r_lbl_item)
				r_lbl_item = torch.LongTensor(list(r_lbl_item))

		if self.remove_eos_from_source:
			eos = self.src_dict.eos()
			if self.src[index][-1] == eos:
				src_item = self.src[index][:-1]
		poswise_lbls = torch.zeros_like(src_item)

		def check_num(s):
			num_list = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '1', '2', '3', '4', '5', '6', '7', '8', '9',
			            '0', '两','零','○','百','千','万','亿','兆']
			for num in num_list:
				if num in s:
					return True
			return False

		for i in range(src_item.size(0)):
			if check_num(self.src_dict[src_item[i]]):
				poswise_lbls[i] = 1
		example = {
			"id": index,
			"source": src_item,
			"target": tgt_item,
			"label": lbl_item,
			"regression_label": r_lbl_item,
			"src_label": poswise_lbls
		}
		return example

	def __len__(self):
		return len(self.src)

	def collater(self, samples, pad_to_length=None):
		res = labelling_collate(
			samples,
			pad_idx=self.src_dict.pad(),
			eos_idx=self.src_dict.eos(),
			left_pad_source=self.left_pad_source,
			left_pad_target=self.left_pad_target,
		)
		return res

	def num_tokens(self, index):
		return max(self.src_sizes[index], self.tgt_sizes[index],
		           self.lbl_sizes[index] if self.lbl_sizes is not None else 0)

	def size(self, index):
		return (
			self.src_sizes[index], self.tgt_sizes[index], self.lbl_sizes[index] if self.lbl_sizes is not None else 0)

	def ordered_indices(self):
		if self.shuffle:
			indices = np.random.permutation(len(self)).astype(np.int64)
		else:
			indices = np.arange(len(self), dtype=np.int64)
		indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
		return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

	@property
	def supports_prefetch(self):
		return getattr(self.src, "supports_prefetch", False) and \
		       getattr(self.tgt, "supports_prefetch", False) and \
		       (self.lbl is None or getattr(self.lbl, "support_prefetch", False))

	def prefetch(self, indices):
		self.src.prefetch(indices)
		self.tgt.prefetch(indices)
		if self.lbl is not None:
			self.lbl.prefetch(indices)
