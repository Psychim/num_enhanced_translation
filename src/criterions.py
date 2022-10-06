import inspect
import logging
import math
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
# from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
import torch
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy, l1_loss

from .dictionary import CombinedDictionary

from .timer import TicTocTimer

logger = logging.getLogger(__name__)

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduction='sum'):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    # assert not torch.isnan(lprobs).any()
    ninf_mask=~torch.isinf(nll_loss)
    assert not torch.all(~ninf_mask)
    nll_loss=nll_loss[ninf_mask]
    target=target[ninf_mask]
    smooth_loss = -torch.where(torch.isinf(lprobs),torch.full_like(lprobs,0),lprobs).sum(dim=-1, keepdim=True)
    smooth_loss = smooth_loss[ninf_mask]
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduction=='sum':
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    elif reduction=='mean':
        nll_loss=nll_loss.mean()
        smooth_loss=smooth_loss.mean()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss
@register_criterion('numeral_translation_ce')
class NumeralTranslationCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(task)
        self.args=args
        self.eps = args.label_smoothing
        self.src_dict=task.source_dictionary
        self.num_dict=task.number_dictionary
        self.all_dict= CombinedDictionary(task.target_dictionary, task.number_dictionary)
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # print(sample['net_input'])

        encoder_out = model.forward_encoder(sample['net_input'])
        x,controller_x,number_x,value_x,extra = model.decoder(sample['net_input']['tgt_tokens'], encoder_out=encoder_out)
        if number_x is None:
            value_x=number_x=model.translate_num(sample['net_input']['src_tokens'],(extra['attn']+extra['controller_attn'])/2)
        assert number_x is not None
        print(number_x)
        net_output=[x,controller_x,number_x,value_x,extra]
        # net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model,encoder_out, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output


    def compute_loss(self, model,encoder_out, net_output, sample, reduce=True):
        num_output=[net_output[2],net_output[-1]]
        assert not torch.isnan(num_output[0]).any()
        tgt_output=[net_output[0],net_output[-1]]
        con_output=[net_output[1],net_output[-1]]
        value_output=[net_output[3],net_output[-1]]
        attn=net_output[-1]['attn']
        num_probs=num_output[0]
        tgt_probs = model.get_normalized_probs(tgt_output, log_probs=True)
        con_probs = model.get_normalized_probs(con_output, log_probs=True)
        all_probs,_=self.all_dict.combine_probs(tgt_probs,num_probs,con_probs.type_as(tgt_probs))
        # all_probs=all_probs.type_as(tgt_probs)
        all_probs=all_probs.view(-1,all_probs.size(-1))

        target = model.get_targets(sample, net_output)
        # print(target)
        target = target.view(-1, 1)
        if self.args.nt_module == 'neural':
            num_log_probs = model.get_normalized_probs(num_output, log_probs=True)  # The function outputs normalized probs of  net_output[0]
            # print(tgt_probs.size(),num_probs.size())
            
        else:
            num_log_probs=num_output[0]
        num_log_probs = num_log_probs.view(-1, num_log_probs.size(-1))
        num_loss, num_nll_loss = label_smoothed_nll_loss(
                num_log_probs, target, 0, ignore_index=self.padding_idx
         )
        sn=sample['target']
        st=sample['general_target']

        st=st.view(-1,1)
        # print(st.size(),all_probs.size())

        # assert not torch.isnan(num_loss).any()
        st=self.all_dict.map_d1(st)
        # assert not torch.isnan(all_probs).any() and not torch.isinf(all_probs).any()
        all_loss, all_nll_loss=label_smoothed_nll_loss(
            all_probs,st,0,ignore_index=self.all_dict.pad()
        )
        assert attn.size()==sample['align'].size(),'%s %s'%(attn.size(),sample['align'].size())
        loss=num_loss+all_loss#+0.1*align_loss#+multihot_loss#+value_loss+exp_loss

        nll_loss=loss
        return loss, nll_loss


    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

@register_criterion('joint_loss')
class JointNumeralLoss(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(task)
        self.args=args
        self.ntcec=NumeralTranslationCrossEntropyCriterion(args,task)
        self.controller_loss=ControllerLabelSmoothedCrossEntropyCriterion(args,task)
        self.beta=args.beta
        self.timer=[TicTocTimer() for _ in range(3)]
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        NumeralTranslationCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--K', default=1, type=float)
        parser.add_argument('--beta',default=1,type=float)
        # fmt: on
    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""
        # Criterions can override this, but for convenience we also try
        # to automatically map argparse.Namespace keys to corresponding
        # arguments in the __init__.

        return cls(args,task)
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # for name,param in model.named_parameters():
        # 	print(name,param.requires_grad)
        # exit(0)

        self.timer[0].tic()
        encoder_out = model.forward_encoder(sample['net_input'])

        x,controller_x,number_x,value_x,extra = model.decoder(sample['net_input']['tgt_tokens'], encoder_out=encoder_out)
        self.timer[0].toc()
        self.timer[1].tic()
        if number_x is None:
            number_x=model.translate_num(sample['net_input']['src_tokens'],(extra['attn']+extra['controller_attn'])/2)
            value_x=number_x
        self.timer[1].toc()
        # assert number_x is not None
        net_output=[x,controller_x,number_x,value_x,extra]
        # net_output = model(**sample['net_input'])
        self.timer[2].tic()
        loss, nll_loss = self.compute_loss(model,encoder_out, net_output, sample , reduce=reduce)
        self.timer[2].toc()
        # for i in range(3):
        # 	print('timer',i,self.timer[i].get())
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output


    def compute_loss(self, model,encoder_out, net_output, sample, reduce=True):
        number_x=net_output[2]
        loss2,nll_loss2=self.controller_loss.compute_loss(model,net_output,sample,reduce=reduce)
        # print(loss2)
        loss=loss2
        nll_loss=nll_loss2
        # if self.args.nt_module =='neural':
        loss3,nll_loss3=self.ntcec.compute_loss(model,encoder_out,net_output,sample,reduce=reduce)
        loss=loss3+self.beta*loss2
        # print('loss',loss)
        nll_loss=nll_loss3+self.beta*nll_loss2
        return loss, nll_loss


    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }


def weighted_label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True,K=1):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    weights=lprobs.new_zeros((lprobs.size(-1),))
    cnt = target.view(-1).bincount()
    n = cnt.sum()
    for i in range(lprobs.size(-1)):
        if i<cnt.size(0) and cnt[i]!=0:
            v=(n-cnt[i])/cnt[i]+K
            weights[i]=torch.log(v.float())
    lprobs=lprobs*weights.squeeze(0)

    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('controller_label_smoothed_cross_entropy')
class ControllerLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(task)
        self.args=args
        self.eps = args.label_smoothing
        self.K=args.K

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--K',default=1,type=float)

        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        model.controller_mode()
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output


    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output[1:], log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output,True)
        target=target.view(-1,1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps
        )
        return loss, nll_loss


    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }


@register_criterion('mseloss')
class MSELoss(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.pw=args.punish_weight
        self.fw=args.first_weight
        self.learn_decoder=args.learn_decoder
        self.learn_encoder=args.learn_encoder
    @staticmethod
    def add_args(parser):
        parser.add_argument('--poswise',action='store_true')
        parser.add_argument('--punish-weight',type=float,default=0.)
        parser.add_argument('--first-weight',type=float,default=1.)
        parser.add_argument('--learn-decoder',action='store_true')
        parser.add_argument('--learn-encoder',action='store_true')
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output,project_out = model(**sample['net_input'],output_proj=True)
        loss, decoder_loss,encoder_loss = self.compute_loss(model, net_output,project_out, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'decoder_loss': (utils.item(decoder_loss.data) if reduce else decoder_loss.data) if self.learn_decoder else 0,
            'encoder_loss': (utils.item(encoder_loss.data) if reduce else encoder_loss.data) if self.learn_encoder else 0,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output


    def compute_loss(self, model, net_output,project_out, sample, reduce=True):
        net_input=sample['net_input']
        vals = net_output[0]

        target = model.get_targets(sample, net_output)


        # print(vals.size(),target.size())
        if reduce:
            reduction='mean'
        else:
            reduction='none'
        loss = 0
        encoder_loss=0
        encoder_target = target.sum(dim=1).half()
        encoder_target = encoder_target.view(-1)
        # print(encoder_target)
        encoder_vals = project_out
        if self.learn_encoder:
            if self.args.poswise:
                poswise_target=sample['src_label'].view(-1).half()
                # print(poswise_target)
                poswise_vals=encoder_vals.contiguous().view(-1)
                # print(poswise_vals)
                poswise_loss=binary_cross_entropy(poswise_vals,poswise_target,reduction=reduction)
                encoder_loss=poswise_loss
            # print(poswise_loss)

            encoder_sums=(encoder_vals[:,1:]-encoder_vals[:,:-1]).relu().sum(dim=-1).view(-1)
            # print(encoder_sums)
            encoder_loss=encoder_loss+l1_loss(encoder_sums,encoder_target,reduction=reduction)
            # print(encoder_loss)
            loss = loss + encoder_loss
            # print(loss)

        def punish_item(x):
            lth=x.size(1)-1
            if lth<2:
                return 0
            first=x.select(1,0)
            first_p=x.select(1,1)-first.detach()
            p=x.narrow(1,2,lth-1)-x.narrow(1,1,lth-1)
            p=torch.relu(p)
            first_p=torch.relu(first_p)
            p=(first_p+p.sum(1)).mean()
            return p
        decoder_loss=0
        return loss, decoder_loss,encoder_loss


    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs)   if sample_size > 0 else 0.,
            'decoder_loss': sum(log.get('decoder_loss',0)for log in logging_outputs) if sample_size > 0 else 0.,
            'encoder_loss': sum(log.get('encoder_loss',0)for log in logging_outputs) if sample_size > 0 else 0.,
            # 'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs)   if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }