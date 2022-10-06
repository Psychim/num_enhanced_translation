prefix=~/data
fairseq-preprocess -s zh -t en --trainpref $prefix/train.context.bpe --validpref $prefix/mt03.numcase.bpe --destdir data-bin-context --srcdict data-bin-numcase/dict.zh.txt --tgtdict data-bin-numcase/dict.en.txt
fairseq-preprocess -s zh -t label --trainpref $prefix/train.context.bpe --validpref $prefix/mt03.numcase.bpe --destdir data-bin-context --srcdict data-bin-numcase/dict.zh.txt --tgtdict data-bin-numcase/dict.label.txt
fairseq-preprocess -s zh -t num --trainpref $prefix/train.context.bpe --validpref $prefix/mt03.numcase.bpe --destdir data-bin-context --srcdict data-bin-numcase/dict.zh.txt --tgtdict data-bin-numcase/dict.num.txt
