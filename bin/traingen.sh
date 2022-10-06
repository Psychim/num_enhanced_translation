export CUDA_VISIBLE_DEVICES=$1
fairseq-generate ~/data-bin-numcase --task numeral_translation --user-dir ../src \
	-s zh -t en --gen-subset train --path checkpoints/checkpoint_$2.pt --results-path numcase --batch-size 10\
