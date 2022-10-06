export CUDA_VISIBLE_DEVICES=$1
fairseq-generate ~/data-bin-mt04 --task numeral_translation --user-dir ../src \
	-s zh -t en --gen-subset valid --path checkpoints/checkpoint_$2.pt --results-path test --batch-size 1 --debugging\
