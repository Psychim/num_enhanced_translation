export CUDA_VISIBLE_DEVICES=$1
fairseq-generate ~/data-bin-$3 --task numeral_translation --user-dir ../src \
	-s zh -t en --gen-subset valid --path checkpoints/checkpoint_$2.pt --results-path $3 --batch-size 10\
