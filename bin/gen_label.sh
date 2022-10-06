export CUDA_VISIBLE_DEVICES=$1
fairseq-generate ~/data-bin-$3 --task sequence_labeling --user-dir ../src \
	-s zh -t en -l label --gen-subset valid --path checkpoints/checkpoint_$2.pt --results-path label-$3 --batch-size 10\
