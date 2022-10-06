export CUDA_VISIBLE_DEVICES=$1
fairseq-train /home/shicm/data-bin-numcase \
	--task joint_numeral_task \
	--user-dir ../src \
	--fp16 -s zh -t en --tensorboard-logdir log --seed 0 --fp16-init-scale 128 \
        --min-loss-scale 0.0001 --optimizer adam --lr-scheduler inverse_sqrt --lr 0.001 \
        --max-tokens 4096 --arch controller --activation-fn relu --dropout 0.1 --encoder-embed-dim 512 \
	--encoder-ffn-embed-dim 2048 --encoder-layers 6 --encoder-attention-heads 8 --decoder-embed-dim 512 \
        --decoder-ffn-embed-dim 2048 --decoder-layers 6 --decoder-attention-heads 8 --share-decoder-input-output-embed \
	--mtmodel-path ~/fp16model/checkpoints/checkpoint_best.pt \
	--criterion joint_loss  \
        --max-epoch 50 --update-freq 2 \
	--save-interval 1 --keep-last-epochs 3 --num-workers 4 \
	--fix-encoder --fix-decoder --fix-embedding --nt-module rule --controller-layer 1 --regression \
	
