hostname
nvidia-smi

python -m methods.contrastive_training.gmm_sm \
            --dataset_name 'cub_ucd' \
            --batch_size 256 \
            --grad_from_block 11 \
            --epochs 60 \
            --base_model vit_dino \
            --num_workers 4 \
            --use_ssb_splits 'True' \
            --use_ucd 'True' \
            --use_sskmeans 'True' \
            --sup_con_weight 0.35 \
            --weight_decay 5e-5 \
            --contrast_unlabel_only 'False' \
            --prop_train_labels 0.5 \
            --transform 'imagenet' \
            --lr 0.1 \
            --print_freq 5 \
            --eval_funcs 'v2'  \
            --exp_root /disk/scratch_fast/bingchen/gmm/ \
            --enable_pcl True \
            --warmup_epochs 1 \
            --enable_proto_pair 'True' \
            --exp_name pcl_debug --evaluate_with_proto 'False'



