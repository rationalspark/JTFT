#!/bin/bash
#SBATCH -J ETTm2
#SBATCH -p cnGPU
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH -o ettm2.log
#SBATCH -e ettm2.log

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi


seq_len=336
model_name=JTFT  #PatchTST #
e_layers=3
n_heads=16
d_model=128 #channel idependent or no patching
d_ff=256
dropout=0.2
fc_dropout=0.2
head_dropout=0.0 #0.2 for channel mixed, 0.0 for channel independent 
patch_len=16 #$stride
stride=8
n_freq=16
n_concat_td=8 #number of TD patches to concat
mod_scal_tfi=0.5
d_compress_max=2

min_epochs=1
train_epochs=100

root_path_name=./dataset/ETT-small
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

random_seed=1
log_name=./logs/$model_name'_'$model_id_name'_ic'$seq_len'_el'$e_layers'_p'$patch_len'_s'$stride'_d'$d_model'_freq'$n_freq'_cat_td'$n_concat_td.log
echo $log_name

#<<COMMENT
for i_rows in {1..10}
do
    echo "  " >>$log_name          
done
echo  Train CI_TFI with routed TST, scale $mod_scal_tfi  >>$log_name
echo "  " >>$log_name   
#COMMENT

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi



random_seed=1
for pred_len in 96 #192 336 720
do
    echo pred_len $pred_len          
    echo "" >>$log_name 
    echo pred_len $pred_len >>$log_name
    echo "" >>$log_name 
    
    python -u run_longExp.py \
        --use_multi_gpu \
        --devices '0, 1' \
        --is_training 1 \
        --decomposition 0 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 7 \
        --n_freq $n_freq \
        --n_concat_td $n_concat_td \
        --d_compress_max $d_compress_max \
        --mod_scal_tfi $mod_scal_tfi \
        --stride $stride \
        --d_model $d_model \
        --e_layers $e_layers \
        --d_ff $d_ff \
        --n_heads $n_heads \
        --dropout $dropout\
        --fc_dropout $fc_dropout\
        --head_dropout $head_dropout \
        --random_seed $random_seed \
        --patch_len $patch_len\
        --des 'Exp' \
        --train_epochs $train_epochs\
        --min_epochs $min_epochs\
        --label_len 1 \
        --num_workers 8 \
        --patience 20\
        --lradj 'TST'\
        --pct_start 0.4 \
        --itr 1 --batch_size 128 --learning_rate 0.0002 >>$log_name  
        for i_rows in {1..5}
        do
            echo "  " >>$log_name          
        done 
done

grep mae $log_name 