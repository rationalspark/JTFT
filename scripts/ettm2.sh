# This file is licensed under the GNU General Public License (GPL) version 2.0.
# See the LICENSE file or https://www.gnu.org/licenses/gpl-2.0.html for more details.

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi


model_name=JTFT 
e_layers=3
e_layers_tfi=1
n_heads=4
d_model=16
d_ff=24
dropout=0.2
fc_dropout=0.2
head_dropout=0
patch_len=16
stride=8
n_freq=16
n_concat_td=16 #number of TD patches to concat
mod_scal_tfi=0.5
d_compress_max=1
lr=0.0001
huber_delta=1.0

gpu_id=6
min_epochs=1
train_epochs=100

root_path_name=./dataset/ETT-small
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

seq_len=512
random_seed=1

log_name=./logs/$model_name'_'$model_id_name'_ic'$seq_len'_el'$e_layers'_p'$patch_len'_s'$stride'_d'$d_model'_freq'$n_freq'_cat_td'$n_concat_td.log
echo $log_name

for pred_len in 96 #192 336 720
do
    echo pred_len $pred_len          
    echo "" >>$log_name 
    echo pred_len $pred_len >>$log_name
    echo "" >>$log_name 
    
    python -u run_longExp.py \
        --use_huber_loss \
        --huber_delta $huber_delta \
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
        --gpu $gpu_id \
        --enc_in 7 \
        --n_freq $n_freq \
        --n_concat_td $n_concat_td \
        --d_compress_max $d_compress_max \
        --mod_scal_tfi $mod_scal_tfi \
        --stride $stride \
        --d_model $d_model \
        --e_layers $e_layers \
        --e_layers_tfi $e_layers_tfi \
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
        --num_workers 4 \
        --patience 10\
        --lradj 'TST'\
        --pct_start 0.4 \
        --itr 1 --batch_size 128 --learning_rate $lr >>$log_name  
        for i_rows in {1..5}
        do
            echo "  " >>$log_name          
        done 
done

