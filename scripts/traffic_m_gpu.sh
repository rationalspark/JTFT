# This file is licensed under the GNU General Public License (GPL) version 2.0.
# See the LICENSE file or https://www.gnu.org/licenses/gpl-2.0.html for more details.

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

seq_len=512
model_name=JTFT
e_layers=3
e_layers_tfi=0
mod_scal_tfi=0.5
n_heads=16
d_model=128
d_ff=256
dropout=0.2
fc_dropout=0.2
head_dropout=$dropout
stride=8
n_freq=16
n_concat_td=32 #number of TD patches to concat
stride=8
patch_len=16
d_compress_max=1
lr=0.0005
huber_delta=1.0
random_seed=1

root_path_name=./dataset/traffic
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

log_name=./logs/$model_name'_'$model_id_name'_ic'$seq_len'_el'$e_layers'_p'$patch_len'_s'$stride'_d'$d_model'_freq'$n_freq'_cat_td'$n_concat_td.log
echo $log_name

for pred_len in 96 192 336 720       
do
    echo pred_len $pred_len           
    echo "" >>$log_name 
    echo pred_len $pred_len >>$log_name   
    echo "" >>$log_name 

    python -u run_longExp.py \
        --use_huber_loss \
        --huber_delta $huber_delta \
        --use_multi_gpu \
        --devices '0, 1' \
        --e_layers_tfi $e_layers_tfi\
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
        --enc_in 862 \
        --n_freq $n_freq \
        --n_concat_td $n_concat_td \
        --d_compress_max $d_compress_max\
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
        --train_epochs 100\
        --min_epochs 10\
        --num_workers 8 \
        --patience 30\
        --lradj 'TST'\
        --pct_start 0.2\
        --itr 1 --batch_size 16 --learning_rate $lr >>$log_name
        echo " ">>$log_name
done

