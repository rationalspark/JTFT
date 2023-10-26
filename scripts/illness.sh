# This file is licensed under the GNU General Public License (GPL) version 2.0.
# See the LICENSE file or https://www.gnu.org/licenses/gpl-2.0.html for more details.

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=JTFT
e_layers=3
e_layers_tfi=1
n_heads=2
d_model=8
d_ff=12
dropout=0.3
fc_dropout=0.3
head_dropout=0
patch_len=4
stride=2
n_freq=16
n_concat_td=32 #number of TD patches to concat
d_compress_max=1
mod_scal_tfi=0.5
lr=0.0025
seq_len=128
random_seed=1
gpu_id=0
train_epochs=100

root_path_name=./dataset/illness
data_path_name=national_illness.csv
model_id_name=illness
data_name=custom


log_name=./logs/$model_name'_'$model_id_name'_ic'$seq_len'_el'$e_layers'_p'$patch_len'_s'$stride'_d'$d_model'_freq'$n_freq'_cat_td'$n_concat_td.log
echo $log_name

for pred_len in 24 36 48 60
do
    echo pred_len $pred_len    
    echo "" >>$log_name 
    echo pred_len $pred_len >>$log_name    
    echo "" >>$log_name 
    python -u run_longExp.py \
        --use_huber_loss \
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
        --d_compress_max $d_compress_max \
        --n_freq $n_freq \
        --n_concat_td $n_concat_td \
        --mod_scal_tfi $mod_scal_tfi \
        --random_seed $random_seed \
        --e_layers $e_layers \
        --e_layers_tfi $e_layers_tfi \
        --n_heads  $n_heads\
        --d_model $d_model \
        --d_ff $d_ff \
        --dropout $dropout\
        --fc_dropout $fc_dropout\
        --head_dropout $head_dropout\
        --stride $stride\
        --patch_len $patch_len\
        --des 'Exp' \
        --b_not_compile \
        --train_epochs $train_epochs\
        --min_epochs 80\
        --lradj 'constant'\
        --label_len 1 \
        --num_workers 2 \
        --itr 1 --batch_size 64 --learning_rate $lr >>$log_name  
        echo "  " >>$log_name          
done
