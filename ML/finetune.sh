datasets=(sigma)
# pretrained_models=(us_lstm_nd us_lstm_dis)
pretrained_models=(us_rep_ea us_ea_dis)
periods=(2)
learning_rates=(1 2 3 4)
seeds=(42 29 18)

for dataset in "${datasets[@]}"; do
    for pretrained_model in "${pretrained_models[@]}"; do
        for period in "${periods[@]}"; do
            for learning_rate in "${learning_rates[@]}"; do
                for seed in "${seeds[@]}"; do
                    python finetune.py --gpu 0 --dataset $dataset --pretrained-dir $pretrained_model --period $period --learning-rate $learning_rate --name ealstm_finetune --seed $seed
                    echo "Finished training with: dataset=$dataset, pretrained_model=$pretrained_model, period=$period, learning_rate=$learning_rate, seed=$seed"
                done
            done
        done
    done
done
