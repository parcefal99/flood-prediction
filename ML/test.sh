# setting hyperparameter ranges 
hidden_sizes=(64 96 128 196 256)
dropouts=(0.0 0.25 0.4 0.5)
batch_sizes=(512 1024 2048)
seq_lengths=(180 270 365)

for hidden_size in "${hidden_sizes[@]}"; do
  for dropout in "${dropouts[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
      for seq_len in "${seq_lengths[@]}"; do
        python run_bash.py --model ealstm --gpu 5 \
        --hidden-size $hidden_size --dropout $dropout \
        --batch-size $batch_size --seq-length $seq_len

        echo "Finished training with: hidden_size=$hidden_size, dropout=$dropout, \
              batch_size=$batch_size, seq_len=$seq_len"
      done
    done
  done
done
