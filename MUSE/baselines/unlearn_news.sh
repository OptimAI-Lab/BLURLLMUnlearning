
CORPUS="news"
FORGET="../data/$CORPUS/raw/forget.txt"
RETAIN="../data/$CORPUS/raw/retain1.txt"
TARGET_DIR="muse-bench/MUSE-News_target"
LLAMA_DIR="meta-llama/Llama-2-7b-hf"

MAX_LEN=2048
EPOCHS=10
LR='2.5e-5'
PER_DEVICE_BATCH_SIZE=4 # 8 GPUs
GAMA=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for algo in 'BLO_forget_lower_npo_gdr'; do
    python unlearn.py \
        --algo $algo \
        --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
        --data_file $FORGET --retain_data_file $RETAIN \
        --out_dir "/home/mhong/shared/hadir/out_dir/$CORPUS/$algo/$GAMA" \
        --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
        --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
        --gama $GAMA 
done


