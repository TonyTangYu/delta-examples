#~/bin/bash

#1: number of GPUs
#2: Model File Address
#3: BertSquad Data Directory Address
#4: Output Directory Address

NGPU_PER_NODE=4
SQUAD_DIR=/cpfs01/shared/pjlab-lingjun-landmarks/tangyu/SQuAD-explorer/dataset
OUTPUT_DIR=./output
NUM_NODES=1
NGPU=$((NGPU_PER_NODE*NUM_NODES))
EFFECTIVE_BATCH_SIZE=512
MAX_GPU_BATCH_SIZE=12
PER_GPU_BATCH_SIZE=$((EFFECTIVE_BATCH_SIZE/NGPU))
if [[ $PER_GPU_BATCH_SIZE -lt $MAX_GPU_BATCH_SIZE ]]; then
       GRAD_ACCUM_STEPS=1
else
       GRAD_ACCUM_STEPS=$((PER_GPU_BATCH_SIZE/MAX_GPU_BATCH_SIZE))
fi
LR=3e-5
# LR=8e-5
MASTER_PORT=$((NGPU+12345))

JOB_NAME="baseline_${NGPU}GPUs_${EFFECTIVE_BATCH_SIZE}batch_size"
run_cmd="TORCH_SHOW_CPP_STACKTRACES=1 python -m torch.distributed.launch --nproc_per_node=4 nvidia_run_squad_baseline.py \
       --bert_model bert-large-uncased \
       --do_train \
       --do_lower_case \
       --train_file $SQUAD_DIR/train-v1.1.json \
       --predict_file $SQUAD_DIR/dev-v1.1.json \
       --train_batch_size $PER_GPU_BATCH_SIZE \
       --learning_rate ${LR} \
       --num_train_epochs 2.0 \
       --max_seq_length 384 \
       --doc_stride 128 \
       --output_dir $OUTPUT_DIR \
       --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
       --world-size 4 \
       --delta       \
       --budget 20000000000  \
       "
echo ${run_cmd}
eval ${run_cmd}

       # --delta       \
       # --budget 63500000000  \
       #        --delta       \
       # --budget 63800000000 \
       # --delta       \
       # --budget 56000000000  \


       
