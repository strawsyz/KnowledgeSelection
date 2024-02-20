#! /bin/bash
NUM_WORKERS=1
NUM_GPUS_PER_WORKER=6
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="visualglm-6b"
# MODEL_TYPE="./checkpoints/finetune-visualglm-6b-01-28-14-39"
MODEL_ARGS="--max_source_length 64 \
    --max_target_length 256 \
    --lora_rank 10 \
    --layer_range 0 14 \
    --pre_seq_len 4"

# OPTIONS_SAT="SAT_HOME=$1" #"SAT_HOME=/raid/dm/sat_models"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

# few-shot dataset
# train_data="./fewshot-data/dataset.json"
# eval_data="./fewshot-data/dataset.json"

# my cifar-10 dataset
# train_data="./cifar-10/cifar-10.json"
# eval_data="./cifar-10/cifar-10-eval.json"

# frog from cifar-10 dataset
# train_data="./cifar-10/frog/train.json"
# eval_data="./cifar-10/frog/eval.json"

# frog in animals from cifar-10 dataset
# train_data="./cifar-10/frog-animal/train.json"
# eval_data="./cifar-10/frog-animal/eval.json"

# objects from cifar-10 dataset
# train_data="./cifar-10/objects/train.json"
# eval_data="./cifar-10/objects/eval.json"

# animal from cifar-10 dataset
# train_data="./cifar-10/animals/train.json"
# eval_data="./cifar-10/animals/eval.json"

# car dataset
# train_data="./cars/train.json"
# eval_data="./cars/test.json"

# dtd dataset
# train_data="./dtd/train.json"
# eval_data="./dtd/eval.json"

# bird dataset
# train_data="./CUB_200_2011/train.json"
# eval_data="./CUB_200_2011/eval.json"

# airplane dataset
# train_data="./fgvc-aircraft-2013b/train.json"
# eval_data="./fgvc-aircraft-2013b/eval.json"

# five datasets
# train_data="./datasets/five_data.json"
# eval_data="./fgvc-aircraft-2013b/eval.json"

# flower datasets
# train_data="./flowers/train.json"
# eval_data="./flowers/eval.json"

# ISIC datasets
# train_data="./ISIC/train.json"
# eval_data="./ISIC/eval.json"

# food-101 datasets
# train_data="./food-101/train.json"
# eval_data="./food-101/eval.json"

# caltech101 dataset
train_data="./Caltech101/caltech-101/train.json"
eval_data="./Caltech101/caltech-101/eval.json"

# pets dataset
# train_data="./pets/train.json"
# eval_data="./pets/eval.json"

# GTSRB dataset
# train_data="./GTSRB/train.json"
# eval_data="./GTSRB/eval.json"

# # fungi dataset
# train_data="./fungi/train.json"
# eval_data="./fungi/eval.json"

# Caltech256
# train_data="./256_ObjectCategories/train.json"
# eval_data="./256_ObjectCategories/eval.json"

# eleven datasets
# train_data="./datasets/eleven_train.json"
# eval_data="./datasets/eleven_eval.json"

# easy classification datasets
# train_data="./datasets/easy_class_train.json"
# eval_data="./datasets/easy_class_eval.json"

# train_data="./datasets/diff_class_train.json"
# eval_data="./datasets/diff_class_eval.json"

gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 15000 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${train_data} \
       --valid-data ${eval_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --save-interval 1500 \
       --eval-interval 300 \
       --save "./checkpoints" \
       --split 1 \
       --eval-iters 10 \
       --eval-batch-size 16 \
       --zero-stage 1 \
       --lr 0.0001 \
       --batch-size 8 \
       --skip-init \
       --fp16 \
       --use_lora
"
# 
              

run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --include localhost:1,2,3,4,5,6,7 --master_port 16666 --hostfile ${HOST_FILE_PATH} finetune_visualglm.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
