## Small2Large
This repository contains the code for the "[Smaller Language Models are capable of selecting Instruction-Tuning Training Data for Larger Language Models](https://placehold.co/600x400)" paper.

## Usage
Once you have pulled the repo, there are four steps required to follow our method for data selection described in the paper.

1. Cluster your dataset with all-MiniLM-L6-v2 model.
2. Train the model which will select the data from your dataset.
3. Generate selected subsets from the model trained in step above.
4. Train final model on the selected subsets.

*Below, we give an example where we use the Llama-2 7B model to select data from the Alpaca dataset for Llama-2 13B.*

*If you would like to generate the exact subsets used in our paper from our trainings, go to step 3 below.*

### 1. Clustering
```
OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=... python clustering.py --dataset data/alpaca_data.json --num_clusters 1000 --output_name "alpaca"
```
Generates `./clustering/{output_name}.pkl` which contains clustered dataset.

### 2. Train selector model
```
CUDA_VISIBLE_DEVICES=... torchrun --nproc_per_node=... --master_port=... dump_pplxs.py \
    --model_name_or_path /path/to/llama2-7b/ \
    --data_path ./data/alpaca_data.json \
    --val_data_path ./data/alpaca_data.json \
    --bf16 True \
    --output_dir /path/to/output/dir/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --seed 42 \
    --gradient_checkpointing True
```
**Important:** `data_path` and `val_data_path` must be the same dataset!

Generates `./dump-pplxs/epoch-pre.pkl`, `epoch-1.pkl`, `epoch-2.pkl`, and `epoch-3.pkl` which contains the perplexity values before training, after epoch 1, after epoch 2, and after epoch 3, respectively.

### 3. Generate subsets

```
python process_pickle.py \
    --dataset data/alpaca_data.json \
    --epoch0 dump-pplxs/epoch-pre.pkl \
    --epoch1 dump-pplxs/epoch-1.pkl \
    --epoch2 dump-pplxs/epoch-2.pkl \
    --epoch3 dump-pplxs/epoch-3.pkl \
    --clustering clustering/alpaca-clustering.pkl \
    --model_name alpaca-llama2-7b
```

Generates `./l1-rankings/{model_name}-l1_ranking.csv` which contains all perplexity and LP(1) values for data analysis.

*We also provide this CSV for all models talked about in the paper, if you would like to generate the exact subsets used in the paper.*

```
python subset.py \
    --dataset data/alpaca_data.json \
    --l1_ranking_file l1-rankings/alpaca-llama2-7b-l1_ranking.csv \
    --lp1
```

### 4. Train final model with subsets

```
CUDA_VISIBLE_DEVICES=... torchrun --nproc_per_node=... --master_port=... train.py \
    --model_name_or_path /path/to/llama2-13b/ \
    --data_path ./data/10_low_lp1-alpaca-llama2-7b.json \
    --bf16 True \
    --output_dir /path/to/output/dir/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --seed 42 \
    --gradient_checkpointing True
```

Congratulations! You have now trained a Llama-2 13b on only 10% of the original Alpaca dataset which outperforms one trained on the entire dataset!

For any questions, use the `-h` argument for any of the Python scripts or contact Alex Nguyen at atn021@ucsd.edu.