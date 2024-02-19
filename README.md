# Smaller Language Models are capable of selecting Instruction-Tuning Training Data for Larger Language Models
This repository contains the <ins>official code</ins> for the "[Smaller Language Models are capable of selecting Instruction-Tuning Training Data for Larger Language Models](https://arxiv.org/abs/2402.10430)" paper.

## Usage
Once you have pulled the repo, there are four steps required to follow our method for data selection described in the paper.

1. Cluster your dataset with all-MiniLM-L6-v2 model.
2. Train the smaller model which will select the data from your dataset.
3. Generate selected subsets from the perplexities of the smaller model trained in step above.
4. Train bigger model on the selected subsets.

*Below, we give an example where we use a smaller Llama-2 7B model to select data from the Alpaca dataset for the bigger Llama-2 13B. If you would like to generate the exact subsets used in our paper from our training runs, go to step 3 below.*

### 1. Clustering
```
OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=... python clustering.py \\
    --dataset data/alpaca_data.json \
    --num_clusters 1000 \
    --output_name "alpaca" \
```
Generates `./clustering/{output_name}.pkl` which contains clustered dataset.

### 2. Train smaller model
```
CUDA_VISIBLE_DEVICES=... torchrun --nproc_per_node=... --master_port=... dump_pplxs.py \
    --model_name_or_path /path/to/smaller/model/aka/llama2-7b/ \
    --data_path ./data/alpaca_data.json \
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

Generates `./dump-pplxs/epoch-pre.pkl`, `epoch-1.pkl`, `epoch-2.pkl`, and `epoch-3.pkl` which contains the perplexity values before training, after epoch 1, after epoch 2, and after epoch 3, respectively.

**Note:** *We also provide pre-computed perplexities in `dump-pplxs/` for all models discussed in the paper, if you would like to generate the exact subsets used in the paper.*

### 3. Generate subsets from smaller model

```
python subset.py \
    --dataset data/alpaca_data.json \
    --clustering clustering/alpaca-clustering.pkl \
    --epoch0 dump-pplxs/epoch-pre.pkl \
    --epoch1 dump-pplxs/epoch-1.pkl \
    --epoch2 dump-pplxs/epoch-2.pkl \
    --epoch3 dump-pplxs/epoch-3.pkl \
    --model_name alpaca-llama2-7b \
    --lp1 \
```

Generates multiple subsets of varying sizes such as `./data/10_low_lp1-{model_name}.json`, which contains 10% of the original dataset.

Passing the `--lp1` argument will generate subsets using the LP(1) metric. Passing the `--lp1_approx` argument will generate subsets using the LP(1) approx metric. Passing the `--clust_rand` argument will generate the clust rand baseline used in the paper. You may also pass any combination of these arguments.

### 4. Train bigger model with subsets

```
CUDA_VISIBLE_DEVICES=... torchrun --nproc_per_node=... --master_port=... train.py \
    --model_name_or_path /path/to/bigger/model/aka/llama2-13b/ \
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

For any questions, use the `-h` argument for any of the Python scripts or contact Alex Nguyen at atn021@ucsd.edu, Dheeraj Mekala at dmekala@ucsd.edu.

## Citation
If you find our work useful, please cite the following:
```
@misc{mekala2024smaller,
      title={Smaller Language Models are capable of selecting Instruction-Tuning Training Data for Larger Language Models}, 
      author={Dheeraj Mekala and Alex Nguyen and Jingbo Shang},
      year={2024},
      eprint={2402.10430},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```