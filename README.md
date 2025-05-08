# Weaker LLMs' Opinions Also Matter: Mixture of Opinions Enhances LLM's Mathematical Reasoning
This repo is the source codebase for paper: https://arxiv.org/abs/2502.19622


# baseline: ICL and MoA(single layer), without any training
the test set for gsm8k: https://huggingface.co/datasets/yananchen/gsm8k_shots_8
```bash
# pip install -U bitsandbytes;pip install -U peft
# export MKL_SERVICE_FORCE_INTEL=TRUE

python eval.py --ds yananchen/gsm8k_shots_8 \
        --llm_name meta-llama/Llama-3.1-8B-Instruct \
        --maxlen 4096   
```   

# baseline: vanilla MoA, without any training

this experiment is based on together.ai
gsm8k as test task, rounds = layers

```bash
python moa.py --bench gsm --rounds 2 
``` 


# baseline: vanilla SFT and our method MoO (SFT with mixture of opinions): 

training set for `gsm8k` under SFT: https://huggingface.co/datasets/yananchen/gsm8k_sft
training set for `gsm8k` under MoO: https://huggingface.co/datasets/yananchen/gsm8k_moa 

for other task, replace `gsm8k` with `aqua` and `math`

```bash
trl sft \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name yananchen/gsm8k_sft \
    --report_to "none" \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --output_dir ~/moo/sft_gsm8k_llama31 \
    --logging_steps 1 \
    --num_train_epochs 10 \
    --save_strategy "epoch" \
    --lr_scheduler_type "constant" \
    --max_steps -1 \
    --gradient_checkpointing \
    --logging_strategy "epoch" \
    --packing False \
    --do_eval True \
    --evaluation_strategy 'no' \
    --overwrite_output_dir True \
    --bf16 True \
    --bf16_full_eval True \
    --max_seq_length 4096 \
    --max_length 4096 \
    --eval_accumulation_steps 4 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 16 \
    --save_only_model True \
    --dataset_text_field 'text' \
    --load_in_4bit \
    --attn_implementation 'flash_attention_2'

    # other parameters can be tried
    #--warmup_ratio 0.1
    #--load_in_8bit   
```


# test the model (either from SFT or MoO) saved on local disk
```bash

# SFT
for task in gsm8k aqua math 
do
    python eval.py --ds yananchen/${task}_sft \
            --llm_name meta-llama/Llama-3.1-8B-Instruct \
            --maxlen 4096 \
            --sft_path  ~/moo/sft_${task}_llama31 
done


# MOO
for task in gsm8k aqua math 
do
	python eval.py --ds yananchen/${task}_moa \
	        --llm_name meta-llama/Llama-3.1-8B-Instruct \
	        --maxlen 4096 \
	        --sft_path  ~/moo/sft_${task}_moa_llama31 
done





```