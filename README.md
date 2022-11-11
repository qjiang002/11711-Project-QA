# 11711-Project-QA

## Requirements
* Clone [FiD](https://github.com/facebookresearch/FiD)
* Clone [ConditionalQA](https://github.com/haitian-sun/ConditionalQA.git)
* `pip install transformers==3.0.2`
* `pip install evaluate` which is used in output processing

### With conda
```bash
conda env create -f environment.yml
conda activate qa
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
git clone git@github.com:haitian-sun/ConditionalQA.git
git clone git@github.com:facebookresearch/FiD.git
```

## Data Processing
Convert ConditionalQA data into the format that FiD accepts

## Train Script
```
cd FiD

python train_reader.py \
    --train_data ../Data/without_conditions/train.json \
    --eval_data ../Data/without_conditions/dev.json \
    --use_checkpoint \
    --lr 0.00005 \
    --optim adamw \
    --scheduler linear \
    --weight_decay 0.01 \
    --text_maxlength 250 \
    --total_step 15000 \
    --warmup_step 100 \
    --eval_freq 100 \
    --save_freq 100 \
    --model_size base \
    --per_gpu_batch_size 1 \
    --n_context 50 \
    --name experiment_without_conditions \
    --checkpoint_dir checkpoint
```

### Train with FiD_new
1. T5-base, without condition
```bash
cd FiD_new
rm -r checkpoint_base_wo_cond
python train_reader.py \
    --train_data ../Data/without_conditions/train.json \
    --eval_data ../Data/without_conditions/dev.json \
    --use_checkpoint \
    --lr 0.00005 \
    --optim adamw \
    --scheduler linear \
    --weight_decay 0.01 \
    --text_maxlength 250 \
    --total_step 15000 \
    --warmup_step 100 \
    --eval_freq 100 \
    --save_freq 100 \
    --model_size base \
    --per_gpu_batch_size 1 \
    --n_context 50 \
    --name experiment_without_conditions \
    --accumulation_steps 4 \
    --checkpoint_dir checkpoint_base_wo_cond
```
2. T5-large without condition
```bash
cd FiD_new
rm -r checkpoint_large_wo_cond
python train_reader.py \
    --train_data ../Data/without_conditions/train.json \
    --eval_data ../Data/without_conditions/dev.json \
    --use_checkpoint \
    --lr 0.00005 \
    --optim adamw \
    --scheduler linear \
    --weight_decay 0.01 \
    --text_maxlength 100 \
    --total_step 15000 \
    --warmup_step 100 \
    --eval_freq 100 \
    --save_freq 100 \
    --model_size large \
    --per_gpu_batch_size 1 \
    --n_context 50 \
    --name experiment_without_conditions \
    --accumulation_steps 4 \
    --checkpoint_dir checkpoint_large_wo_cond
```
3. T5-base, with condition
```bash
cd FiD_new
rm -r checkpoint_base_w_cond
python train_reader.py \
    --train_data ../Data/with_conditions/FiD_train_CON_no_title_html.json \
    --eval_data ../Data/with_conditions/FiD_dev_CON_no_title_html.json \
    --use_checkpoint \
    --lr 0.00005 \
    --optim adamw \
    --scheduler linear \
    --weight_decay 0.01 \
    --text_maxlength 250 \
    --total_step 15000 \
    --warmup_step 100 \
    --eval_freq 100 \
    --save_freq 100 \
    --model_size base \
    --per_gpu_batch_size 1 \
    --n_context 50 \
    --name experiment_with_conditions \
    --accumulation_steps 4 \
    --checkpoint_dir checkpoint_base_w_cond
```
4. T5-large with condition
```bash
cd FiD_new
rm -r checkpoint_large_w_cond
python train_reader.py \
    --train_data ../Data/with_conditions/FiD_train_CON_no_title_html.json \
    --eval_data ../Data/with_conditions/FiD_dev_CON_no_title_html.json \
    --use_checkpoint \
    --lr 0.00005 \
    --optim adamw \
    --scheduler linear \
    --weight_decay 0.01 \
    --text_maxlength 150 \
    --total_step 15000 \
    --warmup_step 100 \
    --eval_freq 100 \
    --save_freq 100 \
    --model_size large \
    --per_gpu_batch_size 1 \
    --n_context 50 \
    --name experiment_with_conditions \
    --accumulation_steps 4 \
    --checkpoint_dir checkpoint_large_w_cond
```

## Test Script
```
cd FiD

python test_reader.py \
    --model_path checkpoint/experiment_without_conditions/checkpoint/best_dev \
    --eval_data ../Data/without_conditions/dev.json \
    --per_gpu_batch_size 1 \
    --n_context 50 \
    --name CQA_test_without_conditions \
    --checkpoint_dir checkpoint \
    --write_results
```
Use `--write_results` and the output is written to `final_output.txt`.

## Output Processing and Evaluation
Convert the FiD output to the format that ConditionalQA can use for evaluation
```
python output_processing.py \
    --FiD_output_file FiD/checkpoint/CQA_test_without_conditions/final_output.txt \
    --CQA_output_file Outputs/without_conditions/dev_output.json
```

Evaluate the output with `ConditionalQA/evaluate.py`.
```
cd ConditionalQA

python evaluate.py --pred_file=../Outputs/without_conditions/dev_output.json --ref_file=v1_0/dev.json
```

## Reproduction Results

|       |      Yes/No         || Extractive           || Conditional           || Overall           ||
|-------|:---------:|:---------:|:----------:|:---------:|:-----------:|:--------:|:---------:|:---------:|
|       | answer    | w/ conds  | answer     | w/ conds  | answer      | w/ conds | answer    | w/ conds  |
| Paper | 64.2/64.2 | 48.0/48.0 | 25.2/37.8  | 22.5/33.4 | 45.2/49.7   |  4.7/5.8 | 44.4/50.8 | 35.0/40.6 |
| w/o cond (dev)   | 69.9/69.9 | 48.6/48.6 |  24.2/35.7 | 22.1/32.8 | 56.8/60.0  |  4.3/5.7 | 46.3/51.5 | 34.7/39.5 |
| w/ cond_SEP_title (dev) | 61.5/61.5 | 50.0/50.0 | 24.5/35.0 | 23.2/32.4 | 31.1/34.6 | 2.4/3.8 | 42.2/46.9 | 35.9/40.0 |

## References
[ConditionalQA: A Complex Reading Comprehension Dataset with Conditional Answers.](https://arxiv.org/abs/2110.06884)
[Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://arxiv.org/abs/2007.01282)