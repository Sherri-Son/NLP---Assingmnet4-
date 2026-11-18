# DS-GA 1011 Assignment 4: Fine-tuning Language Models

Sherri (Chaeri) Son (cs8607), Fall 2026

Code for both parts, run on Google Colab (T4 / L4 GPU) with Colab's preinstalled
`torch` / `transformers`, plus `pip install evaluate datasets nltk wandb sentencepiece`.

## Part 1: BERT sentiment classification (`part1/`)

Files I edited:
- `main.py`
  - `do_train`: training loop (fp16 mixed precision with `autocast` + `GradScaler`,
    per-batch padding trimming, gradient clipping at 1.0)
  - `create_augmented_dataloader`: full training set + 5,000 transformed training examples
  - dataset name `imdb` -> `stanfordnlp/imdb` (required by newer `huggingface_hub`)
- `utils.py`
  - `custom_transform`: WordNet synonym replacement (words longer than 3 letters,
    probability 0.6, first WordNet sense, capitalized words skipped)

Commands:
```
python3 main.py --train --eval                                  # Q1  -> out_original.txt
python3 main.py --eval_transformed                              # Q2  -> out_transformed.txt
python3 main.py --train_augmented --eval_transformed            # Q3  -> out_augmented_transformed.txt
python3 main.py --eval --model_dir ./out_augmented              # Q3  -> out_augmented_original.txt
```

## Part 2: T5 text-to-SQL (`part2/`)

Files I edited:
- `load_data.py`: `T5Dataset`, `normal_collate_fn`, `test_collate_fn`
  (decoder start token `<extra_id_0>`; SQL `<=` / `<` replaced by `le_op` / `lt_op`,
  because the T5 vocabulary has no `<`)
- `t5_utils.py`: `initialize_model`, `save_model`, `load_model_from_checkpoint`
- `train_t5.py`: `eval_epoch`, `test_inference`, `postprocess_sql` (turns placeholders back
  into `<=` / `<`), and a `--eval_every` option
- `utils.py`: database execution timeout raised from 120 s (heavy multi-join queries were cut off)
- `run_test.py`: regenerates the Q7 test files from the best checkpoint

Commands:
```
# Q7 fine-tuned model
python3 train_t5.py --finetune --learning_rate 5e-4 --batch_size 16 --max_n_epochs 20 \
    --num_warmup_epochs 1 --patience_epochs 3 --eval_every 4 --experiment_name experiment_v2
PYTHONPATH=. python3 run_test.py   # -> results/t5_ft_experiment_test.sql, records/t5_ft_experiment_test.pkl

# Extra credit: same architecture from random weights
python3 train_t5.py --learning_rate 3e-4 --batch_size 16 --max_n_epochs 40 \
    --num_warmup_epochs 2 --patience_epochs 3 --eval_every 5 --experiment_name experiment_ec
# -> rename t5_scr_experiment_ec_test.* to t5_ft_experiment_ec_test.*
```

Note: the extra-credit model was trained before the `<` placeholder step was added to
`load_data.py` / `train_t5.py`, so its submitted outputs come from the version without it.

## Checkpoints (Google Drive)
- Extra credit: https://drive.google.com/drive/folders/1W1SE-KuT4T2CSWafBB3ARr1lY-O2LG1r?usp=sharing
- Q7 fine-tuned: see the link in the PDF report
