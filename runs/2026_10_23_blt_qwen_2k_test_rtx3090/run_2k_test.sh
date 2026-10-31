#!/bin/bash
# 2k-document end-to-end test on the RTX 3090: GPU unit tests -> entropy LM -> BLT-Qwen (entropy patching)
source $HOME/Desktop/ssd1/viwixtech/Quoc/env.sh
set -x
D=data/prepared/fw2_2k
VISLM_RUN_OVERFIT=1 python -m pytest tests/test_blt_qwen.py -q -p no:cacheprovider || echo PYTEST_FAILED
python -m vislm.train_blt_qwen experiments/blt_qwen/configs/entropy_lm.yaml profile=rtx24gb dataset=$D \
  run_dir=runs/2k_entropy_lm train.seq_len=2048 train.batch_size=8 train.max_steps=300 train.warmup_steps=30 \
  train.eval_every=100 train.eval_batches=5 train.save_every=0 train.log_every=20 || exit 1
rm -rf runs/2k_entropy_lm/checkpoints
python -m vislm.train_blt_qwen experiments/blt_qwen/configs/blt_qwen35_08b.yaml profile=rtx24gb dataset=$D \
  run_dir=runs/2k_blt_qwen patching.entropy_ckpt=runs/2k_entropy_lm/entropy_lm.pt \
  train.grad_accum=1 train.max_steps=200 train.warmup_steps=20 train.eval_every=50 train.eval_batches=5 \
  train.save_every=0 train.save_final=false train.log_every=10
echo ALL_DONE
