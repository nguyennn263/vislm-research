# backbones

- `modules/transformer_block.py`, `models/transformer_lm.py` — standard Transformer LM,
  used as Arm A's architecture in Pillar 1 question 1.3 (`transformer_standard`).
  Trained end-to-end on Kaggle, see `runs/2026_09_12_train_arm_a_debug_kaggle/`.
- Trụ cột 2 (SSM-Attention hybrid) modules - not implemented yet. See
  `phases/phase-2-small-train.md` - don't build ahead of validated Tier 1 conclusions
  (execution principle #4 in `plans/PLAN.md`).
