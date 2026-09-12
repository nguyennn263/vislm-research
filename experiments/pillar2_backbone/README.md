# Pillar 2 - Backbone (SSM-Attention hybrid)

See `plans/PLAN.md` - Tru cot 2. All questions belong to `phases/phase-2-small-train.md`
(needs real training on the RTX 24GB machine).

- `configs/2_1a_*`, `2_1b_*` - compute-matched SSM:Attention ratio sweep + RoPE ablation, uses `data/prepared/fineweb2_vi`.
- `configs/2_2a_*` - replicate MOHAWK/MambaInLlama on the original English setup first (sanity check, no Vietnamese data needed).
- `configs/2_2b_*` - only after 2_2a matches published numbers; extends the same distillation recipe to a Vietnamese checkpoint.
