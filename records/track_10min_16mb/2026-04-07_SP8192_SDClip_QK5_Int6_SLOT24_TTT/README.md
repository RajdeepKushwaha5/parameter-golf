# SP8192 + SDClip + QK5 + Int6/Int8 + SLOT24 + Score-First TTT

## Key Changes from SP1024 Baseline

### Tokenizer: SP8192 (highest-impact change)
- **Vocab size**: 1024 → 8192 (8× larger BPE vocabulary)
- **Expected gain**: ~0.03–0.05 BPB reduction from better tokenization efficiency
- Tokenizer trained on 5M docs from FineWeb-Edu
- Pre-built data available from `kevclark/parameter-golf` HF repo

### Quantization: Mixed Int6/Int8
- **Int6 GPTQ** for attention and MLP weight matrices (Hessian-aware, SDClip k=12.85)
- **Int8** for embedding table (int8 is better for 8192-vocab precision sensitivity, SDClip k=20)
- Selective ±1 pruning to hit 16MB target
- LZMA-9 compression

### Disabled Features (unnecessary with SP8192)
- **BigramHash**: Disabled — 8192 vocab absorbs bigram/trigram info directly
- **TrigramHash**: Disabled — same reason
- **ValueEmbedding (VE)**: Disabled — larger tok_emb already captures value identity

### Test-Time Optimization
- **SLOT**: 24-step logit bias optimization, cosine LR 0.024→0.001, warm-started bias, vectorized context loss
- **Score-First TTT**: Legal test-time training (each window scored frozen BEFORE any gradient update). SGD with momentum, LR=0.005, 1 epoch

### Architecture
- 11 layers, model_dim=512, 8 heads / 4 KV heads (GQA)
- XSA on ALL 11 layers
- Depth recurrence (layers 3,4,5, start step 2500)
- Parallel residuals (GPT-J style) for last 4 layers
- QK-Gain 5.0, logit softcap 30.0, partial RoPE 16 dims
- EMA decay 0.997, Muon WD 0.06

### Embedding Tuning (for SP8192)
- `tied_embed_lr`: 0.035 → 0.02 (lower LR for 8× larger table)
- `tied_embed_init_std`: 0.005 → 0.003 (tighter initialization)

## Data Preparation
```bash
# Option 1: Download pre-built SP8192 data (~30 min)
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 100

# Option 2: Train tokenizer + tokenize from scratch (~1-2 hours)
bash data/prep_sp8192.sh
```

## Training
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
