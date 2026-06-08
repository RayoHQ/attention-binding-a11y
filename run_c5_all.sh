#!/bin/bash
# Run all missing C5 causal ablations sequentially
# Estimated total: ~2-3 hours
set -e
cd /teamspace/studios/this_studio/attention-binding-a11y

echo "====== C5 RUN QUEUE ======"
echo "Started: $(date)"

# 1. Pythia-1b on original 3 terms (missing model)
echo -e "\n[1/6] Pythia-1b | 3 orig terms | step120000"
python src/run_causal_c5.py \
  --model 1b --checkpoint step120000 \
  --prompts data/prompts/pilot_terms.jsonl \
  --output data/results/causal/1b_step120000_c5_orig.json

# 2. Pythia-160m on 21 new terms (larger N)
echo -e "\n[2/6] Pythia-160m | 21 new terms | step120000"
python src/run_causal_c5.py \
  --model 160m --checkpoint step120000 \
  --prompts data/prompts/expanded_terms_tier123.jsonl \
  --output data/results/causal/160m_step120000_c5_tier123.json

# 3. Pythia-1b on 21 new terms
echo -e "\n[3/6] Pythia-1b | 21 new terms | step143000"
python src/run_causal_c5.py \
  --model 1b --checkpoint step143000 \
  --prompts data/prompts/expanded_terms_tier123.jsonl \
  --output data/results/causal/1b_step143000_c5_tier123.json

# 4. Pythia-2.8b on 21 new terms (larger N replication)
echo -e "\n[4/6] Pythia-2.8b | 21 new terms | step143000"
python src/run_causal_c5.py \
  --model 2.8b --checkpoint step143000 \
  --prompts data/prompts/expanded_terms_tier123.jsonl \
  --output data/results/causal/2.8b_step143000_c5_tier123.json

# 5. OLMo-1B on 9 terms
echo -e "\n[5/6] OLMo-1B | 9 terms | step143k"
python src/run_causal_c5_olmo.py \
  --checkpoint step143k \
  --prompts data/prompts/expanded_terms_100.jsonl \
  --output data/results/causal/olmo_1b_step143k_c5_9terms.json

# 6. Pythia-1b on 9 orig terms (exp100, medium N=45 rec prompts)
echo -e "\n[6/6] Pythia-1b | 9 terms exp100 | step143000"
python src/run_causal_c5.py \
  --model 1b --checkpoint step143000 \
  --prompts data/prompts/expanded_terms_100.jsonl \
  --output data/results/causal/1b_step143000_c5_9terms.json

echo -e "\n====== ALL C5 DONE: $(date) ======"
