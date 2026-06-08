#!/bin/bash
set -e
cd /teamspace/studios/this_studio/attention-binding-a11y
echo "====== C3 GAP RUNS ======"
echo "Started: $(date)"

# 2.8b on 9 terms — early checkpoint (EB* rising phase)
echo -e "\n[1/4] Pythia-2.8b | step15000"
python src/eval_few_shot_c3.py --model 2.8b --checkpoint step15000

# 2.8b on 9 terms — late checkpoint (decoupled phase)
echo -e "\n[2/4] Pythia-2.8b | step143000"
python src/eval_few_shot_c3.py --model 2.8b --checkpoint step143000

# OLMo — early checkpoint (EB* rising, step15k)
echo -e "\n[3/4] OLMo-1B | step15k"
python src/eval_few_shot_c3.py --model olmo --checkpoint step15k

# OLMo — late checkpoint (post-peak, step143k)
echo -e "\n[4/4] OLMo-1B | step143k"
python src/eval_few_shot_c3.py --model olmo --checkpoint step143k

echo -e "\n====== C3 GAPS DONE: $(date) ======"
