from pathlib import Path
import json

base = Path('/teamspace/studios/this_studio/attention-binding-a11y')
f = base / 'data/results/binding/1b_step15000_binding.jsonl'
print('exists:', f.exists())
print('base exists:', base.exists())

if f.exists():
    rows = [json.loads(l) for l in open(f, encoding='utf-8') if l.strip()]
    print('rows:', len(rows))
    terms = list({r['term'] for r in rows})
    print('terms:', terms[:4])
else:
    print('FILE NOT FOUND')
    print('binding dir listing:')
    bd = base / 'data/results/binding'
    if bd.exists():
        for p in list(bd.iterdir())[:5]:
            print(' ', p.name)
    else:
        print('  binding dir missing')

# Also check __file__ path logic
import sys
print('python:', sys.executable)
