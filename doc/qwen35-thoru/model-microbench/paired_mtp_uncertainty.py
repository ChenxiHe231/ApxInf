"""Within-run paired block resampling, conditional on the fixed questions/order.

This is not a confidence guarantee across devices, loads, or time windows.
"""
import argparse, json, math, pathlib, random

p = argparse.ArgumentParser()
p.add_argument('task', type=pathlib.Path)
p.add_argument('--run', required=True)
a = p.parse_args()
root = a.task / 'runs' / a.run
s = json.loads((root / 'summary.json').read_text())
assert s['ok'] and s['performance_eligible']
files = sorted(root.glob('*/samples.jsonl'))
assert len(files) == 4
arms = [[x for x in map(json.loads, f.read_text().splitlines()) if not x['warmup']] for f in files]
groups = {'original5': {'92', '100', '109', '112', '154'}, 'heldout8': {'82', '96', '102', '115', '123', '131', '145', '155'}}
result = {'schema': 'apxinf.paired_block_resampling.v1', 'run': a.run, 'seed': 20261124, 'draws': 10000, 'method': 'Resample five complete question-group repeat blocks within each of the two A/B time-window pairs, then combine both pairs. Fixed questions and order; does not model cross-window or deployment uncertainty.', 'groups': {}}
for group, ids in groups.items():
    pairs = []
    for base, candidate in [(0, 1), (3, 2)]:
        blocks = []
        for repeat in range(5):
            sides = []
            token_counts = []
            for arm in [base, candidate]:
                rows = [x for x in arms[arm] if x['repeat'] == repeat and str(x['case_id']) in ids]
                assert len(rows) == len(ids)
                sides.append(math.fsum(x['timing']['request_wall_ms'] for x in rows))
                token_counts.append(sum(x['output_tokens'] for x in rows))
            assert token_counts[0] == token_counts[1]
            blocks.append(sides)
        pairs.append(blocks)
    rng = random.Random(result['seed'])
    samples = []
    for _ in range(result['draws']):
        picked = [blocks[rng.randrange(5)] for blocks in pairs for _ in range(5)]
        samples.append(100 * (math.fsum(x[0] for x in picked) / math.fsum(x[1] for x in picked) - 1))
    samples.sort()
    lo, hi = samples[int(.025 * (len(samples) - 1))], samples[int(.975 * (len(samples) - 1))]
    result['groups'][group] = {'e2e_gain_pct_interval95': [lo, hi], 'lower_bound_positive': lo > 0, 'paired_time_blocks_ms': pairs}
result['all_group_lower_bounds_positive'] = all(x['lower_bound_positive'] for x in result['groups'].values())
(a.task / 'results' / (a.run + '-uncertainty.json')).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
