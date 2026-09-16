"""Reproducible time-stratified audit of the frozen follow-up trace."""
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
trace = json.loads((root / 'trace-final.json').read_text())
restart = json.loads((root.parent / 'native-pressure-20260914/start.json').read_text())['time']
audit = json.loads((root.parent / 'native-pressure-20260914/FINAL-AUDIT.json').read_text())['time']


def summarize(rows):
    if not rows:
        return {'requests': 0}
    result = {'requests': len(rows),
              'semantic_output_matches': sum(r['previous_output_semantically_matches_history'] for r in rows),
              'exact_estimated_previous_endpoint_hits': sum(r['cached'] == r['estimated_previous_computed'] for r in rows),
              'cache_hits_aligned_1920': sum(r['cached'] % 1920 == 0 for r in rows)}
    for field in ('uncached', 'estimated_new_tokens', 'older_prompt_replay_floor',
                  'previous_output_reused_estimate', 'previous_generated'):
        result['mean_' + field] = statistics.mean(r[field] for r in rows)
    result['mean_estimated_existing_content_replay'] = statistics.mean(
        max(0, r['estimated_previous_computed'] - r['cached']) for r in rows)
    return result


out = {'restart_unix': restart, 'verified_native_cutoff_unix': audit,
       'scope': 'Observational phase split, not a paired throughput experiment. '
                'Output identity ignores tool-call IDs and normalizes JSON arguments; '
                'token-count estimates are not token-prefix proofs.', 'campaigns': {}}
for campaign, data in trace.items():
    rows = data['rows']
    before = [r for r in rows if r['created'] < restart]
    after = [r for r in rows if r['created'] >= audit and r['previous_created'] >= audit]
    out['campaigns'][campaign] = {
        'all': summarize(rows), 'before_restart': summarize(before),
        'both_requests_after_verified_native_start': summarize(after),
        'excluded_transition_rows': len(rows) - len(before) - len(after)}
(root / 'phase-analysis.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
