"""Exercise the deployed completion guard without importing CUDA/model weights.

These checks prove incompatibilities, NOT async readiness. The real finish
method is compiled from its deployed source and exercised with in-flight state.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / 'experiments/allocation-deferral-implementation-20260914/completion.py'
tree = ast.parse(source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CompletionCache')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'finish')
namespace = dict(eligible=lambda request: True,
                 RequestStatus=NS(FINISHED_STOPPED=1, FINISHED_LENGTH_CAPPED=2))
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
request = NS(request_id='finished-with-next-step-in-flight',
             num_computed_tokens=108, num_tokens=105,
             status=1, num_in_flight_tokens=4)
cache = NS(scheduler=NS(_req_status={request.request_id: object()}),
           release_lookup_pin=lambda rid: None, selected={}, restored_pages={})
assert namespace['finish'](cache, request, (), memory=True) is False

# The native snapshot kernel uses this expression. Compare a boundary that
# was available after N against the same requested boundary after N+1 writes.
boundary = 104
accepted = 4
bias_at_finish = accepted-1-(104-boundary)
bias_after_next_step = accepted-1-(108-boundary)
assert 0 <= bias_at_finish <= 3
assert not 0 <= bias_after_next_step <= 3

# Outstanding output placeholders must count against the parking reservation.
tokens, placeholders, lookahead, unit = 7674, 8, 3, 7680
old_required = tokens+lookahead+1
async_required = tokens+placeholders+lookahead+1
assert old_required <= unit < async_required
print(json.dumps(dict(
    in_flight_terminal_snapshot_skipped=True,
    merely_draining_loses_snapshot_boundary=True,
    boundary_bias_before=bias_at_finish,
    boundary_bias_after=bias_after_next_step,
    accepted_token_only_reservation_undercounts=True,
), indent=2))
