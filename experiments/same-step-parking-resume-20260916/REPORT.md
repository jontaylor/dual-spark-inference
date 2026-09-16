# Same-pass park/restore crash recovery

At 17:35:20 UTC on 16 September, the synchronous engine crashed in
`OffloadingConnectorScheduler.build_connector_meta`, asserting that a transfer
job was a store. Both containers exited. No async candidate was installed.
Docker reported no OOM kill; the engine traceback identifies the assertion.

The request was parked, immediately re-admitted, and assigned a new checkpoint
restore load in the same scheduling pass. Its ID remained in the pass's
preempted-request set. The connector treated the newly created restore as an
old store to flush before submitting new transfers.

The fix iterates the request's jobs: old stores still enter the flush set;
loads are permitted only when they are new in this batch and belong to that
same request. New loads remain in the normal load-submission metadata. A load
from an older batch or another request still fails the invariant.

`check.py` reproduces the original assertion by executing the original metadata
method and verifies the patched method, including mixed old-store/new-load
jobs, continued store flushing, and rejection of invalid loads. It does not
claim full-model parking validation.

Both rank configs were updated to this versioned source and SHA256
`0679e7ebaa39c01254918906e0b2a014d2b0f652fc3f7af4057ef3e3fcb59fd7`.
The mounted file hashes match on both nodes. Recovery start was requested at
approximately 17:40 UTC. Async scheduling remains disabled. The recorded readiness check returned HTTP 200 at 17:48:08 UTC.
Subsequent engine logs show request `chatcmpl-b4d814fc6809dcdc-9310d0c0`
parked at 17:57:30 and restored at 17:57:33 (boundary 46078, replay one
token). This verifies one live park/restore cycle after recovery; it is not
exhaustive lifecycle validation.

Evidence and prior configs: `results/async-scheduling-20260916/server-exit-r0.log`,
`recovery-before-r{0,1}.json`, and `recovery-start.json`.
The service's pending on-disk unit changes were not reloaded.
