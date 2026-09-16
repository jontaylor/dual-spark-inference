# SQPOLL comparison candidate (not deployed)

Uses one reader and software cache for normal-versus-SQPOLL ABBA.
Normal gather/submit uses normal io_uring issue. The existing `submit_async`
entrypoint selects the dedicated SQPOLL ring for the first 4096 misses;
remaining rings use normal submission. It does NOT set IOSQE_ASYNC.
Therefore capture policy `async` means SQPOLL for this library only.

One kernel submission thread per reader; 10ms polling idle timeout.
Both ring sets participate in completion and pending-I/O lifetime checks.
Existing reader statistics offsets are unchanged.

Native byte/error/pending tests and 135 switching cases pass. GPU integration
is still required during serving downtime. Do not deploy on native tests alone.
The dedicated-file benchmark here is exploratory, during model loading;
it does not establish live GPU latency or end-to-end speedup.
