# Follow-up cache-rate investigation

Latest history sample 1789389058.706746: cumulative native hit rate 84.16%;
last 10 minutes 92.89%; last six minutes
92.75%. The final deployment audit contained
552,757 cumulative uncached prompt tokens, largely initial cold reprocessing.
Elapsed uptime does not remove those misses from cumulative counters.

The vLLM occupancy gauge is 1 - free_block_queue.num_free_blocks / usable_blocks.
That free list includes zero-reference native cached prefixes. It therefore
reports active/pinned occupancy, not the fraction of GPU pages holding history.
The contemporaneous accounting sample had 213 of 1580 blocks actively referenced
and zero off-table pinned blocks. No completion snapshots are retained.

Native MTP replay excludes the final token, rounds to 1920-token alignment, and
drops one extra aligned block. On an otherwise reusable prefix, that can mean
roughly 1920–3839 tokens of replay before new content. New prompt content also
misses. This inspection does not provide a complete per-request attribution of
the remaining ~7% misses.

The eviction policy has a remaining selectivity problem: a sampled 141 native
backing-store events comprised 140 GDN state pages (35 in each of groups 0–3)
and one attention page (group 4). They are ordinary native-cache evictions, not
active-request preemptions. The later metric sample had 9,830,645,760 stored bytes
across ranks and zero loaded bytes. Saving every evicted native checkpoint is
broader than the desired emphasis on completion/follow-up and suspended state.

No serving configuration changed and no restart was performed in this inspection.
Raw evidence: cache-window-analysis.json and cache-followup-live.log.
