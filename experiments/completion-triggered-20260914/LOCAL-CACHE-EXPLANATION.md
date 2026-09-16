# Why local prefix hit rates are low

Live metric sample in cache-explanation-live.json: local hits 4,343,040 / 8,166,706 queries = 53.18%; connector hits 3,050,734 / 3,823,666 remaining queries = 79.79%. Relative to all queried prompt tokens, connector reuse is 37.36% and fresh computation is 9.46%. Local plus connector reuse is 90.54% in this cumulative sample. These counters include the restart and synthetic validation/pressure traffic, and are not directly comparable to the user's remembered historical 82% local rate.

The completion connector owns a second cache in the same physical GPU block pool. Its GPU-resident snapshots still count as external connector hits. At this sample there were 1,226 resident completion blocks. At 27,156,480 bytes per physical block, that is 31.01 GiB per rank, about 77.6% of the 1,580 usable blocks in the 40 GiB KV pool. Subsequent independent block-accounting log sample at Unix time 1789385967.623906 reported pool_used=1441, pool_usable=1580 and off_table_used=1225 with eight active requests, consistent with substantial off-table residency.

This is real cache competition, not only counter labeling. DiskSlotManager.prepare_memory_store obtains blocks from memory_pool.get_new_blocks and increments memory_ledger.cache_reserved. BlockPool.get_new_blocks takes zero-reference/free-list blocks and calls _maybe_evict_cached_block, which can remove normal prefix-cache entries. A free-list block can contain reusable native prefix data; it is not necessarily empty memory. Completion snapshots are reclaimed by the parking connector when admission/growth needs budget, rather than being assigned a separate small fixed quota. Ordinary native-cache pages and retained completion snapshots therefore have different reclamation paths.

Native lookup also needs valid hybrid state: full-attention and recurrent hits can diverge. The connector-aware lookup can use surviving full-attention pages with externally supplied state, but falls back to a reconciled boundary when required groups are missing. Merely retaining some GPU attention data does not guarantee a usable local hit at the desired position.

The event-checkpoint change removed periodic CPU shadow capture during generation. It did not redesign completed-snapshot residency or its competition with native prefix entries. The 64-token shadows were private transient CPU state, not a set of completed connector cache records at every boundary.

Current code already shares eligible inherited connector full-attention pages, but does not generally alias native full-attention pages into the completion snapshot. A targeted future improvement is shared ownership of immutable native attention pages, separate private recurrent/ring state, and explicit retention balance. Simply lowering connector residency can shift traffic to disk or recomputation; it is not by itself proof of better throughput.

No serving settings changed during this explanation.
