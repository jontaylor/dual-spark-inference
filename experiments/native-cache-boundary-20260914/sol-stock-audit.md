# Stock vLLM audit: Qwen3.8-Flash-Next multi-turn prefix reuse

Checked 2026-09-14 against the official `vllm-project/vllm` repository, not the
installed patched container. The applicable stable release is vLLM v0.29.0,
published 2026-09-09. It is the first stable release that lists
Qwen3.8-Flash-Next support (BF16, FP8, NVFP4, and MTP).

## Direct answer

For the literal default, unmodified vLLM v0.29.0 configuration for
`nvidia/Qwen3.8-Flash-Next-NVFP4` -- no speculative/MTP argument and no cache
retention override -- **the reusable hybrid prefix stops at a checkpoint in the
original input prompt. It does not preserve recurrent checkpoints through the
generated assistant response.** A later chat turn therefore reprocesses the
previous generated response, the tail between the coarse checkpoint and the
end of the old prompt, and the new user turn.

There is an important stock-v0.29 exception: if the operator explicitly enables
MTP/EAGLE-style speculative decoding and leaves retention unset, v0.29 switches
hybrid models to dense recurrent-state retention. In that configuration,
generated output can be reused, but only at resolved physical Mamba block
boundaries and with the EAGLE/MTP lookup rewind. MTP is supported by the model,
but it is not enabled merely because the checkpoint contains an MTP head.

Thus the broad statement "anyone using this model with vLLM has an input-only
cache" is false across every configuration, but it is true of the literal stock
default. The earlier categorical "no" to that default-behaviour question was
wrong.

## Why the literal default is input-only

Qwen3.8-Flash-Next has 36 linear-attention layers and 12 full-attention layers
(one full-attention layer every four layers), plus a PLE short-convolution
state. Stock vLLM therefore uses its hybrid cache manager. Prefix caching is
enabled by default for supported generative hybrid models, and the Mamba cache
mode is changed from `none` to `align` when prefix caching is enabled.

v0.29.0 also defaults `prefix_cache_retention_interval` to `0`. In that mode,
the Mamba manager's retention mask keeps semantic replay checkpoints, chiefly
the boundary derived from `request.num_prompt_tokens - 1`; it does not keep
periodic decode checkpoints. Full-attention KV may exist farther into the
sequence, but the hybrid coordinator returns the shortest hit common to all
cache groups, so the missing recurrent checkpoint caps the usable joint hit at
the old-input boundary.

The wording "previous input" needs one precision: this is a block-aligned
checkpoint inside the previous input, not necessarily its exact endpoint.
Stock lookup also caps a full prompt hit at `prompt_length - 1` because it needs
to recompute a token to obtain logits.

Relevant stock v0.29.0 evidence:

- Release notes: Qwen3.8 support and the retention default are both new in
  v0.29.0: <https://github.com/vllm-project/vllm/releases/tag/v0.29.0>
- `enable_prefix_caching=True`, `prefix_match_unit=None`, and retention `0`:
  <https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/cache.py#L99-L166>
- Prefix caching makes hybrid/Mamba models use `align` mode:
  <https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/model_executor/models/config.py#L614-L644>
- The base cache code identifies the replay boundary from
  `num_prompt_tokens - 1`:
  <https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/core/single_type_kv_cache_manager.py#L432-L479>
- The Mamba retention mask says `0` keeps only reachable prompt boundaries and
  `None` is dense; its `boundary_block` calculation selects the state ending at
  the aligned boundary:
  <https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/core/single_type_kv_cache_manager.py#L1472-L1523>
- The upstream unit test states and verifies that zero retention keeps only the
  prompt replay boundary:
  <https://github.com/vllm-project/vllm/blob/v0.29.0/tests/v1/core/test_prefix_caching.py#L4145-L4180>
- PR #52216 explicitly changed the default to `0` and defines it as retaining
  only the latest replayable prompt boundary:
  <https://github.com/vllm-project/vllm/pull/52216>

## What is retained during decode

`align` mode always needs a mutable running state for the active request. That
is not the same as preserving that state as a prefix-cache checkpoint after the
request moves on or finishes.

With default retention `0`, decode advances the mutable recurrent state, but
states at generated-token positions are not hashed and retained for future
requests. The old prompt replay state is the reusable one.

With dense retention (`prefix_cache_retention_interval=None`), every
materialized **full physical Mamba block boundary** is hashed and retained.
Those boundaries can extend through decode, so a later turn can reuse the
generated response through the last common retained boundary. The final sampled
token has not itself been run through the model when generation stops and must
be recomputed; block alignment generally rewinds farther.

With a positive retention interval `N`, the Mamba cache retains periodic
checkpoints at `N`-token segments plus the semantic prompt boundary. `N` must be
a multiple of the scheduler block size.

`prefix_match_unit` is independent of physical allocation, but its special
partial-state registration in v0.29 is deliberately limited to the final
hash-aligned **prompt** boundary. It does not create equally fine checkpoints
through decode. Therefore setting `prefix_match_unit=64` improves prompt-tail
precision but does not by itself preserve an exact generated-response endpoint:
<https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/core/single_type_kv_cache_manager.py#L1858-L1903>.

## MTP behaviour in the stable release

MTP must be enabled explicitly, for example with a speculative config whose
method is `mtp`; the Qwen support PR's example does so. vLLM classifies `mtp` as
an EAGLE-style hidden-state drafter. Stable v0.29.0 special-cases an unset
retention value for a hybrid model with EAGLE/MTP and changes it from `0` to
`None` (dense):

- Qwen support PR and launch example:
  <https://github.com/vllm-project/vllm/pull/53896>
- MTP is classified by `use_eagle()`:
  <https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/speculative.py#L1843-L1848>
- Stable v0.29.0 dense-retention exception:
  <https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/engine/arg_utils.py#L2388-L2407>
- Release-note description: "dense retention automatically restored for hybrid
  models using EAGLE/MTP":
  <https://github.com/vllm-project/vllm/releases/tag/v0.29.0>

Dense retention means decode checkpoints do survive at physical block
boundaries. EAGLE/MTP lookup also requires a proof/lookahead unit from each
cache group marked as draft-owned and drops it before returning the hit. With
the stock grouping fallback for this model, no draft group is explicitly
annotated, so all groups are treated as EAGLE groups; with the default match
unit the drop is one physical cache block. Consequently the practical returned
hit is normally at least one physical block behind the deepest cached common
prefix. A response shorter than that coarse interval may produce no advance
beyond the old prompt checkpoint, even though decode checkpoints are enabled.

For the known TP=2, BF16-QSA-KV layout used in the local work, stock physical
alignment resolves to approximately:

- no MTP: 1,568 tokens;
- MTP3: 1,600 tokens;
- MTP5: 1,616 tokens when starting from the normal 16-token backend block.

These values follow from the stock page-size alignment rule and the model's
MTP-dependent recurrent-state shapes. They are not universal model constants:
TP, KV dtype/backend, and an explicit larger block request can change them.
The local 1,920-token value was explicitly chosen and is not a stock default.

## Version boundary

The result is unusually version-sensitive because the official stable tag and
the current upstream `main` branch differ.

- v0.29.0 is the applicable released version and contains the hybrid+MTP dense
  default described above.
- Upstream `main` at commit
  `b7e8dd8f372c1324c376ba5dc03d92264eae8f24` (fetched 2026-09-14) defaults
  retention directly to `0` and does not contain v0.29.0's EngineArgs
  hybrid+MTP dense override. A post-release main-branch fix retains the two
  prompt replay boundaries needed for sparse MTP/EAGLE lookup, but it still
  does not retain generated-response checkpoints under the default `0` policy:
  <https://github.com/vllm-project/vllm/commit/b28c3e1568bfae930f61d4b24940e47528c85d4a>.

Current `main` also warns when a hybrid EAGLE/MTP model has no cache group
marked as draft-owned. Qwen3.8's one-layer MTP draft contains full attention,
but its cache spec has no draft marker and Qwen is not covered by the
DeepSeek-specific positional fallback. This warning/support edge is another
reason not to generalize results across versions without naming the revision.

So the released stock answer is:

- default, no MTP: input-prompt checkpoint only;
- explicitly enabled MTP with otherwise default v0.29 settings: generated
  output is reusable only at coarse physical block checkpoints, with an MTP
  rewind;
- exact completion-endpoint reuse: not provided by either stock path.

No conclusion above uses the patched installed container as evidence for stock
behaviour. No service was restarted and no runtime configuration was changed.
