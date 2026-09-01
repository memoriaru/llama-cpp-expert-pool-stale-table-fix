# llama.cpp — MoE expert pool: stale map-table bug (patch + evidence)

Patch, root-cause analysis and reproducibility evidence for a **determinism bug** in the
persistent expert weight pool for MoE CPU offloading (the `ggml_backend_sched_register_expert_pool`
work, RFC #20757): with the pool active (`-mec > 0`), greedy decoding produces a different —
but deterministic — output than the host-copy path (`-mec 0`), forking within the first few
generated tokens.

Tested on Windows 11 / RTX 4090 24GB (CUDA, cc 8.9) with a 512-expert MoE model (48 layers,
top-k 10, separate `ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps` per layer → 288 pools).

## TL;DR

The ids→slot remap (`GET_ROWS` over the per-pool map table) is scheduled into the split
**before** the split whose prologue updates the pool — so the remap consumes the map table
uploaded for the **previous** ubatch. Every cache miss (cold load or eviction reload) then
reads whatever expert last occupied the now-remapped slot. At 512 experts / 32 slots the pool
thrashes (~20–30% of per-step expert reads are wrong weights), which is small enough for greedy
decoding to survive a few tokens before deterministically flipping at a near-tie — producing
fluent but wrong output.

The fix (attached patch, +47/−7 incl. comments):

1. create the map table as 2D `[1, n_expert]` so the graph needs no `RESHAPE` view over it
   (views carry no buffer and are invisible to the split-boundary check, which keys on
   `src->buffer`);
2. register `table_buf` in `expert_pool_by_buf`, so the remap `GET_ROWS` starts its own split;
3. extend the pool-update scan to remap `GET_ROWS` nodes (matched via the table's device
   copy), establishing the ordering invariant **update → upload fresh table → consume in the
   same split**. The `MUL_MAT_ID` branch is kept as a fallback for single-split backends.

## Root cause in detail

Per-pool state lives on the scheduler (`ggml/src/ggml-backend.cpp`):
`ggml_backend_sched_update_expert_pool()` runs in `compute_splits()` before a split is
launched — it reads back the routing ids, loads missing experts into free/evicted slots with
async H2D copies, and rewrites the host-side map table. The table has the `INPUT` flag and is
uploaded to the compute backend by the regular split-input copy path. The code comment states
the design assumption: *"pooled MUL_MAT_ID nodes always start a new split"* — i.e. the remap
and the pool update were expected to land in the same split, update first.

They don't. With host-side expert weights, `build_lora_mm_id()` rewrites the ids as
`reshape(get_rows(table, cont(ids)))`, and the split-boundary rule only fires on nodes whose
**src buffer** belongs to a pool. The pooled `MUL_MAT_ID` starts a new split, but the remap
`GET_ROWS` is *not* a pooled node itself — it lands at the tail of the previous split. The
previous split is launched (and its input copies upload the table) before the pool update for
the new ubatch has run, so the remap maps this ubatch's expert ids onto the **previous**
ubatch's slot layout.

A one-time split dump (144 splits) shows the pattern for every layer:

```
SPLIT 1: TGETROWS(blk.0 gate table)   ← gate remap computed here, table uploaded here (stale)
SPLIT 2: MMID(pool: blk.0 gate)       ← gate pool updated here (prologue) — too late
SPLIT 3: MMID(pool: blk.0 up)         ← up remap computed in SPLIT 2's tail — stale again
...
```

## Evidence chain

All runs: exclusive VRAM (2.4 GiB baseline), fresh server instance per config, greedy decoding,
834-token prompt, 128 new tokens. `mec0` output is byte-identical across code versions and
environments (three independent runs).

| # | Probe / control | Result | Conclusion |
|---|---|---|---|
| 1 | Old code, clean VRAM, `mec32` | 662 B, byte-identical to the historical diverged run (fork at byte 17, "wants"→"needs") | divergence is **not** a VRAM-pressure artifact — real bug |
| 2 | Fixed code (3b7a32d), clean VRAM, `mec32` | still 662 B, byte-identical to #1 | the "skip redundant tensors" hunk does not touch this bug |
| 3 | Env-var bypass (pools allocated, every MUL_MAT_ID forced onto the host path), `mec32` | **402 B, byte-identical to `mec0`** | VRAM layout/allocation effects excluded → bug is in the pool data path |
| 4 | Post-step verification probes (slot bytes vs host weights; device map table vs host table) | 0 mismatches anywhere | writes are correct → read side consumes something else |
| 5 | ids-stale probe (read back the remap output vs post-update host table) + split dump | 9413 stale-mapping records; remap confirmed in the *previous* split | **stale map-table consumption** (root cause) |
| 6 | Fix applied | fork point moves byte 17 → 354 (token ≈ 88, a benign near-tie flip, deterministic and identical for `mec32` and `mec64`); `mec0` and bypass runs byte-exact | stale-table bug fixed |

Why the unit matrix (40/40 bit-exact) missed it: with a single backend the whole graph is one
split, and the pool update runs in its prologue before anything is computed — the cross-split
ordering hazard is structurally unreachable. A "host weights + GPU pool" (multi-backend) case
in `test-expert-pool` would cover it; recommended.

## The fix

`expert-pool-stale-table-fix.patch` applies on top of the branch that adds the persistent
expert pool (`git apply expert-pool-stale-table-fix.patch`). It changes only
`ggml/src/ggml-backend.cpp` and `src/llama-graph.cpp`; `test-expert-pool` still passes on CUDA
after the fix.

## Reproduction

```bash
llama-server -m <512-expert-moe-gguf> -ngl 99 --cpu-moe --no-mmap -mec 32 -c 8192 --port 8999
curl localhost:8999/completion -d '{"prompt": "...", "n_predict": 128, "temperature": 0, "cache_prompt": false}'
# compare `content` against the same request with -mec 0 (fresh instance!)
```

Testing notes that cost us two invalidated rounds (don't repeat them):

- the A/B must **exclusively own VRAM** — a resident production service silently OOMs the
  server and produces garbage "divergence";
- each config needs a **fresh server instance**; a second request on the same instance hits
  the prompt cache (`timings.prompt_n` collapses to a handful of tokens) and the A/B silently
  degenerates into comparing a config with itself.

`evidence/` contains the captured responses (`det*`/`fix*` = historical rounds,
`exp12-*` = bypass controls, `exp13/14-*` = post-fix A/B, `f32.err` = the failed
mec32 launch behind the invalidated "fix verified" round).
`docs/testing-feedback.zh.md` is the full Chinese test log (matrix results, perf, root-cause
section §8).
