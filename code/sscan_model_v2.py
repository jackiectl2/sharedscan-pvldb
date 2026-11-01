#!/usr/bin/env python3
"""Model v2 of shared-scan vs request-level prefix caching (developed after the shared-scan gate).

v2 was written after the gate showed v1 failing on SciFact (C7). It is labeled post-hoc and is
validated only on the confirmatory cells (CONFIRM_PREREG.md). v1 serves one request at a time with
LRU retention, so a scan block larger than the effective KV looks like thrashing. v2 changes only the
cache timeline, replacing it with block-step interleaving of admitted requests, approximating the
installed vLLM 0.30 V1 engine (checked in its source):

  * engine steps with a token budget (max_num_batched_tokens = 2048, as logged by every gate run)
    and at most 256 running requests; running requests are scheduled first, then waiting requests
    FCFS; the last request admitted in a step may be chunked (chunked prefill);
  * on admission a request matches its prompt's 16-token blocks against cached blocks (a block hits
    only if every earlier block hit); the hit blocks become referenced;
  * blocks for the tokens scheduled in a step are allocated and registered in the cache at
    scheduling time (kv_cache_manager.allocate_slots -> cache_blocks), so a request admitted later
    in the same step, or in a later step, reuses them; blocks held by running requests are
    reference-counted and cannot be evicted;
  * a finished request frees its blocks tail first into an LRU free queue; new blocks evict the
    oldest free cached blocks;
  * a step costs alpha * scheduled tokens + beta * requests in the step. A filter request decodes one
    token, so it finishes in the step that schedules its last prompt token.

Everything else is v1 unchanged: alpha, beta, kappa (pool = kappa * KV blocks) and the stage-1 pass
rate come from v1's calibration fit on C1 seed 0; the release policies are v1's (same as the harness).

With params["capacity"] == "configured" (model M3 of CONFIRM_PREREG.md) the pool is instead the
configured KV size, num_gpu_blocks - 1 (vLLM reserves one null block, v1/core/block_pool.py), and kappa
is unused: a post-hoc development check showed that kappa is not identified by the calibration cell.
"""
import heapq
import itertools
import math
import random
from collections import OrderedDict, deque

import sscan_model as v1

BLK = 16
STEP_TOKENS = 2048
MAX_SEQS = 256


class Pool:
    """KV block pool with content-keyed cached blocks, reference counts and an LRU free queue."""

    def __init__(self, n_blocks, enabled=True):
        self.n, self.enabled = n_blocks, enabled
        self.ref = {}             # cached key -> reference count (keys in use or resting in the free queue)
        self.free = OrderedDict()  # cached keys with count 0, oldest (evicted first) at the front
        self.in_use = 0           # blocks referenced by running requests
        self.partial = set()      # allocated blocks not yet full: not cached until full (as in vLLM)

    def prefix_hits(self, keys):
        if not self.enabled:
            return 0
        h = 0
        for k in keys:
            if k not in self.ref or k in self.partial:
                break
            h += 1
        return h

    def can_take(self, hit_keys, n_new):
        """Room for referencing hit_keys and allocating n_new fresh blocks."""
        resting = sum(1 for k in hit_keys if self.ref.get(k, 0) == 0)
        return self.n - self.in_use - resting >= n_new

    def touch(self, keys):
        for k in keys:
            if self.ref[k] == 0:
                del self.free[k]
                self.in_use += 1
            self.ref[k] += 1

    def allocate(self, keys):
        for k in keys:
            if self.enabled and k in self.ref:
                # Same content cached again after a prefix miss: shared here, whereas vLLM allocates a
                # duplicate block (an approximation; v2 is a reported diagnostic, not the primary model).
                if self.ref[k] == 0:
                    del self.free[k]
                    self.in_use += 1
                self.ref[k] += 1
                continue
            while len(self.ref) >= self.n and self.free:
                old, _ = self.free.popitem(last=False)
                del self.ref[old]
            self.ref[k] = 1
            self.in_use += 1

    def release(self, keys):
        for k in reversed(keys):  # tail blocks enter the free queue first, so they are evicted first
            self.partial.discard(k)
            self.ref[k] -= 1
            if self.ref[k] == 0:
                self.in_use -= 1
                if self.enabled:
                    self.free[k] = True
                else:
                    del self.ref[k]


def simulate(cfg, trace, policy, params, rng):
    """Same interface and outputs as sscan_model.simulate (filter family)."""
    assert cfg.get("family", "filter") == "filter", "v2 models one-token filter requests only"
    a, b, kap, p_pass = params["alpha"], params["beta"], params["kappa"], params["pass_rate"]
    n, L, SP, pre = cfg["n"], cfg["doc_tokens"], cfg["shared_prefix"], cfg["preamble"]
    n_pool = cfg["kv_tokens"] // BLK - 1 if params.get("capacity") == "configured" else int(kap * cfg["kv_tokens"] / BLK)
    pool = Pool(n_pool, enabled=cfg.get("apc", True))
    pre_blocks = pre // BLK
    stats = dict(computed=0.0, prompt=0.0, requests=0)
    q_by = {q["qid"]: dict(q) for q in trace}
    for q in q_by.values():
        q["_docs"] = set(v1.docs_of(q, n))
    pending = {qid: 0 for qid in q_by}
    released_all = {qid: False for qid in q_by}
    done_t = {}
    waiting, running = deque(), []
    events, seq = [], itertools.count()
    for q in q_by.values():
        heapq.heappush(events, (q["arrival"], next(seq), "arrive", q["qid"]))
    if policy == "window":
        W = cfg["window"]
        last = max(q["arrival"] for q in q_by.values())
        k = 1
        while (k - 1) * W <= last:
            heapq.heappush(events, (k * W, next(seq), "window", None))
            k += 1
    block = cfg["block"]
    nb = math.ceil(n / block)
    need, active, s1_inflight, window_buf, pos = {}, [], {}, [], [0]
    uid = itertools.count()

    def keys_for(d):
        """One key per KV block of the prompt (ceil(L/16) blocks). Full preamble and document blocks
        are shared across questions; the rest (question tail, partial last block) are unique."""
        shared = min(SP[d] // BLK, L[d] // BLK) if pool.enabled else 0  # APC off: nothing is shared
        u = next(uid)
        return ([("pre", i) for i in range(min(pre_blocks, shared))] + [(d, i) for i in range(pre_blocks, shared)]
                + [("u", u, i) for i in range(shared, -(-L[d] // BLK))])

    def enqueue(qid, d, stage, blk):
        waiting.append(dict(qid=qid, d=d, stage=stage, blk=blk))
        pending[qid] += 1

    def try_release():
        while len(s1_inflight) < cfg["inflight"]:
            dm = {}
            for qid in active:
                for bk in need[qid]:
                    if bk not in s1_inflight:
                        dm[bk] = dm.get(bk, 0) + 1
            if not dm:
                return
            bsel = next((pos[0] + k) % nb for k in range(nb) if (pos[0] + k) % nb in dm)
            pos[0] = (bsel + 1) % nb
            docs = range(bsel * block, min(n, (bsel + 1) * block))
            qs = [qid for qid in active if bsel in need[qid]]
            cnt = 0
            for qid in qs:  # query-major within a block, as in the harness
                for d in docs:
                    if d in q_by[qid]["_docs"]:
                        enqueue(qid, d, 1, bsel)
                        cnt += 1
            for qid in qs:
                need[qid].discard(bsel)
            if cnt:
                s1_inflight[bsel] = cnt

    def check_done(qid, t):
        if qid in done_t or pending[qid] > 0:
            return
        if policy == "lazy":
            if qid not in need or need[qid]:
                return
        elif not released_all[qid]:
            return
        done_t[qid] = t
        if policy == "lazy" and qid in active:
            active.remove(qid)

    def handle(kind, payload):
        if kind == "arrive":
            q = q_by[payload]
            if policy == "eager":
                for d in v1.docs_of(q, n):
                    enqueue(payload, d, 1, None)
                released_all[payload] = True
            elif policy == "window":
                window_buf.append(payload)
            else:
                active.append(payload)
                need[payload] = {d // block for d in q["_docs"]}
                try_release()
        elif kind == "window" and window_buf:
            batch = list(window_buf)
            window_buf.clear()
            for d, qid in sorted((d, qid) for qid in batch for d in q_by[qid]["_docs"]):
                enqueue(qid, d, 1, None)
            for qid in batch:
                released_all[qid] = True

    def mark_partial(r, old_done):
        """Blocks become cacheable once full; the block holding a chunk's last token may be partial."""
        for k in r["keys"][old_done // BLK:r["done"] // BLK]:
            pool.partial.discard(k)
        if r["done"] % BLK:
            pool.partial.add(r["keys"][r["done"] // BLK])

    def schedule_chunk(r, budget):
        """Schedule up to `budget` prompt tokens of running request r; allocate and register the
        blocks that hold them. Returns scheduled tokens, or 0 if the pool has no room."""
        ntok = min(L[r["d"]] - r["done"], budget)
        upto = -(-(r["done"] + ntok) // BLK)
        new = r["keys"][r["nalloc"]:upto]
        if new and not pool.can_take([], len(new)):
            return 0
        pool.allocate(new)
        r["nalloc"] = upto
        old = r["done"]
        r["done"] += ntok
        mark_partial(r, old)
        return ntok

    clock = 0.0
    while True:
        while events and events[0][0] <= clock:
            _, _, kind, payload = heapq.heappop(events)
            handle(kind, payload)
        if not waiting and not running:
            if not events:
                break
            clock = events[0][0]
            continue
        budget, nreq, sched = STEP_TOKENS, 0, 0
        for r in running:  # running requests first (continuing chunked prefills)
            if budget <= 0:
                break
            got = schedule_chunk(r, budget)
            budget -= got
            sched += got
            nreq += int(got > 0)
        while waiting and budget > 0 and len(running) < MAX_SEQS:
            req = waiting[0]
            keys = keys_for(req["d"])
            hits = pool.prefix_hits(keys)
            hits = min(hits, (L[req["d"]] - 1) // BLK)  # the last prompt token is always computed
            ntok = min(L[req["d"]] - BLK * hits, budget)
            upto = -(-(BLK * hits + ntok) // BLK)  # blocks holding the tokens scheduled so far
            if not pool.can_take(keys[:hits], upto - hits):
                break
            waiting.popleft()
            pool.touch(keys[:hits])
            pool.allocate(keys[hits:upto])
            req.update(keys=keys, nalloc=upto, done=BLK * hits + ntok)
            mark_partial(req, BLK * hits)
            stats["computed"] += L[req["d"]] - BLK * hits
            stats["prompt"] += L[req["d"]]
            stats["requests"] += 1
            running.append(req)
            budget -= ntok
            sched += ntok
            nreq += 1
        if sched == 0:
            # Nothing can be scheduled (pool full of referenced blocks): wait for the next event.
            if events:
                clock = max(clock, events[0][0])
                continue
            raise RuntimeError("deadlock: KV pool cannot hold any schedulable request")
        clock += a * sched + b * nreq
        still = []
        for r in running:
            if r["done"] < L[r["d"]]:
                still.append(r)
                continue
            pool.release(r["keys"][:r["nalloc"]])
            qid = r["qid"]
            pending[qid] -= 1
            if r["stage"] == 1 and q_by[qid]["p2"] is not None and rng.random() < p_pass:
                enqueue(qid, r["d"], 2, r["blk"])
            if policy == "lazy" and r["stage"] == 1:
                s1_inflight[r["blk"]] -= 1
                if s1_inflight[r["blk"]] == 0:
                    del s1_inflight[r["blk"]]
                    try_release()
        running = still
        for qid in q_by:
            check_done(qid, clock)
    lat = sorted(done_t[qid] - q_by[qid]["arrival"] for qid in q_by)
    p95 = lat[min(len(lat) - 1, int(math.ceil(0.95 * len(lat))) - 1)]
    return dict(computed=stats["computed"], prompt=stats["prompt"], requests=stats["requests"],
                cached_frac=1 - stats["computed"] / stats["prompt"], makespan=max(done_t.values()),
                p95=p95, mean_latency=sum(lat) / len(lat))


POLICY_OF = dict(v1.POLICY_OF, P6_lazyscan_b16="lazy", P6_lazyscan_b64="lazy")
BLOCK_OF = {"P6_lazyscan_b16": 16, "P6_lazyscan_b64": 64}


def predict_run(cfg, trace, name, params, reps=3, sim=simulate):
    pol = POLICY_OF[name]
    c = dict(cfg)
    if pol == "window":
        c["window"] = 30.0 if name.endswith("30") else 10.0
    if name in BLOCK_OF:
        c["block"] = BLOCK_OF[name]
    outs = [sim(c, [dict(q) for q in trace], pol, params, random.Random(r)) for r in range(reps)]
    return {k: sum(o[k] for o in outs) / reps for k in ("cached_frac", "makespan", "p95", "mean_latency", "computed")}
