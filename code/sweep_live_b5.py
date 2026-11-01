#!/usr/bin/env python3
"""Follow-up runs B2/B3 of the shared-scan paper (refine-logs/EXPERIMENT_PLAN.md), derived from the frozen
confirmatory harness confirm_live.py, which stays unchanged. Additions: an adaptive batch-when-idle policy
(P2_adaptive), a 60-second window, and P6 with any block size (P6_lazyscan_b<N>). Frozen mode reads the sweep
freeze file (sweep_freeze.py), which carries the confirmatory corpora and traces unchanged.

Confirmatory round for the shared-scan paper (CONFIRM_PREREG.md), derived from shared_scan_live.py.

Additions: a long-document corpus (PubMed Open Access articles from Cohan et al. 2018, via
ccdv/pubmed-summarization, test split; documents are articles with >= 3,000 characters, cut to
their first 3,000 characters), P6 variants with block sizes 16/32/64 run on the same trace, a
configurable maximum model length, pinned scheduler limits (max_num_batched_tokens = 2048 and
max_num_seqs = 256, the values every gate run logged), and a frozen mode: with --freeze, the
documents, arrival trace and policies of one cell and seed are read from the frozen file, and the
run stops unless its prompts reproduce the frozen token counts.

Shared-scan gate (Idea D as a predictive shared-scan systems paper), derived from gonogo_d_live.py.

Adds: corpus choice (arXiv | SciFact), APC on/off, query-range overlap level, an extraction
family (stage 2 = short extraction with longer decode), eager rerun for output variability,
and a fixed SGLang cache flush. Original description follows.

Live go/no-go test for Idea D: online cross-query sharing of semantic-filter work.

One continuously running vLLM AsyncLLM engine (automatic prefix caching on, KV capacity set
with num_gpu_blocks_override). A seeded finite trace of semantic queries arrives as a Poisson
process. Each query is a 1- or 2-stage filter pipeline over a contiguous document range
(a stand-in for a relational range predicate); stage 2 runs only on documents that pass
stage 1. Every policy uses the same query-last prompts and the same model:

  P0 serial          one query at a time (also the isolated same-layout reference)
  P1 eager           each query submits all its stage-1 requests on arrival (engine FCFS + APC)
  P2 window W        arrived queries are released together, document-major, every W seconds
  P6 lazy scan       circular block scan; active queries attach at the next block; stage 2
                     submitted as soon as stage 1 returns; next block released when fewer
                     than F blocks are in flight; blocks with no demand are skipped
  P3 ours            relevance-ordered block choice with aging, capacity-aware
                     drain-or-advance, and an admission limit
  P3drain            ablation of P3 that always drains stage 2 before advancing
  P4 fused           P6 scan, but one prompt per document asks all active stage-1 questions

Latency is measured from arrival to the last result of the query; the same trace is used for
every policy, and all queries must complete.
"""
import argparse
import asyncio
import gzip
import json
import math
import os
import random
import time
import uuid

from huggingface_hub import hf_hub_download, snapshot_download

PREDICATES = [
    "Is this paper primarily about machine learning?",
    "Is this paper about computer vision or image analysis?",
    "Is this paper about natural language processing or computational linguistics?",
    "Is this paper about quantum physics or quantum information?",
    "Is this paper about astronomy or astrophysics?",
    "Is this paper about condensed matter physics?",
    "Is this paper primarily a pure or applied mathematics paper?",
    "Is this paper about high-energy or particle physics?",
    "Does the abstract report experimental or numerical results?",
    "Does the paper propose a new algorithm or method?",
    "Is the work primarily theoretical, with proofs or analytical derivations?",
    "Does the abstract mention a dataset, survey, or observational data?",
    "Does the paper study a physical system at low temperature?",
    "Does the abstract mention neural networks or deep learning?",
    "Does the paper involve graphs or networks?",
    "Does the abstract mention an application to biology or medicine?",
]
PREAMBLE = "Read the following paper abstract and answer the question that follows."
YESNO = "Answer with a single word, Yes or No."
SCIFACT_PREDICATES = [
    "Does the abstract describe a study in humans (patients or participants)?",
    "Does the abstract describe experiments in mice or other animals?",
    "Does the abstract report a statistically significant finding?",
    "Is the abstract about molecular or cellular mechanisms?",
    "Is the abstract about cancer or tumors?",
    "Is the abstract about infection, viruses, or bacteria?",
    "Is the abstract about the heart or blood vessels?",
    "Is the abstract about the brain or nervous system?",
    "Does the abstract describe a randomized controlled trial?",
    "Does the abstract mention a drug or pharmacological treatment?",
    "Is the abstract about genes, DNA, or gene expression?",
    "Is the abstract about the immune system?",
    "Is the abstract about diabetes, obesity, or metabolism?",
    "Does the abstract describe a meta-analysis or systematic review?",
    "Is the abstract about pregnancy, children, or infants?",
    "Does the abstract mention aging or older adults?",
]
PUBMED_PREDICATES = [p.replace("the abstract", "the text") for p in SCIFACT_PREDICATES]
PUBMED_PREAMBLE = "Read the following excerpt from a biomedical research article and answer the question that follows."
EXTRACTIONS = [
    "In at most eight words, name the main topic of this paper.",
    "In at most eight words, name the main method or approach used.",
    "In at most eight words, name the main object, system, or population studied.",
    "In at most eight words, state the main result or finding.",
]
EXTRACT_PRE = "Read the following paper abstract and answer the question that follows briefly."
CFG = dict(family="filter", corpus="arxiv")


def load_docs(n, seed, scan_lines=400000):
    p = hf_hub_download("gfissore/arxiv-abstracts-2021", "arxiv-abstracts.jsonl.gz", repo_type="dataset")
    rows = []
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= scan_lines:
                break
            r = json.loads(line)
            ab = " ".join((r.get("abstract") or "").split())
            if 400 <= len(ab) <= 2000:
                rows.append(dict(id=r.get("id"), title=r.get("title", ""), abstract=ab))
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]
    rows.sort(key=lambda r: r["id"])  # arXiv id order ~ submission date, so ranges act like date ranges
    return rows


def load_scifact_docs(n, seed):
    import pandas as pd
    from huggingface_hub import list_repo_files
    files = list_repo_files("BeIR/scifact", repo_type="dataset")
    cf = next(f for f in files if "corpus" in f and f.endswith(".parquet"))
    df = pd.read_parquet(hf_hub_download("BeIR/scifact", cf, repo_type="dataset")).fillna("")
    rows = [dict(id=str(r["_id"]), title=r["title"], abstract=" ".join(str(r["text"]).split())[:2000])
            for r in df.to_dict("records") if len(str(r["text"])) >= 400]
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]
    rows.sort(key=lambda r: r["id"])
    return rows


def load_pubmed_docs(n, seed):
    import pandas as pd
    from huggingface_hub import list_repo_files
    files = list_repo_files("ccdv/pubmed-summarization", repo_type="dataset")
    f = next(x for x in files if x.startswith("document/test") and x.endswith(".parquet"))
    df = pd.read_parquet(hf_hub_download("ccdv/pubmed-summarization", f, repo_type="dataset")).fillna("")
    rows = []
    for i, r in enumerate(df.to_dict("records")):
        art = " ".join(str(r["article"]).split())
        if len(art) >= 3000:
            rows.append(dict(id=f"pm{i:06d}", title="", abstract=art[:3000]))
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]
    rows.sort(key=lambda r: r["id"])
    return rows


class Engine:
    """vLLM AsyncLLM backend."""

    def __init__(self, model, blocks, max_len=1024, apc=True):
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.sampling_params import RequestOutputKind
        from vllm.v1.engine.async_llm import AsyncLLM
        self.args = AsyncEngineArgs(model=snapshot_download(model), enable_prefix_caching=apc,
                                    max_model_len=max_len, gpu_memory_utilization=0.88,
                                    num_gpu_blocks_override=blocks, seed=0, disable_log_stats=True,
                                    max_num_batched_tokens=2048, max_num_seqs=256)
        self.llm = AsyncLLM.from_engine_args(self.args)
        self.sp = SamplingParams(temperature=0.0, max_tokens=1, output_kind=RequestOutputKind.FINAL_ONLY)
        self.spf = SamplingParams(temperature=0.0, max_tokens=64, output_kind=RequestOutputKind.FINAL_ONLY)
        self.spx = SamplingParams(temperature=0.0, max_tokens=24, output_kind=RequestOutputKind.FINAL_ONLY)
        self.stats = None

    def reset_stats(self):
        self.stats = dict(requests=0, prompt_tokens=0, cached_tokens=0)

    async def call(self, ids, fused=False, extract=False):
        out = None
        sp = self.spx if extract else (self.spf if fused else self.sp)
        async for o in self.llm.generate({"prompt_token_ids": ids}, sp, uuid.uuid4().hex):
            out = o
        self.stats["requests"] += 1
        self.stats["prompt_tokens"] += len(ids)
        self.stats["cached_tokens"] += out.num_cached_tokens or 0
        return out.outputs[0].text

    async def reset(self):
        ok = await self.llm.reset_prefix_cache()
        assert ok is not False, "vLLM prefix-cache reset failed"

    def tokenizer(self):
        return self.llm.get_tokenizer()

    def shutdown(self):
        self.llm.shutdown()


class SGLEngine:
    """SGLang backend with the same interface; KV pool sized in tokens to match vLLM."""

    def __init__(self, model, kv_tokens, policy, max_len=1024):
        import sglang as sgl
        from transformers import AutoTokenizer
        path = snapshot_download(model)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.llm = sgl.Engine(model_path=path, schedule_policy=policy, max_total_tokens=kv_tokens,
                              context_length=max_len, random_seed=0, log_level="error")
        self.sp = {"temperature": 0.0, "max_new_tokens": 1}
        self.spf = {"temperature": 0.0, "max_new_tokens": 64}
        self.spx = {"temperature": 0.0, "max_new_tokens": 24}
        self.stats = None

    def reset_stats(self):
        self.stats = dict(requests=0, prompt_tokens=0, cached_tokens=0)

    async def call(self, ids, fused=False, extract=False):
        sp = self.spx if extract else (self.spf if fused else self.sp)
        out = await self.llm.async_generate(input_ids=ids, sampling_params=sp)
        self.stats["requests"] += 1
        self.stats["prompt_tokens"] += len(ids)
        self.stats["cached_tokens"] += int(out.get("meta_info", {}).get("cached_tokens", 0) or 0)
        return out["text"]

    async def reset(self):
        # Engine.flush_cache() drives its own event loop; call the async tokenizer-manager API instead.
        out = await self.llm.tokenizer_manager.flush_cache()
        assert getattr(out, "success", True), "SGLang cache flush failed"

    def tokenizer(self):
        return self.tok

    def shutdown(self):
        self.llm.shutdown()


class Prompts:
    def __init__(self, tok, docs):
        self.tok, self.docs, self.cache = tok, docs, {}

    def _chat(self, user):
        t = self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        return self.tok.encode(t, add_special_tokens=False)

    def single(self, d, p):
        k = (d, p)
        if k not in self.cache:
            doc = self.docs[d]
            if CFG["corpus"] == "pubmed":
                body = f"{PUBMED_PREAMBLE}\n\nArticle excerpt:\n{doc['abstract']}"
            else:
                body = f"{PREAMBLE}\n\nPaper:\nTitle: {doc['title']}\nAbstract: {doc['abstract']}"
            self.cache[k] = self._chat(f"{body}\n\nQuestion: {PREDICATES[p]} {YESNO}")
        return self.cache[k]

    def extract(self, d, e):
        # Same document prefix as single(): only the preamble line differs before the document.
        k = ("x", d, e)
        if k not in self.cache:
            doc = self.docs[d]
            self.cache[k] = self._chat(f"{PREAMBLE}\n\nPaper:\nTitle: {doc['title']}\nAbstract: {doc['abstract']}"
                                       f"\n\nQuestion: {EXTRACTIONS[e]}")
        return self.cache[k]

    def fused(self, d, ps):
        doc = self.docs[d]
        qs = "\n".join(f"Q{i + 1}. {PREDICATES[p]}" for i, p in enumerate(ps))
        return self._chat(f"Read the following paper abstract and answer each question with Yes or No.\n\nPaper:\n"
                          f"Title: {doc['title']}\nAbstract: {doc['abstract']}\n\nQuestions:\n{qs}\n\n"
                          f"Reply with exactly {len(ps)} lines in the form 'Q1: Yes'.")

    def doc_tokens(self, d):
        return len(self.single(d, 0))


def make_trace(n_docs, m, seed, rate, overlap="high"):
    rng = random.Random(seed)
    t, qs = 0.0, []
    lo_f, hi_f = {"high": (0.5, 1.0), "mid": (0.3, 1.0), "low": (0.05, 0.2)}[overlap]
    for i in range(m):
        if i > 0:
            t += rng.expovariate(rate)
        frac = rng.uniform(lo_f, hi_f)
        lo = rng.randrange(n_docs)
        p1 = rng.randrange(len(PREDICATES))
        if CFG["family"] == "extract":
            p2 = rng.randrange(len(EXTRACTIONS))  # every query extracts on documents passing its filter
        else:
            p2 = rng.choice([p for p in range(len(PREDICATES)) if p != p1]) if rng.random() < 0.5 else None
        qs.append(dict(qid=i, arrival=t, p1=p1, p2=p2, lo=lo, n=int(frac * n_docs)))
    return qs


async def stage2_label(eng, pr, q, d):
    """Stage-2 call: a yes/no filter, or (extract family) a short extraction with longer decode."""
    if CFG["family"] == "extract":
        txt = await eng.call(pr.extract(d, q["p2"]), extract=True)
        return " ".join(txt.strip().lower().split())
    txt = await eng.call(pr.single(d, q["p2"]))
    return 1 if txt.strip().lower().startswith("yes") else 0


def docs_of(q, n_docs):
    return [(q["lo"] + k) % n_docs for k in range(q["n"])]


class Run:
    """Bookkeeping shared by all policies for one trace."""

    def __init__(self, eng, pr, trace, n_docs):
        self.eng, self.pr, self.trace, self.N = eng, pr, trace, n_docs
        self.labels = {}  # (qid, doc, stage) -> 0/1
        self.done_t = {}
        self.t0 = None

    def now(self):
        return time.monotonic() - self.t0

    async def wait_arrival(self, q):
        dt = q["arrival"] - self.now()
        if dt > 0:
            await asyncio.sleep(dt)

    async def eval_doc(self, q, d):
        txt = await self.eng.call(self.pr.single(d, q["p1"]))
        y = 1 if txt.strip().lower().startswith("yes") else 0
        self.labels[(q["qid"], d, 1)] = y
        if y and q["p2"] is not None:
            self.labels[(q["qid"], d, 2)] = await stage2_label(self.eng, self.pr, q, d)

    def finish(self, q):
        self.done_t[q["qid"]] = self.now()


async def p_serial(R):
    for q in R.trace:
        await R.wait_arrival(q)
        await asyncio.gather(*(R.eval_doc(q, d) for d in docs_of(q, R.N)))
        R.finish(q)


async def p_eager(R):
    async def one(q):
        await R.wait_arrival(q)
        await asyncio.gather(*(R.eval_doc(q, d) for d in docs_of(q, R.N)))
        R.finish(q)
    await asyncio.gather(*(one(q) for q in R.trace))


async def p_window(R, W):
    pending, tasks = list(R.trace), []

    async def batch(qs):
        need = {}
        for q in qs:
            for d in docs_of(q, R.N):
                need.setdefault(d, []).append(q)
        coros = {q["qid"]: [] for q in qs}
        for d in sorted(need):  # document-major release order
            for q in need[d]:
                coros[q["qid"]].append(asyncio.ensure_future(R.eval_doc(q, d)))

        async def fin(q):
            await asyncio.gather(*coros[q["qid"]])
            R.finish(q)
        await asyncio.gather(*(fin(q) for q in qs))

    k = 1
    while pending:
        await asyncio.sleep(max(0.0, k * W - R.now()))
        k += 1
        ready = [q for q in pending if q["arrival"] <= R.now()]
        if ready:
            pending = [q for q in pending if q not in ready]
            tasks.append(asyncio.ensure_future(batch(ready)))
    await asyncio.gather(*tasks)


async def p_adaptive(R):
    """Batch when idle: whenever the previous batch has no outstanding stage-1 work, release every query that
    has arrived and not yet been released, document-major (as the window policy does). Stage-2 requests
    follow each positive stage-1 label immediately and may overlap the next batch, as in P6."""
    pending, tasks = list(R.trace), []

    async def batch(qs):
        need = {}
        for q in qs:
            for d in docs_of(q, R.N):
                need.setdefault(d, []).append(q)
        left = [sum(len(v) for v in need.values())]
        s1_done = asyncio.Event()
        if not left[0]:
            s1_done.set()

        async def one(q, d):
            txt = await R.eng.call(R.pr.single(d, q["p1"]))
            y = 1 if txt.strip().lower().startswith("yes") else 0
            R.labels[(q["qid"], d, 1)] = y
            left[0] -= 1
            if left[0] == 0:
                s1_done.set()
            if y and q["p2"] is not None:
                R.labels[(q["qid"], d, 2)] = await stage2_label(R.eng, R.pr, q, d)
        coros = {q["qid"]: [] for q in qs}
        for d in sorted(need):  # document-major release order
            for q in need[d]:
                coros[q["qid"]].append(asyncio.ensure_future(one(q, d)))

        async def fin(q):
            await asyncio.gather(*coros[q["qid"]])
            R.finish(q)
        bt = asyncio.ensure_future(asyncio.gather(*(fin(q) for q in qs)))
        tasks.append(bt)
        w = asyncio.ensure_future(s1_done.wait())
        await asyncio.wait({w, bt}, return_when=asyncio.FIRST_COMPLETED)
        if bt.done() and bt.exception() is not None:  # a failed request must not leave the barrier waiting
            w.cancel()
            raise bt.exception()
        await w

    while pending:
        now = R.now()
        ready = [q for q in pending if q["arrival"] <= now]
        if not ready:
            await asyncio.sleep(max(0.0, min(q["arrival"] for q in pending) - now))
            continue
        pending = [q for q in pending if q not in ready]
        await batch(ready)
    await asyncio.gather(*tasks)


async def p_scan(R, block, inflight, smart=False, always_drain=False, admit=None, fused=False,
                 kv_tokens=None, aging=30.0):
    """Shared block scan over the corpus.

    In-flight accounting counts stage-1 work per block; stage-2 requests are tracked
    separately per block. P6 (smart=False) releases the next block whenever fewer than
    `inflight` blocks have stage-1 work outstanding and never waits for stage 2 (advance).
    always_drain waits for all stage-2 work before releasing a new block. smart=True (P3)
    orders blocks by relevance with aging, enforces the admission limit and decides
    drain-or-advance from the estimated KV footprint of blocks with outstanding work.
    """
    nb = math.ceil(R.N / block)
    need, qdocs, qpend, attach_t = {}, {}, {}, {}
    waiting, active, finished = [], [], set()
    pos = 0
    s1_inflight = set()          # blocks with outstanding stage-1 work
    s2_out = {}                  # block -> outstanding stage-2 requests
    block_tok = [sum(R.pr.doc_tokens(d) for d in range(b * block, min(R.N, (b + 1) * block))) for b in range(nb)]
    wake = asyncio.Event()
    tasks = []

    async def arrivals():
        for q in R.trace:
            await R.wait_arrival(q)
            waiting.append(q)
            wake.set()

    def admit_waiting():
        while waiting and (admit is None or len(active) < admit):
            q = waiting.pop(0)
            active.append(q)
            qdocs[q["qid"]] = set(docs_of(q, R.N))
            need[q["qid"]] = {d // block for d in qdocs[q["qid"]]}
            qpend[q["qid"]] = 0
            attach_t[q["qid"]] = R.now()

    async def stage2(q, d, b):
        try:
            R.labels[(q["qid"], d, 2)] = await stage2_label(R.eng, R.pr, q, d)
        finally:
            s2_out[b] -= 1
            qpend[q["qid"]] -= 1
            wake.set()

    def spawn_stage2(q, d, b):
        s2_out[b] = s2_out.get(b, 0) + 1
        qpend[q["qid"]] += 1
        tasks.append(asyncio.ensure_future(stage2(q, d, b)))

    async def stage1_block(b, qs):
        try:
            await stage1_block_body(b, qs)
        finally:
            for q in qs:
                qpend[q["qid"]] -= 1
            s1_inflight.discard(b)
            wake.set()

    async def stage1_block_body(b, qs):
        docs = list(range(b * block, min(R.N, (b + 1) * block)))
        if fused:
            # (5) Fused predicate set per document: only queries whose range contains it.
            dps = {d: sorted({q["p1"] for q in qs if d in qdocs[q["qid"]]}) for d in docs}
            docs = [d for d in docs if dps[d]]
            outs = await asyncio.gather(*(R.eng.call(R.pr.fused(d, dps[d]), fused=True) for d in docs))
            for d, txt in zip(docs, outs):
                ps = dps[d]
                row = {}
                for line in txt.splitlines():
                    line = line.strip()
                    if line[:1].upper() == "Q" and ":" in line:
                        try:
                            row[ps[int(line[1:line.index(":")]) - 1]] = int(
                                line.split(":", 1)[1].strip().lower().startswith("yes"))
                        except (ValueError, IndexError):
                            pass
                for q in qs:
                    if d in qdocs[q["qid"]]:
                        y = row.get(q["p1"], 0)
                        R.labels[(q["qid"], d, 1)] = y
                        if y and q["p2"] is not None:
                            spawn_stage2(q, d, b)
        else:
            async def one(q, d):
                txt = await R.eng.call(R.pr.single(d, q["p1"]))
                y = 1 if txt.strip().lower().startswith("yes") else 0
                R.labels[(q["qid"], d, 1)] = y
                if y and q["p2"] is not None:
                    spawn_stage2(q, d, b)
            await asyncio.gather(*(one(q, d) for q in qs for d in docs if d in qdocs[q["qid"]]))

    def demand():
        dm = {}
        for q in active:
            for b in need[q["qid"]]:
                if b not in s1_inflight:
                    dm[b] = dm.get(b, 0) + 1
        return dm

    def pick_block(dm):
        if not smart:
            for k in range(nb):  # circular order from the scan position, skipping empty blocks
                b = (pos + k) % nb
                if b in dm:
                    return b
        old = [q for q in active if R.now() - attach_t[q["qid"]] > aging and need[q["qid"]]]
        if old:  # aging: serve the longest-attached query's blocks first
            o = min(old, key=lambda q: attach_t[q["qid"]])
            cand = [b for b in dm if b in need[o["qid"]]]
            if cand:
                return max(cand, key=lambda b: (dm[b], -((b - pos) % nb)))
        return max(dm, key=lambda b: (dm[b], -((b - pos) % nb)))

    def footprint():
        live = s1_inflight | {b for b, c in s2_out.items() if c > 0}
        return sum(block_tok[b] for b in live)

    arr = asyncio.ensure_future(arrivals())
    while len(finished) < len(R.trace):
        for t in tasks:
            if t.done() and not t.cancelled() and t.exception() is not None:
                raise t.exception()
        admit_waiting()
        for q in list(active):
            if not need[q["qid"]] and qpend[q["qid"]] == 0:
                active.remove(q)
                finished.add(q["qid"])
                R.finish(q)
                wake.set()
        admit_waiting()
        dm = demand()
        release = bool(dm) and len(s1_inflight) < inflight
        if release and always_drain and (s1_inflight or any(c > 0 for c in s2_out.values())):
            release = False
        if release and smart and not always_drain and kv_tokens is not None and (s1_inflight or s2_out):
            nxt = pick_block(dm)
            fp = footprint()
            if fp > 0 and fp + block_tok[nxt] > kv_tokens:
                release = False  # drain: wait for outstanding work before evicting live prefixes
        if release:
            b = pick_block(dm)
            qs = [q for q in active if b in need[q["qid"]]]
            for q in qs:
                need[q["qid"]].discard(b)
                qpend[q["qid"]] += 1
            s1_inflight.add(b)
            pos = (b + 1) % nb
            tasks.append(asyncio.ensure_future(stage1_block(b, qs)))
            continue
        wake.clear()
        try:
            await asyncio.wait_for(wake.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            pass
    await arr
    await asyncio.gather(*tasks)


async def run_policy(eng, pr, trace, n_docs, name, fn):
    await eng.reset()
    eng.reset_stats()
    R = Run(eng, pr, trace, n_docs)
    R.t0 = time.monotonic()
    await fn(R)
    makespan = max(R.done_t.values())
    lat = [R.done_t[q["qid"]] - q["arrival"] for q in trace]
    lat_sorted = sorted(lat)
    res = dict(policy=name, makespan=makespan, mean_latency=sum(lat) / len(lat),
               p50=lat_sorted[len(lat) // 2], p95=lat_sorted[min(len(lat) - 1, int(math.ceil(0.95 * len(lat))) - 1)],
               max_latency=lat_sorted[-1], completed=len(R.done_t), latencies=lat, **eng.stats)
    res["computed_prefill"] = res["prompt_tokens"] - res["cached_tokens"]
    return res, R.labels


def agreement(ref, lab, trace, n_docs):
    """Stage-1 agreement over every pre-registered (query, document) pair; stage-2 agreement
    conditional on both runs passing stage 1 (denominator reported); identical final results."""
    s1_keys = [(q["qid"], d, 1) for q in trace for d in docs_of(q, n_docs)]
    s1_same = sum(1 for k in s1_keys if ref.get(k) == lab.get(k))
    s2_keys = [(q["qid"], d, 2) for q in trace if q["p2"] is not None for d in docs_of(q, n_docs)
               if ref.get((q["qid"], d, 1)) == 1 and lab.get((q["qid"], d, 1)) == 1]
    s2_same = sum(1 for k in s2_keys if ref.get(k) == lab.get(k))
    final_same = 0
    for q in trace:
        if CFG["family"] == "extract":
            def final(L):
                return {(d, L.get((q["qid"], d, 2))) for d in docs_of(q, n_docs) if L.get((q["qid"], d, 1)) == 1}
        else:
            def final(L):
                return {d for d in docs_of(q, n_docs)
                        if L.get((q["qid"], d, 1)) == 1 and (q["p2"] is None or L.get((q["qid"], d, 2)) == 1)}
        final_same += int(final(ref) == final(lab))
    return dict(stage1_agreement=s1_same / max(1, len(s1_keys)), stage1_pairs=len(s1_keys),
                stage2_agreement=(s2_same / len(s2_keys)) if s2_keys else None, stage2_pairs=len(s2_keys),
                queries_with_identical_result=final_same / len(trace))


def common(a_, b_):
    k = 0
    while k < min(len(a_), len(b_)) and a_[k] == b_[k]:
        k += 1
    return k


def setup_corpus(corpus):
    """Select the predicate set and prompt format of a corpus."""
    global PREDICATES
    CFG["corpus"] = corpus
    PREDICATES = {"scifact": SCIFACT_PREDICATES, "pubmed": PUBMED_PREDICATES}.get(corpus, PREDICATES)


def token_stats(pr, n):
    """Prompt lengths, tokens a document's prompts share across questions (preamble + document), and
    the preamble shared by every prompt; inputs of the cache/queue models (sscan_model*.py)."""
    doc_tok = [pr.doc_tokens(d) for d in range(n)]
    shared_prefix = [common(pr.single(d, 0), pr.single(d, 1)) for d in range(n)]
    return doc_tok, shared_prefix, common(pr.single(0, 0), pr.single(1, 0))


def p6_params(name, default_block):
    """Block size and blocks in flight of a SharedScan policy name P6_lazyscan[_b<N>][_F<k>]."""
    assert name.startswith("P6_lazyscan"), name
    bs, fl = default_block, 2
    for part in name[len("P6_lazyscan"):].split("_")[1:]:
        if part[:1] == "b":
            bs = int(part[1:])
        elif part[:1] == "F":
            fl = int(part[1:])
        else:
            raise ValueError(name)
    return bs, fl


def sha256_file(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


async def main_async(a):
    frozen = None
    if a.freeze:
        assert sha256_file(a.freeze) == a.freeze_sha, "freeze file does not match the recorded sha256"
        F = json.load(open(a.freeze))
        cell = F["cells"][a.cell]
        corp = F["corpora"][cell["corpus"]]
        assert a.seed in cell["seeds"], f"seed {a.seed} not pre-registered for {a.cell}"
        a.corpus, a.blocks, a.load, a.m, a.max_len = cell["corpus"], cell["blocks"], cell["load"], cell["m"], cell["max_len"]
        a.backend, a.overlap, a.family = cell["backend"], cell["overlap"], cell["family"]
        a.block, a.sgl_policy, a.no_apc, a.model = cell["p6_block"], cell["sgl_policy"], not cell.get("apc", True), F["model"]
        a.policies = ",".join(cell["policies"])
        frozen = dict(cell=cell, corp=corp, trace=F["traces"][cell["trace"]][str(a.seed)], model_revision=F["model_revision"])
        setup_corpus(a.corpus)
        docs = corp["docs"]
    else:
        setup_corpus(a.corpus)
        docs = {"scifact": load_scifact_docs, "pubmed": load_pubmed_docs}.get(a.corpus, load_docs)(a.n, 0)
    CFG["family"] = a.family  # after the frozen settings are applied
    kv_tokens = a.blocks * 16
    if a.backend == "vllm":
        eng = Engine(a.model, a.blocks, max_len=a.max_len, apc=not a.no_apc)
    else:
        eng = SGLEngine(a.model, kv_tokens, a.sgl_policy, max_len=a.max_len)
    pr = Prompts(eng.tokenizer(), docs)
    model_revision = os.path.basename(snapshot_download(a.model).rstrip("/"))
    doc_tok, shared_prefix, preamble_tok = token_stats(pr, len(docs))
    if frozen:
        assert model_revision == frozen["model_revision"], "model snapshot differs from the frozen revision"
        c = frozen["corp"]
        assert (doc_tok, shared_prefix, preamble_tok) == (c["doc_tokens"], c["shared_prefix_tokens"], c["preamble_tokens"]), \
            "prompts do not reproduce the frozen token counts"
    assert max(doc_tok) < a.max_len, "a prompt exceeds the maximum model length"
    out = dict(backend=a.backend, sgl_policy=a.sgl_policy if a.backend == "sglang" else None, model=a.model,
               corpus=a.corpus, family=a.family, overlap=a.overlap, apc=not a.no_apc, n_docs=len(docs),
               m=a.m, blocks=a.blocks, kv_tokens=kv_tokens, load=a.load, block=a.block, max_len=a.max_len,
               model_revision=model_revision, p6_block=a.block, default_inflight=2,  # per-run block/inflight recorded below
               cell=a.cell if frozen else None, freeze_sha256=a.freeze_sha if frozen else None,
               corpus_prefix_tokens=sum(doc_tok), doc_tokens=doc_tok, shared_prefix_tokens=shared_prefix,
               preamble_tokens=preamble_tok, runs=[])
    print("corpus prefix tokens", out["corpus_prefix_tokens"], "kv tokens", kv_tokens, flush=True)
    # Warm-up: one full-corpus single-stage query in isolation. Without --freeze it also sets the
    # arrival rate; with --freeze the frozen trace is used and this time is only recorded.
    eng.reset_stats()
    cal = [dict(qid=0, arrival=0.0, p1=0, p2=None, lo=0, n=len(docs))]
    R = Run(eng, pr, cal, len(docs))
    R.t0 = time.monotonic()
    await p_eager(R)
    t1 = R.now()
    out["calibration_single_query_s"] = t1
    mean_frac = {"high": 0.75, "mid": 0.65, "low": 0.125}[a.overlap]
    print("calibration", t1, flush=True)
    all_policies = {
        "P1_eager": p_eager,
        "P1_eager_rerun": p_eager,
        "P2_window30": lambda R: p_window(R, 30.0),
        "P2_window10": lambda R: p_window(R, 10.0),
        "P2_window60": lambda R: p_window(R, 60.0),
        "P2_adaptive": p_adaptive,
        "P6_lazyscan": lambda R: p_scan(R, a.block, 2),
        "P6_lazyscan_b16": lambda R: p_scan(R, 16, 2),
        "P6_lazyscan_b64": lambda R: p_scan(R, 64, 2),
        "P0_serial": p_serial,
    }
    names = a.policies.split(",") if a.policies else ["P1_eager", "P1_eager_rerun", "P2_window30", "P6_lazyscan"]
    for nm in names:  # P6 with any block size and blocks in flight: P6_lazyscan[_b<N>][_F<k>]
        if nm.startswith("P6_lazyscan") and nm not in all_policies:
            bs, fl = p6_params(nm, a.block)
            all_policies[nm] = (lambda bs, fl: (lambda R: p_scan(R, bs, fl)))(bs, fl)
    assert names[0] == "P1_eager", "P1_eager must run first: it is the quality reference"
    seeds = [a.seed] if frozen else list(range(a.seed0, a.seed0 + a.seeds))
    for seed in seeds:
        if frozen:
            trace = [dict(q) for q in frozen["trace"]]
            out["rate_frozen"] = frozen["cell"]["rate"]
        else:
            trace = make_trace(len(docs), a.m, 1000 + seed, a.load / (t1 * mean_frac), a.overlap)
        ref = None
        for name in names:
            res, lab = await run_policy(eng, pr, trace, len(docs), name, all_policies[name])
            if name == "P1_eager":
                ref = lab  # quality reference
            res.update(agreement(ref, lab, trace, len(docs)))
            if name.startswith("P6_lazyscan"):
                res["block"], res["inflight"] = p6_params(name, a.block)
            res.update(seed=seed, load=a.load, trace=[{k: q[k] for k in ("qid", "arrival", "lo", "n", "p1", "p2")} for q in trace])
            if a.save_labels:
                res["labels"] = [[k[0], k[1], k[2], v] for k, v in sorted(lab.items())]
            out["runs"].append(res)
            print(json.dumps({k: v for k, v in res.items() if k not in ("trace", "labels")}), flush=True)
            with open(a.out + ".tmp", "w") as f:  # atomic replace: a killed job never leaves truncated JSON
                json.dump(out, f)
            os.replace(a.out + ".tmp", a.out)
    eng.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--corpus", default="arxiv", choices=["arxiv", "scifact", "pubmed"])
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--family", default="filter", choices=["filter", "extract"])
    ap.add_argument("--overlap", default="high", choices=["high", "mid", "low"])
    ap.add_argument("--no-apc", action="store_true")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--m", type=int, default=60)
    ap.add_argument("--blocks", type=int, default=800)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--load", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "sglang"])
    ap.add_argument("--sgl-policy", default="lpm")
    ap.add_argument("--policies", default="")
    ap.add_argument("--freeze", default="", help="frozen sweep inputs (sweep_freeze.py output)")
    ap.add_argument("--freeze-sha", default="", help="sha256 of the sweep freeze file")
    ap.add_argument("--cell", default="")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save-labels", action="store_true")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
