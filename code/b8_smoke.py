#!/usr/bin/env python3
"""B8 smoke check: engine starts at 10,000 KV blocks, and the extended predicates get Yes/No first tokens.
Usage: b8_smoke.py --freeze <b8_freeze.json> --freeze-sha <sha> --out <json>"""
import argparse
import asyncio
import collections
import hashlib
import json

import b8_predicates as P
import sweep_live_b5 as H


async def run(a):
    F = json.load(open(a.freeze))
    assert hashlib.sha256(open(a.freeze, "rb").read()).hexdigest() == a.freeze_sha
    out = dict(model=F["model"])
    eng = H.Engine(F["model"], 10000, max_len=2048, apc=True)
    eng.reset_stats()
    for c, extra in (("arxiv", P.ARXIV_EXTRA), ("pubmed", P.PUBMED_EXTRA)):
        H.PREDICATES = list(H.PREDICATES)
        H.setup_corpus(c)
        base = list(H.PREDICATES)
        H.PREDICATES = base + extra
        pr = H.Prompts(eng.tokenizer(), F["corpora"][c]["docs"])
        tasks = [eng.call(pr.single(d, len(base) + i)) for i in range(len(extra)) for d in range(0, 1000, 100)]
        txt = await asyncio.gather(*tasks)
        first = collections.Counter(t.strip().lower()[:3] for t in txt)
        yes_per_pred = [sum(t.strip().lower().startswith("yes") for t in txt[i * 10:(i + 1) * 10]) / 10 for i in range(len(extra))]
        out[c] = dict(n=len(txt), yes=first["yes"], no=first["no"], other=len(txt) - first["yes"] - first["no"],
                      other_examples=[t for t in txt if t.strip().lower()[:3] not in ("yes", "no")][:10],
                      mean_yes_rate=sum(yes_per_pred) / len(yes_per_pred))
        H.PREDICATES = list(H.PREDICATES[:16])
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out)[:1500], flush=True)
    eng.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--freeze-sha", required=True)
    ap.add_argument("--out", required=True)
    asyncio.run(run(ap.parse_args()))
