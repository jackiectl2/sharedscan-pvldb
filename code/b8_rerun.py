#!/usr/bin/env python3
"""Same-configuration eager rerun of a frozen B8 cell: b8_live.py's path with the selected cell's policy list replaced by
P1_eager, P1_eager_rerun. Both run on the cell's frozen trace in one engine, so the rerun's agreement with P1_eager is a
within-job reference for that cell, measured by the harness's agreement(). The harness and freeze are not edited.
Usage: b8_rerun.py --freeze <b8_freeze.json> --freeze-sha <sha> --cell <cell> --seed <s> --save-labels --out <json>"""
import hashlib
import json
import os
import sys

import b8_live
import sweep_live_b5 as H

POLICIES = ["P1_eager", "P1_eager_rerun"]
RUNTIME_CODE = ["sweep_live_b5.py", "b8_live.py", "b8_predicates.py"]  # the code this run executes, as hashed in the freeze


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def arg(name):
    return sys.argv[sys.argv.index(name) + 1]


def main():
    assert "--policies" not in sys.argv, "the policy list is fixed by this script"
    freeze, cell, out = os.path.realpath(arg("--freeze")), arg("--cell"), arg("--out")
    assert not os.path.exists(out), f"{out} exists; refusing to overwrite"
    here = os.path.dirname(os.path.abspath(__file__))
    frozen = json.load(open(freeze))["code_sha256"]
    code = {f: sha(os.path.join(here, f)) for f in RUNTIME_CODE}
    assert all(code[f] == frozen[f] for f in RUNTIME_CODE), "runtime code differs from the freeze"
    load = H.json.load

    def load_once(f, *args, **kw):
        obj = load(f, *args, **kw)
        if os.path.realpath(getattr(f, "name", "")) == freeze:
            H.json.load = load
            obj["cells"][cell]["policies"] = list(POLICIES)
        return obj

    H.json.load = load_once
    b8_live.main()
    assert H.json.load is load, "the freeze was never read through the patched loader"
    d = json.load(open(out))
    d.update(policy_override=POLICIES, code_sha256_verified=code, rerun_adapter_sha256=sha(os.path.abspath(__file__)))
    with open(out + ".tmp", "w") as f:
        json.dump(d, f)
    os.replace(out + ".tmp", out)


if __name__ == "__main__":
    main()
