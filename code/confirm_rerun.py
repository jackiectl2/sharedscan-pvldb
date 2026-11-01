#!/usr/bin/env python3
"""Same-configuration eager rerun of a frozen confirmatory cell (rerun reference for answer agreement).

Runs confirm_live.py's frozen path with one change: the selected cell's policy list becomes P1_eager,
P1_eager_rerun. Both run on the cell's frozen trace in the same engine, so the rerun's agreement with
P1_eager is a within-job rerun reference for that cell, measured as in B6 (sweep_live_b5.py).
confirm_live.py is imported, not edited, so its sha256 recorded in the freeze still matches. The
override is recorded in the output (policy_override, rerun_adapter_sha256).
Usage: confirm_rerun.py --freeze <confirm_freeze.json> --freeze-sha <sha> --cell <cell> --seed <s> --save-labels --out <json>
"""
import hashlib
import json
import os
import sys

import confirm_live as H

POLICIES = ["P1_eager", "P1_eager_rerun"]
FREEZE_LIVE_SHA = "f1bc7602d5be5a0058788ee37eeaef6fcfcbe72a3bd86975ce1c34580a4a89e9"  # confirm_live.py in the freeze


def arg(name):
    return sys.argv[sys.argv.index(name) + 1]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    assert sha(H.__file__) == FREEZE_LIVE_SHA, "confirm_live.py differs from the frozen version"
    assert "--policies" not in sys.argv, "the policy list is fixed by this script"
    freeze, cell, out = os.path.realpath(arg("--freeze")), arg("--cell"), arg("--out")
    assert not os.path.exists(out), f"{out} exists; refusing to overwrite"
    load = H.json.load

    def load_once(f, *args, **kw):
        """Replace the selected cell's policy list when the freeze is read (confirm_live.py:604), then restore."""
        obj = load(f, *args, **kw)
        if os.path.realpath(getattr(f, "name", "")) == freeze:
            H.json.load = load
            obj["cells"][cell]["policies"] = list(POLICIES)
        return obj

    H.json.load = load_once
    H.main()
    assert H.json.load is load, "the freeze was never read through the patched loader"
    d = json.load(open(out))
    d.update(policy_override=POLICIES, rerun_adapter_sha256=sha(os.path.abspath(__file__)), runner_sha256=FREEZE_LIVE_SHA)
    with open(out + ".tmp", "w") as f:
        json.dump(d, f)
    os.replace(out + ".tmp", out)


if __name__ == "__main__":
    main()
