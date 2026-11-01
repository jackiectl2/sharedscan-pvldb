import json, os, sys
R = sys.argv[1]
done = []
for blk, fn in (("b5", "b5_freeze.json"), ("b6", "b6_freeze.json"), ("b7", "b7_freeze.json")):
    try:
        F = json.load(open(os.path.join(R, blk, fn)))
    except Exception:
        continue
    for name, c in F["cells"].items():
        for s in c["seeds"]:
            try:
                X = json.load(open(os.path.join(R, blk, f"{name}_s{s}.json")))
                if [r["policy"] for r in X["runs"]] == c["policies"]:
                    done.append(f"{name}_s{s}")
            except Exception:
                pass
print(len(done), " ".join(done))
