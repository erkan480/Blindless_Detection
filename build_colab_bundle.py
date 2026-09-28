"""Package portable training code and existing audited split metadata; no images/secrets."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile


def build(root, output):
    root = Path(root).resolve()
    output = Path(output)
    sources = ["train_colab.py", "src/ai/retina.py", "src/ai/preprocessing.py", "COLAB.md", "LICENSE"]
    payload = {name: (root / name).read_bytes() for name in sources}
    for name in ("src/__init__.py", "src/ai/__init__.py"):
        payload[name] = b""
    splits, ids, hashes = {}, set(), set()
    for split in ("train", "val", "test"):
        with (root / "data" / f"{split}.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"Empty split: {split}")
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=["id_code", "diagnosis", "sha256"], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if row["id_code"] in ids or row["sha256"] in hashes:
                raise ValueError("Split leakage/duplicate detected; bundle was not written")
            ids.add(row["id_code"])
            hashes.add(row["sha256"])
            writer.writerow({k: row[k] for k in writer.fieldnames})
        payload[f"data/{split}.csv"] = text.getvalue().encode("utf-8")
        splits[split] = len(rows)
    audit = json.loads((root / "data" / "audit.json").read_text(encoding="utf-8"))
    audit.pop("source", None)
    payload["data/audit.json"] = json.dumps(audit, ensure_ascii=False, indent=2).encode("utf-8")
    meta = {"format": "blindless-colab-bundle-v1", "splits": splits,
            "includes": "Code, image IDs, labels, original pixel hashes. No images, weights, local paths or credentials.",
            "files": {name: hashlib.sha256(content).hexdigest() for name, content in payload.items()}}
    payload["bundle.json"] = json.dumps(meta, indent=2).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in payload.items():
            archive.writestr(name, content)
    print(f"Bundle: {output.resolve()} ({output.stat().st_size:,} bytes); splits={splits}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build(args.root, args.output or args.root / "colab" / "blindless_colab_bundle.zip")
