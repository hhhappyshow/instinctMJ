"""Fetch the pinned official G1 velocity export and matching assets, verifying Git blobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT = "1425b15f73bd4095f0df53709d7c389c3eb9e790"
REPO = "unitreerobotics/unitree_rl_mjlab"
DEFAULT_ROOT = Path("outputs/p4/checkpoint_candidates/unitree_rl_mjlab")


def fetch(root):
    with urllib.request.urlopen(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1", timeout=30) as response:
        tree = json.load(response)["tree"]
    extra = {
        "deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx": "policy.onnx",
        "deploy/robots/g1/config/policy/velocity/v0/params/deploy.yaml": "deploy.yaml",
        "LICENCE": "LICENCE",
    }
    selected = [item for item in tree if item["type"] == "blob" and (
        item["path"] in extra or item["path"].startswith("src/assets/robots/unitree_g1/")
        or item["path"] in ("src/tasks/velocity/mdp/observations.py", "src/tasks/velocity/velocity_env_cfg.py"))]

    def download(item):
        path = item["path"]
        destination = root / extra.get(path, "source/" + path)
        data = destination.read_bytes() if destination.exists() else b""

        def blob_sha(value):
            return hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()

        if blob_sha(data) != item["sha"]:
            with urllib.request.urlopen(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}", timeout=60) as response:
                data = response.read()
            if blob_sha(data) != item["sha"]:
                raise RuntimeError(f"Git blob mismatch: {path}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        return {"path": path, "local": str(destination.relative_to(root)), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(download, selected))
    root.mkdir(parents=True, exist_ok=True)
    (root / "asset_manifest.json").write_text(json.dumps({"repository": REPO, "commit": COMMIT, "files": records}, indent=2))
    print(f"Verified {len(records)} files ({sum(item['bytes'] for item in records)} bytes) in {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    fetch(parser.parse_args().root)
