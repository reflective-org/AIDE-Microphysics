# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Reflective
"""Publish emulator weights to the Hugging Face Hub repo named in each scheme's models.json.

    python scripts/upload_hf.py <dir>      # <dir> holds tomas40/<model>/model.pt (+ README.md model card)

Checks every file's sha256 against models.json before uploading, uploads norm_stats.npz next to the
weights, and prints the commit. Needs `hf auth login` with write access to the org.
"""
import hashlib
import json
import os
import sys

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = sys.argv[1]
man = json.load(open(os.path.join(ROOT, "aide_micro/schemes/tomas40/models.json")))
for name, e in man["models"].items():
    p = os.path.join(src, e["file"])
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert h == e["sha256"], f"{name}: {h} != manifest {e['sha256']}"
    print(f"ok {name} {e['file']} {os.path.getsize(p) / 1e6:.1f} MB")
ns = os.path.join(ROOT, "aide_micro/schemes/tomas40/norm_stats.npz")
os.makedirs(os.path.join(src, "tomas40"), exist_ok=True)
with open(ns, "rb") as f, open(os.path.join(src, man["norm_stats"]), "wb") as g:
    g.write(f.read())
api = HfApi()
api.create_repo(man["hf_repo"], repo_type="model", exist_ok=True)
info = api.upload_folder(folder_path=src, repo_id=man["hf_repo"], repo_type="model",
                         allow_patterns=["tomas40/**", "README.md"], commit_message="tomas40 emulators: bintf_d384, bintf_d256, cs_b")
print(info)
