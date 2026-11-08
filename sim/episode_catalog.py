"""Discover usable episode indexes, ignoring asset metadata and interrupted copies."""
import json
from pathlib import Path

def episode_catalog(root):
    found=[]
    for path in sorted(Path(root).rglob('manifest.json')):
        try:
            meta=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError):
            continue
        if isinstance(meta,dict) and meta.get('version')==1 and meta.get('state_flag') is not None and meta.get('scene') and meta.get('chunks'):
            found.append((path,meta))
    return found
