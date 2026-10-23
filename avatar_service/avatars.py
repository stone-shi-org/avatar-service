"""Avatar catalog: portraits on the model volume.

`<models>/avatars/avatars.json` (optional) describes them:

    {"alexa": {"image": "alexa.png", "name": "Alexa", "voice": "af_heart", "emo": 4}}

Without it, every `*.png` / `*.jpg` in the folder is an avatar named after the file.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass

ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass
class Avatar:
    id: str
    image: str          # absolute path
    name: str
    voice: str = "af_heart"
    emo: int = 4        # Ditto emotion class 0..7 (4 = the model's default)

    def public(self) -> dict:
        d = asdict(self)
        d["image"] = os.path.basename(self.image)
        return d


def load_catalog(avatars_dir: str) -> dict[str, Avatar]:
    out: dict[str, Avatar] = {}
    if not os.path.isdir(avatars_dir):
        return out
    meta_path = os.path.join(avatars_dir, "avatars.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        for aid, m in meta.items():
            if not ID_RE.match(aid):
                continue
            img = os.path.join(avatars_dir, os.path.basename(m.get("image", f"{aid}.png")))
            if os.path.isfile(img):
                out[aid] = Avatar(aid, img, m.get("name", aid.title()), m.get("voice", "af_heart"), int(m.get("emo", 4)))
    else:
        for fn in sorted(os.listdir(avatars_dir)):
            base, ext = os.path.splitext(fn)
            if ext.lower() in (".png", ".jpg", ".jpeg") and ID_RE.match(base.lower()):
                out[base.lower()] = Avatar(base.lower(), os.path.join(avatars_dir, fn), base.title())
    return out
