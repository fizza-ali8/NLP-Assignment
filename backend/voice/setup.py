"""Download the Piper voice used by voice/tts.py. Safe to re-run."""

import os
import urllib.request

from voice.tts import ONNX_PATH, _DIR

FILES = {
    "en_US-lessac-medium.onnx": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
        "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    ),
    "en_US-lessac-medium.onnx.json": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
        "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    ),
}


def main():
    os.makedirs(_DIR, exist_ok=True)
    for name, url in FILES.items():
        dest = os.path.join(_DIR, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"already present: {dest}")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"saved {dest} ({os.path.getsize(dest)} bytes)")
    print(f"voice ready: {ONNX_PATH}")


if __name__ == "__main__":
    main()
