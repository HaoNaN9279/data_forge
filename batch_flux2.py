"""flux2-klein batch: prompt from txt -> generate image -> save as png."""
import json, copy, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from data_forge.tools.comfyui import ComfyUIClient

client = ComfyUIClient(
    "http://tech.ai.test1.blackjack-local.com:30050",
    timeout=300,
    proxies={"http": None},
)

wf_path = Path("src/data_forge/resources/ComfyUI_workflows/flux2-klein_9b_t2i.json")
base_wf = json.loads(wf_path.read_text("utf-8"))

fish_dir = Path("D:/EF_Fish/fish")
out_dir = Path("D:/EF_Fish/fish_rag")
out_dir.mkdir(parents=True, exist_ok=True)

txts = sorted(fish_dir.glob("*.txt"))

# Arg: "test" = 1 file, "batch" = all
if len(sys.argv) > 1 and sys.argv[1] == "test":
    txts = txts[:1]
    mode = "TEST"
elif len(sys.argv) > 1 and sys.argv[1] == "batch":
    mode = "BATCH"
else:
    txts = txts[:1]
    mode = "TEST (default)"

print(f"Mode: {mode} | Files: {len(txts)}\n")

success, fail = 0, 0

for i, txt_path in enumerate(txts):
    stem = txt_path.stem
    out_path = out_dir / f"{stem}.png"

    print(f"[{i+1}/{len(txts)}] {stem[:40]}... ", end="", flush=True)
    try:
        prompt_text = txt_path.read_text("utf-8").strip()
        wf = copy.deepcopy(base_wf)
        wf["88"]["inputs"]["string"] = prompt_text
        wf["9"]["inputs"]["filename_prefix"] = f"f2k_{i:03d}"

        pid = client.submit(wf)
        history = client.wait(pid, poll_interval=3.0)

        paths = client.download_outputs(pid, out_dir, overwrite=True)
        if paths:
            if out_path.exists():
                out_path.unlink()
            paths[0].rename(out_path)
            print("OK")
            success += 1
        else:
            print("WARN: no output image")
            fail += 1
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        fail += 1

print(f"\nDone: {success} ok, {fail} failed")
