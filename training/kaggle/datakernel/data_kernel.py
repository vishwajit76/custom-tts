"""Kaggle CPU kernel (no GPU quota): ingest AI4Bharat Rasa Hindi (CC-BY-4.0), run the V7 quality gates, upload the prepared Piper dataset to
the private HF repo under data/<VOICE_NAME>/. Runs on Kaggle because the datacenter link is ~100x faster than the owner's (Rasa Hindi is 17.6 GB).
The repo code it needs is vendored by push_data.sh as one base64 zip (CODE_B64). Secrets: HF token from the private dataset cttsh-secrets.
Speaker ids are fine-grained ({gender}_{style}); the advisor regroups styles later by rewriting metadata.csv (no audio reprocessing)."""
import base64, glob, io, json, os, pathlib, subprocess, sys, time, zipfile

CODE_B64 = ""  # stamped by push_data.sh
VOICE_NAME = os.environ.get("VOICE_NAME", "hi_v7")
REPO = "vishwajit76/custom-tts-hindi-train"
W = pathlib.Path(os.environ.get("W_ROOT", "/tmp/w")); W.mkdir(parents=True, exist_ok=True)
LOCAL = os.environ.get("LOCAL_PARQUET")  # smoke test: local parquet directory instead of the HF repo; no upload
T0 = time.time()


def sh(*cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, **kw)


src = W / "src"
zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_B64))).extractall(src)
env = dict(os.environ, PYTHONPATH=str(src), PYTHONUNBUFFERED="1",
           PRONUNCIATION_RULES=os.environ.get("PRONUNCIATION_RULES", "all"))  # train on rule-corrected text; the V7 catalog entries pin the same rules
if not LOCAL:
    tok = glob.glob("/kaggle/input/**/hf_token", recursive=True)
    assert tok, "hf_token not found under /kaggle/input"
    env["HF_TOKEN"] = open(tok[0]).read().strip()
    sh(sys.executable, "-m", "pip", "install", "-q", "soxr", "pyarrow", "huggingface_hub>=0.30")

raw, out = W / "rasa_hi", W / "stage" / "data" / VOICE_NAME  # staged so upload_large_folder keeps the repo layout data/<name>/
test_out = W / "stage_test" / "data" / f"{VOICE_NAME}_rasa_test"
for split in ("train", "test"):  # Rasa's own test split stays a separate held-out set (never trained on)
    sh(sys.executable, "-m", "training.ingest_hf", *([LOCAL] if LOCAL else []), "--preset", "rasa",
       "--out", raw if split == "train" else test_out, "--split", split,  cwd=src, env=env)

# Piper speaker = gender x style group (Rasa labels seen 2026-10-02 on Hindi/test-00000). Read-speech styles share one "neutral" speaker;
# CONV and ALEXA (voice-assistant requests) are the conversational register we serve; each emotion keeps its own speaker so it is learned, not DSP.
import csv, collections
GROUP = {"CONV": "conversational", "ALEXA": "conversational", "HAPPY": "happy", "SAD": "sad", "ANGER": "angry", "FEAR": "fearful",
         "SURPRISE": "surprised", "DISGUST": "disgusted"}  # everything else (WIKI, BOOK, NEWS, INDIC, PROPER NOUN, DIGI, UMANG, BB, ...) -> neutral
for meta in (raw / "metadata.csv", test_out / "metadata.csv"):
    rows, new_of = list(csv.DictReader(open(meta, encoding="utf-8"))), collections.defaultdict(set)
    for r in rows:
        new = f"{(r.get('gender') or 'x')[0].lower()}_{GROUP.get((r.get('style') or '').upper(), 'neutral')}"
        new_of[r["speaker_id"]].add(new); r["speaker_id"] = new
    with open(meta, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    rights = meta.parent / "rights.jsonl"  # the rights entry covers the same recordings under their new speaker ids
    ents = [json.loads(l) for l in rights.read_text("utf-8").splitlines() if l.strip()]
    for e in ents:
        e["speaker_ids"] = sorted({n for o in e.get("speaker_ids", []) for n in new_of.get(o, {o})})
    rights.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in ents), "utf-8")

# what is in the corpus: hours per gender x style (printed for the advisor; also uploaded)
hours = collections.Counter()
for r in csv.DictReader(open(raw / "metadata.csv", encoding="utf-8")):
    hours[(r.get("gender"), r.get("style"))] += float(r.get("duration") or 0) / 3600
stats = {f"{g}|{s}": round(h, 3) for (g, s), h in sorted(hours.items())}
print("HOURS", json.dumps(stats, ensure_ascii=False), flush=True)
(raw / "style_hours.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))

sh(sys.executable, "-m", "training.prepare_dataset", "--manifest", raw / "metadata.csv", "--output", out, "--workers", str(os.cpu_count() or 4),
   *os.environ.get("PREP_ARGS", "").split(), cwd=src, env=env)
for f in ("style_hours.json", "ingest_info.json", "rights.jsonl"):
    if (raw / f).exists():
        (out / f"source_{f}").write_bytes((raw / f).read_bytes())
print("prepared in", round((time.time() - T0) / 60, 1), "min:", (out / "report.json").read_text()[:2000], flush=True)

if not LOCAL:
    from huggingface_hub import HfApi
    api = HfApi(token=env["HF_TOKEN"])
    assert not any(f.startswith(f"data/{VOICE_NAME}/") for f in api.list_repo_files(REPO)), f"data/{VOICE_NAME} exists on HF: refusing to overwrite"
    for stage in (W / "stage", W / "stage_test"):
        api.upload_large_folder(repo_id=REPO, repo_type="model", folder_path=str(stage), num_workers=8)
    print("uploaded data/%s and data/%s_rasa_test in %.1f min" % (VOICE_NAME, VOICE_NAME, (time.time() - T0) / 60), flush=True)
