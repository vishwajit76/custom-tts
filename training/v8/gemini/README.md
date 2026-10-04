# Gemini offline tooling (google-genai)

Gemini is used ONLY as an offline annotator, text generator and judge.

- No LLM in the TTS hot path (docs/v8.md).
- Gemini-generated AUDIO is never used as training data (Gemini API terms restrict using outputs to develop competing models; also v8.md provenance rules). The judge only listens to our own engines' wavs.
- No GPU use. Key `PLATFORM_GEMINI_API_KEY` is read from `.env` in-process; never printed or persisted.

Models: `gemini-3.5-flash` (bulk text), `gemini-3.1-pro-preview` (audio judge). Responses cached by hash in `.cache/` (reruns free); exponential backoff on 429/5xx.

- `style_annotate.py [--full]` -> `datasets/manifest/style_labels.jsonl` (Phase 13 styles, intensity, pauses, emphasis; keyed by audio_path)
- `eval_corpus.py` -> `bench/corpus/hi_eval_v3.tsv` (v2 rows unchanged on top)
- `audio_judge.py` -> `benchmarks/v8_model_bakeoff/results/gemini_judge.json` + REPORT.md table (automated proxy, not human MOS)
- `python -m pytest training/v8/gemini/test_gemini.py` (offline)

Labels and judge scores are LLM-derived and noisy; spot-check before training on them.
