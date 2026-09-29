# Voice system

Sections below are owned by different upgrade phases. "Routing" is still a placeholder.

## Conditioning API & capabilities

Code: `app/services/conditioning.py` (schema, capabilities, validation), `app/api/speech.py::prepare_ex`.

### Request shape

All conditioning is **optional** and lives in one nested object, `condition`, so every existing request body
(`{input, voice, speed, response_format, sample_rate}` and WS `speak` messages) behaves exactly as before.

```json
POST /v1/audio/speech        (same on /v1/audio/speech/stream)
{"input": "जी, मैं आपकी मदद कर सकती हूँ।", "voice": "default",
 "condition": {"emotion": "empathetic", "emotion_strength": 0.6, "style": "warm", "role": "customer_support",
               "speed": 1.1, "fallback": "ignore"}}

WS: {"type":"speak","id":"r1","text":"...","condition":{"emotion":"calm","fallback":"ignore"}}
```

| Field | Values |
|---|---|
| `speaker_id`, `speaker_embedding` (floats), `reference_audio` (base64), `reference_text`, `style_reference` (base64) | identity / reference inputs |
| `emotion` | neutral, happy, sad, excited, calm, concerned, empathetic, apologetic, serious |
| `style` | conversational, professional, warm, narration, storytelling, energetic, soft |
| `role` | customer_support, assistant, narrator, teacher, announcer |
| `emotion_strength`, `style_strength`, `prosody_strength` | 0..1 (uncalibrated; a strength needs its emotion/style) |
| `speed` | 0.5..2.0, default 1.0 (same bounds as the top-level `speed`) |
| `pitch` (semitones, -12..12), `energy` (0..2) | |
| `fallback` | `reject` (default) or `ignore` |

`reference_audio`/`reference_text` may also be given at top level (legacy, qwen3). Top-level `speed` is used unless
`condition.speed` is set explicitly. An invalid enum or range is a normal 422 schema error.

### Behaviour: never claim what was not applied

`validate_condition(cond, caps)` compares the request with the serving engine's `EngineCapabilities`:

- `fallback: "reject"` (default): any unsupported control -> **HTTP 422** `{"detail": {"message": ..., "unsupported": [names]}}`
  (WS: `error` with `code: "unsupported_control"` and `unsupported`).
- `fallback: "ignore"`: unsupported controls are dropped, synthesis proceeds, and they are listed as ignored.
- Response reports (only when `condition` was sent, so old clients see unchanged responses):
  HTTP headers `X-TTS-Applied-Controls` and `X-TTS-Ignored-Controls` (comma-separated names);
  WS `start` message gains `applied_controls` and `ignored_controls` arrays.
- Applied names carry a suffix where honesty needs it: `emotion:steered` = prompt-steered, best effort, not validated;
  `speed:clamped_to_0.7` = the engine clamps to its own range.
- `speaker_id` resolves through the speaker registry (see below); it is rejected (422) or listed as ignored when the speaker has no binding/reference the serving engine can use.
- Requests are validated against the engine that owns the chosen voice (multi-engine servers route per voice).

### Engine capabilities (verified from code, not from marketing)

| Engine | cloning | native emotion/style | prompt-steered | role/pitch/energy | speed | streaming | native rate |
|---|---|---|---|---|---|---|---|
| piper | no | no | no | no | yes, VITS `length_scale` | sentence | per voice (22.05 kHz medium) |
| supertonic | no | no | no | no | yes, native, clamped to 0.7-2.0 | sentence | per model (see `/v1/voices`) |
| kokoro | no | no | no | no | yes, native | sentence | 24 kHz |
| qwen3 | yes (reference audio) | no | no | no | yes, librosa time-stretch (DSP) | sentence | 24 kHz |

"sentence" = the pipeline (`tts.stream`) synthesizes one sentence/clause per engine call and streams between calls;
no engine streams inside a model call. **Today no engine implements emotion, style, role, pitch or energy**, so
requesting them returns 422 (or is listed as ignored). The API and tests are in place for a future expressive engine.
Output at 8/16/22.05/24/44.1/48 kHz is soxr resampling for every engine, independent of the native rate.

### Discovery

- `GET /v1/voices`: unchanged fields plus a `capabilities` object per voice.
- `GET /v1/capabilities`: `{engines: {name: {..capabilities.., voices: [...]}}, controls: {emotion, style, role, fallback}, output_sample_rates}`.

## Speaker registry

Code: `app/services/speaker_registry.py`, `app/api/speakers.py`, `app/services/speaker_encoder.py`.

A speaker is a persistent, owner-scoped record under `SPEAKERS_DIR` (default `speakers/`, gitignored), one directory each:
`speaker.json` (atomic write, fsync + rename, process lock + flock), `refs/<ref>.wav|.npy`, `embedding.npy`.

Record: `id` (`[A-Za-z0-9][A-Za-z0-9_-]{0,63}`, no dots or slashes, so no traversal), `display_name`, `languages`,
`gender`/`presentation` (descriptive metadata only, never identity), `engine_bindings` (`{engine: voice_id}`),
`references[]` (relative path, sha256, duration, sample rate, level, clipping, speech fraction, transcript, heuristic
quality), `embedding_path` + backend, `training` provenance, `consent`, `retention`, `previews`. Stored paths are relative
and re-resolved inside the speaker directory on every access (symlinks and `..` refused).

**Owner** = `key_` + first 16 hex of SHA-256(API key) (`anonymous` when auth is disabled). Only the owner can read
(the API never returns the owner field), modify, upload to, synthesize with or delete a speaker; other keys get 404 so ids
cannot be enumerated. Note: with `API_KEYS` empty every caller is `anonymous` and shares speakers (dev only).

### REST (all under the normal API-key auth)

| Route | Purpose |
|---|---|
| `POST /v1/speakers` | create `{id, display_name, languages?, gender?, presentation?, retention?, consent?, training?}` (409 if exists) |
| `GET /v1/speakers`, `GET /v1/speakers/{id}` | list / read (own speakers only) |
| `PATCH /v1/speakers/{id}` | metadata, `engine_bindings` (validated against loaded engines/voices), retention |
| `PUT /v1/speakers/{id}/consent` | set consent (below) |
| `POST /v1/speakers/{id}/references` | multipart `audio` (+ `transcript`); returns metrics, `similarity_to_existing`, `warnings` |
| `DELETE /v1/speakers/{id}/references/{ref_id}` | secure-delete one reference |
| `DELETE /v1/speakers/{id}` | secure-delete everything (also removes the engine's own clip copy, e.g. qwen3 `voices/`) |

**Reference validation** (422 on failure): <= `REF_MAX_BYTES` (10 MB), content-type WAV/FLAC/octet-stream and decodable
by libsndfile, `REF_MIN_SECONDS`..`REF_MAX_SECONDS` (3..30 s), >= `REF_MIN_SAMPLE_RATE` (16 kHz), clipping <= 1% of samples,
level >= -45 dBFS, >= 30% speech frames, non-finite audio, duplicate sha256, <= `MAX_REFERENCES_PER_SPEAKER` (10).

### Consent and use

`consent = {status: granted|revoked|pending, consent_record_id, granted_by, date, permitted_uses[]}`; uses are
`tts`, `cloning`, `training`, `evaluation`. `granted` requires `consent_record_id`, `granted_by` and at least one use.
New speakers are `pending`. Synthesis with `condition.speaker_id` requires `granted` + `tts` (403 otherwise), and `cloning`
too when it clones from a stored reference. **Revoking** consent immediately purges all references, embeddings and the
engine-side clip copies (the record and its audit fields remain). The consent record id is a pointer to your own
consent evidence; this service does not verify it.

### Resolution of `condition.speaker_id`

1. owner check (404) -> consent check (403);
2. an `engine_bindings` entry for a loaded engine whose voice exists -> that voice is used (`voice` is overridden);
3. else, on a cloning engine (qwen3), the speaker's best retained reference (+ its transcript) is the clone prompt;
4. else `fallback:"reject"` -> 422 `{"unsupported": ["speaker_id"]}`; `fallback:"ignore"` -> synthesized with the request's
   `voice`, and `speaker_id` is listed in `X-TTS-Ignored-Controls`.

`speaker_id` together with `reference_audio` is a 400. The applied list contains `speaker_id` only when it took effect.

### Retention and deletion

`retention = {keep_raw_reference (default `KEEP_RAW_REFERENCE=true`), delete_after_days (default
`REFERENCE_DELETE_AFTER_DAYS`)}`. With `keep_raw_reference=false` the upload is validated, hashed and embedded, and the
audio is never written (`raw_retained:false`); such a speaker can only be used through an engine binding. With
`delete_after_days`, raw reference files are removed after that age (checked on startup and on reads); sha256, metrics and
embeddings stay as provenance. Deletion overwrites each file with random bytes, fsyncs and unlinks it, then removes the
directory. Caveat: on SSDs/journaling or copy-on-write filesystems, snapshots and backups overwrite is best-effort only;
use full-disk encryption and keep backups out of the registry dir if deletion guarantees matter.

### Speaker encoder (informational)

`app/services/speaker_encoder.py`: lazy load, resample to 16 kHz, min-duration/silence/clipping checks, L2-normalized
embeddings, cache by audio sha256 (memory, optional `.npy` on disk), cosine similarity, `consistency()` (each
embedding vs the mean of the others). Backends (`SPEAKER_ENCODER`): `mfcc` (default; deterministic MFCC statistics,
**not a neural speaker verifier**: scores are uncalibrated and unrelated voices can still score high), and optional
`resemblyzer` / `speechbrain` (lazy import; `BackendUnavailable` if not installed; licences in `docs/licenses.md`).
Embeddings are only used for reference QA (warning when a new reference is dissimilar, neural backends only) and eval.
**No engine declares `speaker_embedding`, so an embedding never conditions synthesis**; `condition.speaker_embedding`
is still rejected/ignored.


## Cloning

### Zero-shot (reference clip -> voice, qwen3 only)

- Per request: `condition.reference_audio` (+ `reference_text`), or legacy top-level `reference_audio`. Nothing is stored.
- Stored: `POST /v1/voices` (legacy multipart `voice_id`, `audio`, optional `transcript`) writes the clip into
  `VOICES_DIR` and is used with `voice=<voice_id>`. Contracts of `POST/DELETE /v1/voices` are unchanged. It now also
  mirrors a registry speaker bound to that clip with consent `pending` (send the optional `consent_record_id` +
  `granted_by` form fields to mark it granted for `tts`/`cloning`), and `DELETE` secure-deletes the clip and mirror.
  The legacy `voice=` path itself is not consent-gated (backward compatibility); the `speaker_id` path is.
- Registry: create a speaker, set consent, upload references, then use `condition.speaker_id`. With several references
  qwen3 gets the **best** one (highest heuristic quality: 5-12 s, healthy level, low clipping, speech-dense). Qwen3-TTS
  builds a voice prompt from a single clip (`create_voice_clone_prompt(ref_audio=...)`; lists in its API are batches, not
  merged references), so clips are not concatenated. A transcript makes qwen use the full in-context prompt; without one
  it is x-vector only.

### Few-shot / fine-tuned (training a voice)

Zero-shot quality is limited by one clip. For a permanent, better voice: collect 30+ minutes with consent, build a
manifest with `speaker_authorization` and `permitted_uses: ["training"]` (`docs/training.md`, data-rights gate), fine-tune a
Piper voice, drop the `.onnx` into `MODELS_DIR`, then `PATCH /v1/speakers/{id}` with `engine_bindings: {"piper": "<voice_id>"}`
and record provenance in `training`. From then on `speaker_id` resolves to the fast engine and the raw clips can be
deleted (`keep_raw_reference=false`).

### Safeguards

- Synthesis and cloning need recorded, unrevoked consent for the specific use; consent is per speaker and revocable
  with immediate purge.
- Voices are private to the creating API key; no listing or reading across keys.
- No identity from metadata: gender/presentation are labels only.
- Raw audio retention is configurable; deletion is an overwrite-and-unlink.
- Embeddings are not returned by the API and not used for identity or authentication decisions.
- This service does not verify that the uploader is the speaker or that consent evidence is genuine; operators must
  collect and audit consent out of band. Do not use it to imitate people without permission.


## Routing

_Placeholder: filled by the routing phase._
