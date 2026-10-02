# Voice system

Sections: conditioning API, control taxonomy (native neural / prompt / reference conditioning / DSP post-processing), speaker registry and the audio-to-model boundary, cloning, text normalization for a Hindi voice, routing, telephony output and barge-in, real-time behaviour, production notes, evaluation.

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

### Control taxonomy: how a control can really be realised

Every control is realised by exactly one mechanism, and the API reports which one (`control_kinds` per engine in
`GET /v1/capabilities`; suffixes in `X-TTS-Applied-Controls`). Nothing is ever relabelled to look like something stronger.

| Kind | What it is | Label | Who has it today |
|---|---|---|---|
| native | the model was trained with this control and takes it as input (emotion, style, role, pitch, energy tokens/embeddings) | `emotion`, `style`, `role` | **nobody**. `ExpressiveEngine` (below) is the boundary for a future engine |
| reference conditioning | identity or delivery comes from audio given at request time, either as an in-context prompt or through an encoder (cloning, speaker embedding, style reference) | `reference_audio`, `speaker_embedding`, `style_reference` | qwen3 cloning only (reference clip used directly). No engine consumes an external `speaker_embedding` or `style_reference` |
| prompt conditioning | a text instruction asks the model for an emotion or style; best effort, not validated | `emotion:steered`, `style:steered` | nobody (`prompt_emotion` is False everywhere) |
| DSP | signal processing on the output waveform | `pitch:dsp`, `energy:dsp`, `prosody_strength:dsp` | any engine, **only when `DSP_PROSODY=true`** |
| engine speed | `length_scale` or time-stretch inside the engine | `speed`, `speed:clamped_to_X` | all |

Emotion, style and role are never satisfiable by DSP: on today's engines they stay rejected (422) or listed as ignored.

### Four mechanisms, kept separate (P7)

The words "emotion", "style" and "voice" are used loosely elsewhere; in this repo they mean exactly one of these four things, and
the API reports which one (`control_kinds`, `:steered`/`:dsp` suffixes). A feature is never described by a stronger mechanism than it has.

| Mechanism | What is actually happening | Decided when | Today |
|---|---|---|---|
| **Native neural conditioning** | the model was *trained* with a control (emotion/style/role label, speaker id, pitch/energy token) and takes it as an input tensor | training time + request time | **None for emotion/style/role.** Piper's only learned input is a speaker id in multi-speaker models (none ship); Supertonic/Kokoro select a fixed preset voice |
| **Prompt conditioning** | a natural-language instruction or tag in the model's text prompt asks for a delivery; the model may or may not follow it | request time, unvalidated | **None.** `prompt_emotion` is false on every engine; no instruct/VoiceDesign path is wired |
| **Reference conditioning** | audio supplied at request time (or a stored reference) conditions identity/delivery, either in-context (the model reads the clip) or through an encoder embedding | request time | **Qwen3 only**, and only the reference clip (+ optional transcript) used directly; no embedding is passed to any engine |
| **DSP post-processing** | signal processing applied to the finished waveform (time-stretch, pitch shift, gain, resampling) | after synthesis | opt-in `DSP_PROSODY` pitch/energy (`:dsp`); Qwen3 `speed` is a time-stretch; output resampling for every engine. It is **never** emotion or style |

Consequences that are easy to get wrong:

- **A Piper voice's identity is baked in by fine-tuning, not conditioned.** Fine-tuning on consented recordings produces a *new model file*; at request time there is no
  speaker input to change. You cannot hand Piper a reference clip or an embedding, and this service does not pretend to.
- **Speed is not expressiveness.** `length_scale` (Piper/Kokoro/Supertonic) changes the duration of every phoneme; it does not make speech calmer or more urgent.
- **DSP cannot add an emotion.** Pitch shifting and gain change the sound of the same prosody; they are labelled `:dsp` and rejected for emotion/style/role.
- **`noise_scale` / `noise_w` are sampling temperature, not a control.** They widen or narrow VITS's random pitch/rhythm variation. They are deployment settings
  (`NOISE_SCALE`, `NOISE_W`; unset = the voice's own `config.json` `inference` block), not request fields and not an emotion knob. The infer grid on the custom voice
  (`bench/infer_grid.py`, milestone 350000, 18 configs x 15 sentences x 2 repeats; table in [training-progress.md](training-progress.md), audio in `docs/samples/infer_grid/`)
  points to `noise_scale` 0.5 with `noise_w` 0.8-1.0 and `length_scale` 1.0. That recommendation is **pending a listening check**: single-config differences are inside the
  ~0.02 CER noise band, the UTMOS used is English-trained, and a lower `noise_scale` may sound flatter, which no automatic metric here can show. **It has not been applied:
  the server defaults are unchanged.**

### Native emotion/style/role: integration boundary only (Phase 5 is blocked on a model)

No selected engine supports native emotion or style, and the bake-off in [model-selection.md](model-selection.md) has not run,
so no expressive synthesis is implemented. What exists is the contract, in `app/services/expressive_engine.py`:

- `ExpressiveEngine` base class: an implementation declares `capabilities` (`native_emotion`, `native_style`, `role`, `pitch`,
  `energy`) and implements `synth` (neutral) and `synth_native(text, voice, speed, controls, ...)`.
- Flow: `condition` -> `validate_condition` (gates on declared capabilities) -> `split_controls` -> `tts.stream(controls=...)` ->
  `engine.synth_native`. An engine only ever receives controls it declared. The phrase cache key includes the controls.
- Disabled unless configured: `ENGINES=piper,expressive` plus `EXPRESSIVE_ENGINE=package.module:ClassName`. Empty -> startup error,
  nothing imported. `synth_native` defaults to `NotImplementedError` so a half-implemented engine fails loudly.
- Verified only with a fake engine (`tests/test_expressive_dsp.py`): conditioning reaches the engine, an engine without the
  capability never sees it, `fallback:"ignore"` reports the control as ignored. This proves the plumbing, not any audio quality.

### DSP prosody (opt-in): pitch and energy

`DSP_PROSODY=true` (needs `librosa`: `pip install -r requirements-dsp.txt`, or build the image with `--build-arg REQUIREMENTS=requirements-dsp.txt`; startup fails clearly without it) lets `condition.pitch` (semitones, -12..12), `condition.energy` (gain, 0..2, 1.0 = unchanged) and
`prosody_strength` (0..1, scales both toward neutral) work on every engine, applied per chunk after the phrase cache and before
resampling (`app/services/dsp.py`). Off by default, because it changes the sound and can degrade it.

- Pitch: `librosa.effects.pitch_shift` (phase vocoder + soxr HQ). Duration is preserved. Formants shift with pitch, so the voice
  sounds smaller/larger: acceptable to about +/-3 semitones, audibly artificial beyond about +/-6. Cost measured here: ~8 ms per
  second of audio (4 vCPU), plus a one-off multi-second warm-up done at startup.
- Energy: linear gain with a tanh soft limiter above 0.8 full scale, so a boost never hard-clips.
- Reported as `pitch:dsp`, `energy:dsp` (never as emotion, never as "expressive"). Listening previews label it `[dsp]`.
- It does not make a voice "happy" or "sad". Do not sell it as emotion.

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
too when it clones from a stored reference or uses an engine binding to a cloning engine (that voice was produced from the
speaker's reference audio). **Revoking** consent immediately purges all references, embeddings and the
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
`delete_after_days`, raw reference files are removed after that age (checked at startup, on reads, and by a background sweep every `RETENTION_SWEEP_MINUTES`, default 60; 0 = startup/reads only); sha256, metrics and
embeddings stay as provenance. When the last raw reference expires, the cloning engine's own copy (e.g. the qwen3 voices_dir
clip) is deleted too and the speaker is marked `clone_capable:false` (bindings to cloning engines are then ignored). Deletion overwrites each file with random bytes, fsyncs and unlinks it, removes `speaker.json` **last** (a failure midway leaves a record you can
retry the delete against, never unreachable orphan audio), then removes the directory. Revocation is saved *before* the purge: if the purge is interrupted the speaker is
already revoked and blocked, and the next startup/sweep finishes removing the files. A failed upload rolls back the files it wrote. Caveat: on SSDs/journaling or copy-on-write filesystems, snapshots and backups overwrite is best-effort only;
use full-disk encryption and keep backups out of the registry dir if deletion guarantees matter.

### Speaker encoder (informational)

`app/services/speaker_encoder.py`: lazy load, resample to 16 kHz, min-duration/silence/clipping checks, L2-normalized
embeddings, cache by audio sha256 (bounded in-memory LRU of 512, optional `.npy` on disk), cosine similarity, `consistency()` (each
embedding vs the mean of the others). Backends (`SPEAKER_ENCODER`): `mfcc` (default; deterministic MFCC statistics,
**not a neural speaker verifier**: scores are uncalibrated and unrelated voices can still score high), and optional
`resemblyzer` / `speechbrain` (lazy import; `BackendUnavailable` if not installed; licences in `docs/licenses.md`).
Embeddings are only used for reference QA (warning when a new reference is dissimilar, neural backends only) and eval.
**No engine declares `speaker_embedding`, so an embedding never conditions synthesis**; `condition.speaker_embedding`
is still rejected/ignored.


### Boundary: reference audio -> encoder -> embedding -> compatible model

```text
reference audio (consented, 3-30 s)
   |-- analyze_reference: decode, length/level/clipping/speech-fraction checks, sha256          (registry, always)
   |-- speaker encoder -> embedding (mfcc | resemblyzer | speechbrain)                        (QA + evaluation only)
   |-- used directly as a clone prompt:  Qwen3 (create_voice_clone_prompt, ONE clip)           (the only consumer of audio)
   '-- used to train a model:            Piper fine-tune -> new .onnx -> engine_bindings       (identity = a file, not an input)
```

| Model | Consumes reference audio | Consumes an embedding | How a speaker's identity reaches it |
|---|---|---|---|
| Piper (VITS) | **no** | **no** (no speaker-embedding input exists in the exported graph; `speaker_id` is a lookup into trained speakers) | fine-tuning: a new `.onnx`, then `PATCH engine_bindings {"piper": "<voice_id>"}` |
| Supertonic, Kokoro | no | no | fixed preset voices only |
| Qwen3-TTS | **yes**, one clip (+ transcript) per prompt | no (its internal x-vector is computed by the model from the clip, not supplied by us) | `reference_audio`, or the speaker's best retained reference via `condition.speaker_id` |
| future expressive/cloning engine | declared through `EngineCapabilities.speaker_embedding` / `cloning` | only if it declares `speaker_embedding` | not implemented; `condition.speaker_embedding` is rejected (422) or listed as ignored |

- **The encoder's embedding never reaches a synthesis call.** It is used to (a) warn when a new reference does not resemble the existing ones, and (b) compare voices in `bench/`.
  Do not describe it as "the voice print the model uses".
- **Multiple references are QA/selection, not fusion.** With several references, Qwen3 receives the single highest-quality clip (heuristic: 5-12 s, healthy level, low clipping, speech-dense);
  clips are never concatenated or averaged, because Qwen3's API builds one prompt from one clip. Extra references help only by giving `best_reference` a better candidate and by
  the consistency check. A model that genuinely benefits from several references (none is wired) would be the exception.
- **Three different numbers, never interchangeable.** The response labels which one you got (`encoder.similarity_kind`):

| Label | What it is | Can it decide identity? |
|---|---|---|
| `mfcc_statistics_cosine` | cosine between MFCC mean/std/delta vectors (default `mfcc`). Characterises timbre and channel roughly; same-voice 0.99, different-voice 0.96 on our synthetic voices | **No.** Unrelated voices score high; the service emits no warning threshold for it |
| `neural_embedding_cosine` | cosine between GE2E (`resemblyzer`) or ECAPA (`speechbrain`) embeddings. Same-voice 0.94 vs different-voice 0.64 on 7 synthetic TTS voices (benchmarks.md 9e) | **No, not as shipped.** Meaningful for QA, but the 0.75 warning threshold is an uncalibrated default (not fitted on same/different-speaker trials for either backend, whose scales differ), and it only produces a warning, never a rejection |
| speaker *verification* | an accept/reject decision from a model calibrated on labelled same/different-speaker trials at a chosen false-accept rate | **Not implemented.** Nothing here is verification; `calibrated` is `false` for every backend |

- Mixed backends: embeddings from two encoders live in different spaces, so a speaker keeps the backend of its first embedding; a later reference embedded by another backend is
  stored **without** an embedding (and the response warns), instead of being averaged into the centroid.
- The default `mfcc` backend needs `librosa`, which is **not** in `requirements.txt` (only `requirements-dsp.txt` / `requirements-qwen.txt`). Without it the upload still succeeds and the
  response carries `embedding skipped: ... needs librosa`.


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

**Using the personal custom voice.** `training/export_latest.sh` writes `voices/hi_IN-custom-medium.onnx` + `.onnx.json` (gitignored, personal-use, intermediate checkpoint; see
[custom-voice-runbook.md](custom-voice-runbook.md)). It is selectable next to the bundled voices with `MODELS_EXTRA=voices` (default empty: no other deployment changes; a voice whose
file stem is already loaded from `MODELS_DIR` is never shadowed): `GET /v1/voices` then lists `hi_IN-custom-medium` (engine `piper`, 22.05 kHz, no gender claimed), and
`{"voice": "hi_IN-custom-medium"}` or `DEFAULT_VOICE=hi_IN-custom-medium` selects it. `voices/` is also `VOICES_DIR`, where the qwen3 engine keeps its `*.wav` reference clips; the two
uses do not collide (Piper reads only `*.onnx`). Its inference defaults are the voice's own (`noise_scale` 0.667, `noise_w` 0.8, `length_scale` 1.0 from its `.onnx.json`); see the infer-grid
note under "Four mechanisms" for the pending recommendation. The `speaker_id` path for it needs a registry speaker with `engine_bindings {"piper": "hi_IN-custom-medium"}` and recorded consent.

### Safeguards

- Synthesis and cloning need recorded, unrevoked consent for the specific use; consent is per speaker and revocable
  with immediate purge.
- Voices are private to the creating API key; no listing or reading across keys.
- No identity from metadata: gender/presentation are labels only.
- Raw audio retention is configurable; deletion is an overwrite-and-unlink.
- Embeddings are not returned by the API and not used for identity or authentication decisions.
- This service does not verify that the uploader is the speaker or that consent evidence is genuine; operators must
  collect and audit consent out of band. Do not use it to imitate people without permission.


## Persona grammar (V7)

Hindi first-person verbs, participles and a few predicate adjectives agree with the speaker's gender (`करता/करती हूँ`,
`गया/गई`, `करूँगा/करूँगी`). `app/services/persona_grammar.py` rewrites a script to the voice's gender:
`apply(text, Persona(gender, persona="assistant", age_group="adult"))`; `rewrite()` returns the same text plus the list
of changes (rule, offsets into the original) for logs and tests; `llm_hint(persona)` is the Hindi system-prompt sentence
the demo adds so the LLM writes the right forms in the first place. `hinglish.apply_persona_gender` is a thin wrapper.

**Opt-in and speaker-authored only.** Nothing calls it automatically. On `POST /v1/audio/speech`, `/speech/stream` and
the WebSocket `speak` message, an optional `"persona": {"gender": "female"|"male"|"neutral", "persona": "...",
"age_group": "young_adult"|"adult"|"mature"}` applies it to the text; absent = text unchanged. Send it only for text the
assistant itself wrote: a user's "मैं जाता हूँ" is the user's grammar and must not be rewritten.

**How it decides** (rule-based, clause-bounded, one pass). Quoted spans (`"..."`, curly, guillemets, backticks) are opaque.
Text is split into clauses at punctuation and at conjunctions/subordinators (और, लेकिन, कि, जब, ...). A first-person
clause is anchored by `हूँ/हूं`, a 1sg future (`-ूँगा/-ऊँगा`, any nasal spelling), or `था/थी` with an explicit `मैं`.
The anchor and the agreeing chain to its left flip: `-ता/-ती` by morphology (with a noun/name guard), `रहा`, `चुका`,
`वाला`, `गया`, a passive participle before `गया` (`बनाया गया हूँ`), and predicate adjectives from a closed list
(`अच्छा`, `नया`, `थका हुआ`, ...) only when they touch the auxiliary. With an explicit `मैं`, a clause-final perfective
(`मैं घर गया`) and `आपका/आपकी` + a role noun (`सहायक`, `असिस्टेंट`, ...) flip too. The writer's `ँ/ं` spelling is kept.

**Guarantees.** `neutral` and non-Devanagari text return unchanged; idempotent (applying twice = once); the other
direction is symmetric (f->m); quoted speech, third person (`वह/वो/वे`, named subjects without `मैं`), second person
(`आप/तुम`), English, and anything outside a first-person clause stay byte-identical. When unsure it leaves text
unchanged (precision over recall). About 40 us for a 130-character sentence on the M4 (budget 500 us).

**Not rewritten, on purpose.**
- `हम`: plural or royal, may include other genders, so there is no safe single flip.
- Reported speech: the clause after `कहा कि`/`बोला कि`/`he said that`, and a comma-separated clause next to a
  non-first-person reporting verb (`राहुल ने कहा, मैं आऊँगा`). A `कि` clause is rewritten only under a first-person
  matrix (`मुझे लगता है कि मैं ...`).
- `मैंने ...` (ergative: the verb agrees with the object) and `मुझे`-constructions (`मुझे पता है`, `मुझे जाना है`).
- Nouns and names (`मैं लड़का हूँ`, `मैं गीता हूँ`, `रास्ता`, `पता`); invariant predicates (`तैयार`, `खुश`, `ठीक`).
- Subject ellipsis across conjunctions: `मैं आया और सो गया` rewrites only the first clause.

**Known limits.** A second, differently-subjected predicate inside one comma-less clause with an explicit `मैं`
(`मैं पहुँची ट्रेन जा चुकी थी`) can be rewritten wrongly; names ending in `ता/ती` that are not in the guard list can be
mistaken for verbs directly before `हूँ`; the adjective/participle lists are closed, so rare adjectives
(`मैं शर्मिंदा हूँ`) are left masculine (precision over recall).

**`age_group`.** Hindi grammar does not inflect for age, so it changes no rewrite. It is metadata for voice selection
(`hi-IN-young-female`, ...) and for LLM persona prompts only.

## Text normalization for a Hindi voice

Code: `app/services/text_normalizer.py` (rules), `hinglish.py` + `lexicon_hi.tsv` (Romanized Hindi, names, brands), `indian_english.py` (English runs -> phonemes).
Tests: `tests/test_pronunciation_corpus.py` + `tests/data/pronunciation_corpus.tsv` (384 rows), `tests/test_normalizer_p3.py`. **Every corpus row is machine-drafted and marked
`needs native review`; no native speaker has signed any off.** Normalization costs about 0.4 ms for a sentence and 15 ms for 4000 characters (single thread).

### What a Latin-script token becomes (decision record)

| Class | Example | Spoken as | Why |
|---|---|---|---|
| Acronym | `UPI`, `OTP`, `CRM`, `VoIP`, `API`, `SIP`, `ID`, `QR` | letters in Devanagari (`यू पी आई`) | English G2P reads `CRM`/`SIP` as words; letter names are what callers say |
| Romanized Hindi | `kya aap free hain` | Devanagari, **only** inside a sentence that has at least one word that is never English (`kya`, `hain`, `aapka`...) | `the`, `is`, `he`, `in`, `to`, `use` are also Hindi spellings; an all-English sentence must stay English (`The AI is ready`) |
| Indian name / place / curated **brand** | `Rahul`, `Bengaluru`, `Paytm`, `Zomato`, `Jio`, `Airtel`, `Amazon` | Devanagari respelling from `lexicon_hi.tsv` (kind `name`), always, case-insensitive | espeak-ng `hi` reads Devanagari with Hindi rules; its English G2P + our Indian-English mapping mangles many of these (table below) |
| Any other English | `loan`, `payment`, `WhatsApp`, `video` | stays Latin; each English run becomes a Piper raw-phoneme block (en-us IPA mapped to Indian-English: retroflex t/d, tapped r, v/w merged, monophthongs) | correct for ordinary words and avoids an endless respelling table |

**Brand evidence** (custom voice `hi_IN-custom-medium` = the personal fine-tune at an intermediate checkpoint, and `hi_IN-rohan-medium`; 15 brands x 2 carrier sentences, the brand slot
alone changes; Whisper-small Hindi decoding, CER of the whole sentence against the carrier with any accepted spelling of the brand; 30 sentences per voice per route):

| Voice | English route (Latin + phoneme mapping) | Devanagari route |
|---|---|---|
| custom | 0.331 | **0.303** |
| rohan | 0.297 | **0.253** |

That proxy is noisy (a few hundredths per brand; the custom voice's own carrier error is large), so it was not used alone. The phonemes espeak-ng produces are the stronger evidence
(Latin route -> Devanagari route): Zomato `zəmˈɔɾoː` -> `zoːmˈɛːʈoː`; Airtel `ˈɛɾɾəl` -> `eːjəɾʈˈeːl`; Tata `ʈˈɔɾə` -> `ʈˈaːʈaː`; PhonePe `fˈoːn pˈiː` -> `fˈoːnpeː`; Jio `ɖzˈiːʲoː` -> `ɟˈɪjoː`;
Paytm `pˈeːɾəm` -> `peːʈiːeːm`; Amazon `ˈɛmʌzɔn` -> `ʌmeːzən`; Instagram `ˈɪnsʈʌɡɾɛm` -> `ĩsʈaːɡɾaːm`; Xiaomi `zˌaɪəˈoːmi` -> `ʃaːˈoːmi`. The Devanagari spellings are the intended pronunciations; the English
route's output is not. **Exception: WhatsApp stays on the English route**: 0.156 vs 0.21-0.33 (custom, five Devanagari spellings tried) and 0.211 vs 0.196-0.348 (rohan): no spelling
beat it reliably. Flipkart is spelled with a nukta (`फ़्लिपकार्ट`; 0.239 vs 0.325 custom, 0.185 vs 0.251 rohan). **This is a proxy, not a listening test**: a native listener may
prefer a different spelling; the lexicon is one line per brand, so changing it is cheap. Do not add a brand without phonemizing its respelling with `espeak("hi", ...)` first.

### Conversational rules added in this audit

| Input | Spoken | Notes |
|---|---|---|
| `98765 43210`, `+91 98765 43210`, `1800 123 4567`, `022-2345 6789` | digit by digit with a comma (short pause) between groups | the writer's grouping is kept; an ungrouped 10-digit mobile is 5+5; longer runs by 4 |
| `9:30 pm`, `5 PM`, `12 AM` | `रात साढ़े नौ बजे`, `शाम पाँच बजे`, `रात बारह बजे` | am/pm used to be dropped, making 9 am and 9 pm identical; not doubled after `सुबह`/`शाम`... |
| `15 Aug 2026`, `Aug 15`, `1st Jan` | `पंद्रह अगस्त दो हज़ार छब्बीस`... | a bare month word (`May I help`) is never a date |
| `₹500-₹1000`, `₹5-10 lakh` | `पाँच सौ से एक हज़ार रुपये`, `पाँच से दस लाख रुपये` | |
| `rahul.sharma92@gmail.com`, `www.flipkart.com/offers` | `राहुल डॉट शर्मा नौ दो एट जीमेल डॉट कॉम`, `डब्ल्यू डब्ल्यू डब्ल्यू डॉट फ़्लिपकार्ट डॉट कॉम स्लैश offers` | scheme dropped; bare domains only for common TLDs (so `Ok.In the morning` is text) |
| `Q3`, `MP3`, `B2B`, `2BHK` | `क्यू थ्री`, `एम पी थ्री`, `बी टू बी`, `टू बी एच के` | never a token that mixes scripts |
| `AI` / `ai` | `ए आई` | uppercase is the acronym; lowercase inside a Romanized-Hindi sentence is a lazily typed AI; `the ai bot` (English) is untouched |
| `use karein` / `use bulao` | English `use` / `उसे` | before a light verb it is the English verb |
| `Hi, main Amit` | `Hi` stays | capital `Hi` is the greeting; lowercase `hi` is the particle `ही` |
| `ticket #4521`, `order no. 12345`, `press *` | `नंबर ...`, `नंबर ...`, `स्टार` | |

Known limits: Romanized Hindi words outside `lexicon_hi.tsv` (~900 entries) that do not match the shape heuristic stay Latin and are read with English rules (add them to the lexicon);
order numbers of 6-9 digits are read as a quantity, not digit by digit (ambiguous without context); `WhatsApp` and other unlisted brands depend on the English route;
possessives (`Google's`) produce a mixed token.


## Routing

Code: `app/services/routing.py`, `app/api/speech.py::prepare_ex`. Optional. No `routing_policy` = exactly the old behaviour.

`routing_policy` (`fast` | `balanced` | `expressive` | `clone`) may be sent top-level on `/v1/audio/speech*` and WS `speak`, or
inside `condition`. It selects an **engine** (and its default voice) when, and only when, the caller did not choose one:
`voice` is `"default"` and there is no `reference_audio` and no resolved `speaker_id`. **An explicit voice, speaker or reference
always wins**; the response then says `X-TTS-Routing: explicit` (WS `start`: `"routing":"explicit"`).

| Policy | Picks |
|---|---|
| `fast` | the lowest latency tier among engines that can honour every requested control |
| `balanced` | prefers tier `balanced`, then `fast`, then `slow`, among capable engines |
| `expressive` | only an engine with native/steered emotion, style or role (DSP never qualifies) |
| `clone` | only an engine with `cloning` |

Capability is decided by the same `validate_condition` used for validation, so routing cannot disagree with it. Tiers are
configuration, not guesses: `ENGINE_LATENCY_TIERS` (default `piper:fast,supertonic:balanced,kokoro:balanced,expressive:slow,qwen:slow`,
key = lowercase class name without `Engine`), from the measurements in [benchmarks.md](benchmarks.md) (Piper meets 200 ms TTFA at
4 streams; Kokoro/Supertonic ~0.25-0.55 RTF; Qwen needs a GPU). Unlisted engines count as `slow`. Re-measure on your hardware.

**No silent downgrade.** If no loaded engine can honour the policy and the requested controls (e.g. `expressive` with only
Piper, `clone` in a `piper,kokoro` server, or emotion with `fast`), the request fails with **422**
`{"unsupported": [...]}`. With `condition.fallback:"ignore"` it is routed as `balanced`, and the response lists
`routing_policy` (plus whatever the engine then drops, e.g. `emotion`) in `X-TTS-Ignored-Controls`.
Response: `X-TTS-Routing: policy;engine=<name>;voice=<id>` (WS `start`: `routing`, `routed_engine`, `routed_voice`).
`GET /v1/capabilities` lists the policies, each engine's `latency_tier` and `control_kinds`.

Limits: qwen3 (cloning) cannot be loaded together with other engines, so `clone` needs a cloning-only server; `MultiEngine`
still routes by voice id underneath. There is no load-aware routing (queue depth is not consulted) and no automatic
mid-request failover.

## Real-time behaviour (what "streaming" is here)

Code: `app/services/tts.py::stream`, `scheduler.py`, `app/api/ws.py`.

**Streaming is pipeline-level chunking, not model streaming.** `normalize` -> `split_for_stream` (sentence ends `। . ? !`; a sentence over `MAX_CHUNK_CHARS` = 120 is cut at commas/spaces;
the first chunk is cut to <= `FIRST_CHUNK_CHARS` = 60 at a clause/word boundary) -> **one complete engine call per chunk** (VITS runs the whole chunk, then returns the whole waveform)
-> trim lead silence -> soxr resample -> PCM. Audio for chunk *k* is sent as soon as chunk *k* is finished; nothing is emitted from inside a model call.

- **Time to first audio (TTFA)** = WebSocket/HTTP handling + normalize (<1 ms) + espeak phonemization + one VITS run over <= 60 characters + resample, *plus queueing behind other
  streams' chunks*. It is not a per-frame model latency, and no engine here has one to report.
- **The server produces faster than real time**, so a stream's audio can be in client/gateway buffers long before it is played (WebSocket backpressure only throttles it when the socket
  buffer is full). Consequences: barge-in needs the client to flush its buffer; `ttfa_ms` says nothing about when a given word is *heard*.
- **Scheduling.** Each chunk is queued with a deadline (when the audio already sent runs out) and workers take the earliest deadline first, so new calls' first chunks pass chunks with slack.
  A chunk that is running cannot be preempted; `MAX_CHUNK_CHARS` bounds that. `WORKERS` x `THREADS_PER_WORKER` default to 1 x 4 on 4 vCPU (2 x 4 on the 10-core M4 that the README numbers came from).
- **Cancellation.** `{"type":"cancel"}` cancels the current request's task (the awaiting `scheduler.run` future is cancelled, so queued chunks are skipped and the running one's audio is
  dropped), drains queued requests with `cancelled` acks, and frees the `MAX_STREAMS` slot. A `cancel{id}` is only remembered for ids that are still queued (a stale one used to poison a later
  request reusing the id). An HTTP stream whose client disconnects, and a WebSocket dropped without a close frame, release their slot (see benchmarks.md section 10 for the churn measurement).
- **Bounded state.** Phrase cache `CACHE_SIZE` (LRU), TTFA ring (2000), WS queue (32), encoder cache (512), rate-limit deques (<= `RATE_LIMIT_PER_MINUTE` per key). Numbers: benchmarks.md section 10.

## Production notes (P6)

| Area | Behaviour | Gap / what to do |
|---|---|---|
| Auth | `Authorization: Bearer` (WS also `?api_key=`); keys compared in constant time; empty `API_KEYS` disables auth (logged at startup) | failed attempts are not throttled: keep the service behind a gateway/WAF, use long random keys |
| `/health`, `/metrics` | unauthenticated (probes, Prometheus) | expose them only on the internal listener; `/metrics` reveals load and RSS |
| Rate limit | per key, per process, in memory, `RATE_LIMIT_PER_MINUTE` (600); counts HTTP requests and WS *connections* (a refused WS handshake reaches clients as HTTP 403, not 429) | WS `speak` messages are not counted (concurrency is bounded by `MAX_STREAMS` and the 32-deep queue); with auth disabled all clients share one bucket; per-replica, not global |
| Request size | upload routes (`/v1/voices`, `/v1/speakers/*/references`): `UPLOAD_MAX_BYTES` (11 MB) -> 413. JSON speech routes: `MAX_INPUT_CHARS * 6 + 64 KiB` (+ a base64 clip only when the engine clones) -> 413 | WS messages are bounded by uvicorn's `--ws-max-size` (16 MB default); text beyond `MAX_INPUT_CHARS` is a 400 / `bad_request` |
| Uploads | header-first decode (a small FLAC cannot expand to gigabytes), duration/level/clipping checks, content-type allowlist, sha256 dedupe | validated speaker ids `[A-Za-z0-9][A-Za-z0-9_-]{0,63}`, voice ids `fullmatch`ed, every stored path re-resolved inside the speaker dir (no `..`, no symlinks) |
| Registry writes | atomic (temp file + fsync + rename), process lock + `flock`; blocking work runs in a worker thread, not on the event loop | single node, single filesystem: no multi-replica sharing of `SPEAKERS_DIR` |
| Cache invalidation | voice upload/delete/consent revoke/retention expiry clear the phrase cache for that voice; a chunk synthesized while that happens is not cached afterwards | the phrase cache is keyed by voice + text + speed + controls, never by speaker |
| Shutdown | uvicorn `--timeout-graceful-shutdown 10`, compose `stop_grace_period 15s`; open WebSockets are closed, their tasks cancelled; the retention sweep task is cancelled | no drain mode (a SIGTERM'd replica still accepts streams until uvicorn stops): deregister it from the load balancer first |
| Retention | startup, on read, and every `RETENTION_SWEEP_MINUTES` (60) | backups/snapshots are outside the overwrite guarantee |


## Telephony output and barge-in

Code: `app/services/tts.py::stream`, `app/services/audio_utils.py`, `app/api/ws.py`. Tests: `tests/test_telephony.py`, `tests/test_robustness.py`.

- `sample_rate` 8000 and 16000 (also 22050/24000/44100/48000) give **PCM signed 16-bit little-endian, mono**. WS: `start.encoding =
  "pcm_s16le"`; HTTP stream: `audio/L16` with `X-Sample-Rate`. Any native rate is converted by a `soxr` **HQ** streaming
  resampler (one instance per request, so chunk borders have no seams); float -> int16 uses rounding and clipping.
- Measured in tests: a 5 kHz tone (above the 8 kHz stream's Nyquist) is more than 130 dB below the 1 kHz tone after 24 kHz -> 8 kHz
  conversion (no aliasing into the band), in-band tones are preserved, chunked streaming output matches one-shot resampling
  to < 1e-3, and WS frames (`frame_ms`) are exact multiples of whole samples.
- Sending 8 kHz means the band above 4 kHz is gone; there is no G.711 mu-law/A-law encoding in the server (the telephony gateway must
  encode). `bench/eval.py` reports the 8 kHz round-trip loss per voice.
- **Level, headroom, pauses (checked on the custom voice and in tests).** There is no level normalisation: output level is the model's (20 utterances: peak -2.4 dBFS, mean RMS
  -20.9 dBFS, **0 full-scale samples** at 22.05/16/8 kHz), so a gateway or call platform that needs a target loudness must apply its own AGC. Pauses are preserved: each model chunk has
  its ~110-140 ms of leading silence (custom voice; ~250 ms for rohan) cut to `LEAD_SILENCE_MS` = 30 ms, internal pauses are untouched, and silences of 120 ms+ in the 8 kHz stream have
  exactly the lengths of the same waveform at 22.05 kHz (`tests/test_telephony.py::test_internal_pause_length_is_preserved_through_trim_and_resample`). The 0.01 amplitude threshold of the lead trim cut at most
  12 ms of audible (> 0.002) onset in 20 test starts (fricative-initial sentences, two voices), so it does not clip word beginnings.
- **Intelligibility by output rate** (custom voice `hi_IN-custom-medium`, first 10 sentences of `bench/sentences.txt`, 2 independent syntheses each = 20 utterances; the three rates
  are produced from the **same** cached native waveform through the real `tts.stream` chain, so sampling noise cancels; Whisper-small int8 CPU, `language=hi`, beam 5; CER against the
  normalized text):

  | Output | CER mean | delta vs 22.05 kHz (mean +/- s.e., n=20) | utterances worse / same / better |
  |---|---|---|---|
  | 22.05 kHz (native) | 0.300 | - | - |
  | 16 kHz | 0.311 | +0.011 +/- 0.005 | 8 / 10 / 2 |
  | 8 kHz | 0.329 | +0.029 +/- 0.020 | 9 / 7 / 4 |
  | 8 kHz, then mu-law encode + decode (`audioop`) | 0.340 | +0.040 +/- 0.045 | 9 / 7 / 4 |

  Reading: absolute CER is high because the voice is an intermediate checkpoint and the reference contains numbers and English words that Whisper spells differently; only the
  deltas are meaningful. 16 kHz costs about one point of CER (small, consistent), 8 kHz about three points, which is **not distinguishable from zero** at this sample size; the
  mu-law step adds a further, noisy one point. **The mu-law row is G.711 mu-law quantisation only** (`audioop.lin2ulaw`/`ulaw2lin`, removed in Python 3.13): no network codec chain, no
  packet loss or jitter, no handset or noise, no human listener. It supports "8 kHz + mu-law quantisation did not make Whisper's transcripts much worse", not "telephone quality is validated".
- **Barge-in: the client must flush its own playback buffer.** `{"type":"cancel"}` stops synthesis and drops queued requests, and the
  server acks with `cancelled` in milliseconds, but the server sends faster than real time, so seconds of audio may already
  be in the client/gateway/network buffers. Stop and clear those the moment the caller speaks; the server cannot recall them.
  Chunks already running on a worker finish (cannot be interrupted) but their audio is discarded.
- Streaming here is **pipeline-level**: text is split into sentences/clauses and each is synthesized as a whole. No engine streams inside
  a model call, so time to first audio is one short chunk of synthesis, not per-frame model streaming.

## Evaluation

- `python -m bench.eval` runs identical sentences over voices x conditions and writes JSON (`bench/results/<label>.json`): duration,
  F0 median/spread, RMS level, pause count/length, clipping, silence, speaker similarity, 8 kHz round-trip loss, optional
  CER (`--cer`, only if a local Whisper is already cached; otherwise recorded as skipped). Unsupported conditions are recorded as
  `unsupported` rows, not synthesized.
- `python -m bench.previews` writes listening previews only for (voice or speaker) x control combinations the engine supports, named
  and labelled by mechanism (`[native]`, `[dsp]`, ...), plus `manifest.json` listing what was skipped.
- **Predicted MOS is not human MOS.** UTMOS (`bench/quality.py`) is an English-trained model's guess and only ranks similar
  systems; speaker similarity with the default `mfcc` backend is not a neural speaker verifier. None of this measures whether
  a voice sounds natural or expressive to native listeners: only a blind listening test does, and none has been run.

