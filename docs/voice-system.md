# Voice system

Sections below are owned by different upgrade phases. "Speaker registry", "Cloning" and "Routing" are placeholders
to be filled by their phases.

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
- `speaker_id` is rejected everywhere for now (voices are selected with `voice`; the registry is a later phase).
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

_Placeholder: filled by the speaker registry phase._

## Cloning

_Placeholder: filled by the cloning phase._

## Routing

_Placeholder: filled by the routing phase._
