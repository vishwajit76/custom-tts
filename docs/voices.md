# Voices and the voice catalog

Code: `app/services/voice_catalog.py` (load + validate), `app/services/piper_engine.py` (binding to models), data:
`voices/catalog.json` (setting `VOICE_CATALOG`, default `voices/catalog.json`; the Dockerfiles copy it into the image).

The catalog is the single place that says which public voice is which model + speaker and what is known about it. **Gender
and age are declared, never inferred from a voice id** (a voice without `gender` reports `gender: null`; a voice outside the
catalog reports `age_group: "unspecified"`). A missing catalog file is an empty catalog with one warning; an invalid one stops
startup.

## Format

```json
{"version": 1, "voices": [ { "voice_id": "hi-IN-young-female", "engine": "piper", "model": "hi_IN-v7-medium", ... } ]}
```

Unknown fields are rejected. Fields:

| Field | Req. | Meaning |
|---|---|---|
| `voice_id` | yes | Public id used as `voice=`. Same charset as every voice id (`[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}`), unique with all aliases. |
| `engine` | yes | `piper`, `kokoro`, `supertonic`. Only Piper entries are bound to models by the catalog; the others carry metadata for voices their engine already lists. |
| `model` | piper | File stem of `<model>.onnx` + `<model>.onnx.json` in `MODELS_DIR` / `MODELS_EXTRA`. |
| `speaker` | no | Key in the model's `speaker_id_map`. Omit for single-speaker models. |
| `styles` | no | `{style name: speaker key}`. Names must be `condition.emotion` or `condition.style` values (`conditioning.Emotion` / `Style`), otherwise the entry is rejected (the style could never be requested). Several names may share a speaker. |
| `default_style` | no | Key of `styles` used when the request names none. If `speaker` is also set, both must name the same speaker. A voice with `styles` needs `speaker` or `default_style`. |
| `gender` | no | `"F"` or `"M"`. Omitted = `null`. |
| `age_group` | no | `young_adult`, `adult`, `mature`, `unspecified` (default). |
| `age_evidence` | when `age_group` is not `unspecified` | Free text: where the label comes from. |
| `style` | no | The voice's character, free text ("conversational, warm"). Not the request `style` control. |
| `family` | no | Model family ("piper-v7", "kokoro-82m"). |
| `language`, `accent`, `role` | no | Free text. `language` of legacy voices is the engine code (`hi`). |
| `license`, `data_source` | no | Licence and data provenance in one line each; copy from `docs/licenses.md`, do not paraphrase upwards. |
| `status` | yes | `production`, `experimental`, `planned`. |
| `aliases` | no | Extra ids that resolve to this voice (any engine). Not listed in `/v1/voices`. |
| `name` | no | Display `name` in `/v1/voices` (Piper default: `Piper <voice_id>`). |

### Behaviour

- `planned` entries are never registered: not in `/v1/voices`, not routable, aliases do not resolve.
- A Piper entry whose model file is absent is skipped; one warning lists the skipped ids (`planned` ones are not reported).
  An entry whose speaker keys are not in the loaded model's `speaker_id_map` (or that names a speaker for a single-speaker
  model) is skipped with a warning. If no voice at all loads, startup fails as before.
- Single-speaker model: always routable as its file stem (as before), plus any catalog entry for it.
- Multi-speaker model **with** catalog entries: only the catalog voices are exposed; its raw speakers are not. **Without**
  entries: one voice `<stem>:<speaker>` per speaker, as before.
- Piper selects the speaker with `sid` (the existing PiperVoice path); single- and multi-speaker models can run in the same server.

### Styles (learned, not DSP)

For a voice with `styles`, `condition.emotion` / `condition.style` switch to the recorded speaker of the same voice. The
voice's `capabilities` then report `native_emotion` / `native_style` only for the listed names (`emotion_values`,
`style_values`), `discrete_styles: true`, and applied controls are labelled plainly (`emotion`, not `emotion:dsp`). Because
each style is a separate recorded speaker:

- an emotion/style not in the list follows the usual rule: 422 with `fallback:"reject"`, listed in `X-TTS-Ignored-Controls` with `"ignore"` (the default speaker is used);
- `emotion_strength` / `style_strength` are unsupported (no strength knob exists);
- `emotion` and `style` together: emotion is applied, style is reported ignored (one speaker per request).

Voices without `styles` keep the engine-wide capabilities (speed only). DSP pitch/energy stays separate (`pitch:dsp`).

## API

`GET /v1/voices` keeps every existing field and adds `gender`, `age_group`, `style`, `family`, `styles` (list of names),
`status`, `language` (a value the engine already sent, e.g. `language: "hi"`, wins) and, in `capabilities`, `emotion_values`,
`style_values`, `discrete_styles`. Voices outside the catalog: `gender`/`style`/`family`/`status` null, `age_group`
`"unspecified"`, `styles` `[]`. `voice_catalog.gender_of(voice_id)` returns `"F"`, `"M"` or `None`; use it (or `gender` from
`engine.voices()`, which is the same data) for persona grammar.

## Current voices

Licence facts are from `docs/licenses.md` (evidence tags there; Hugging Face cards were not re-read). No `age_group` is
claimed for any current voice: there is no age evidence for them.

| voice_id | engine | gender (evidence) | status | licence / commercial |
|---|---|---|---|---|
| `hi_IN-rohan-medium` (default) | piper | M (model card dataset "Hindi Mono Male") | production | IITM EULA data, fine-tuned from lessac (research-only): **No (risk)** |
| `hi_IN-pratham-medium` | piper | M (name + median pitch ~98 Hz; the card does not say) | production | CC BY-NC-SA 4.0: **No** |
| `hi_IN-priyamvada-medium` | piper | F (name + median pitch ~214 Hz; the card does not say) | production | CC BY-NC-SA 4.0: **No** |
| `hi_IN-custom-medium` | piper | F (single female IndicTTS training speaker) | experimental | personal, non-commercial only; no listening test yet |
| `kokoro:hf_alpha`, `hf_beta` / `hm_omega`, `hm_psi` | kokoro | F / M (upstream name convention) | production | Apache-2.0 weights, **Conditional** (training-data provenance) |
| `supertonic:F1-F5` / `M1-M5` | supertonic | F / M (upstream preset name) | production | OpenRAIL-M, **Conditional** (disclose machine-generated audio) |
| `hi-IN-young-female`, `hi-IN-young-male` | piper (`hi_IN-v7-medium`) | F / M (Rasa speaker labels) | **planned** | Rasa data CC-BY-4.0 (docs/V7-ROADMAP.md); model licence to confirm |

The two V7 entries in `voices/catalog.json` are placeholders: speaker keys (`f_neutral`, `f_happy`, `f_conversation`, ...) and
style names are guesses until the trained `speaker_id_map` exists. They stay hidden while `planned`. `hi-IN-adult-female` /
`hi-IN-adult-male` can be added the same way once an adult speaker is trained.

## Evidencing age and gender

- **Gender**: dataset metadata or the recording speaker's documented label. State it in `data_source` (as the legacy entries do,
  including when it is only name + pitch).
- **Age**: `age_group` other than `unspecified` requires `age_evidence`: dataset metadata (quote the field and version), or a
  stated listener screening (who, how many, date, what they were asked). Never derive it from the id or from pitch alone.
  Rasa ships no speaker age, so V7 voices stay `unspecified` until screening exists.

## Add a voice

1. Put `<model>.onnx` and `<model>.onnx.json` in `MODELS_DIR` or a `MODELS_EXTRA` directory (for a multi-speaker model the
   json must have `num_speakers` > 1 and `speaker_id_map`).
2. Add an entry to `voices/catalog.json` (start from a neighbour). Set `status` to `experimental` until a listening test passes.
3. `pytest tests/test_voice_catalog.py` (the shipped catalog is validated there), start the server, check `/v1/voices`.

Registering the V7 model: add `hi_IN-v7-medium.onnx(.json)` to `voices/`, run with `MODELS_EXTRA=voices`, and set `status`
from `planned` to `experimental` on `hi-IN-young-female` / `hi-IN-young-male` after correcting `speaker`, `styles` and
`default_style` to the real `speaker_id_map` keys (style names must be `Emotion`/`Style` values, so a recorded style such as
`conversation` is listed as `conversational`). Add `"age_group"` + `"age_evidence"` only once the evidence exists.
