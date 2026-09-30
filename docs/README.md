# Documentation index

Last reviewed 2026-09-30. Times in docs are UTC with IST (UTC+5:30) where it matters.

## Start here

**(a) Run the server**: [../README.md](../README.md) (quick start, API, env) then `.env.example`; multi-engine and voices in the
README "Engines and voices"; latency/sizing in [benchmarks.md](benchmarks.md). To serve the custom voice, see
[custom-voice-runbook.md](custom-voice-runbook.md) section 7.

**(b) Train or resume the custom voice**: [custom-voice-runbook.md](custom-voice-runbook.md) (relaunch, secrets, heartbeat,
troubleshooting) -> [../training/kaggle/README.md](../training/kaggle/README.md) (kernel internals) -> [training.md](training.md)
(generic pipeline, data rights, flags) only if you need to change how data is prepared or run on another machine.

**(c) Evaluate quality**: [custom-voice-runbook.md](custom-voice-runbook.md) section 6 (50-sentence eval protocol, stopping rule, `bench/milestone_eval.py`, `bench/infer_grid.py`) ->
[training-progress.md](training-progress.md) (all milestone scores) -> [benchmarks.md](benchmarks.md) sections 7 and 9 (cross-engine
UTMOS/CER, `bench/quality.py`, `bench/eval.py`). A human listening test has not been done yet.

## Index

| Doc | Purpose | Read it when |
|---|---|---|
| [custom-voice-runbook.md](custom-voice-runbook.md) | Operational runbook for the custom Hindi voice: assets, Kaggle training, secrets, evaluation, serving, troubleshooting, lessons | you operate, resume or debug the voice training |
| [training-progress.md](training-progress.md) | Milestone log: step, session, CER, losses, with caveats | you compare checkpoints or pick one to try |
| [voice-quality-research.md](voice-quality-research.md) | Naturalness levers, evaluation protocol, stopping rule, alternative models | before changing training or deciding when to stop |
| [training.md](training.md) | Generic training pipeline: dataset prep, manifests, rights gate, splits, flags, CPU fallback, smoke test | you prepare new data or change the pipeline |
| [PROGRESS.md](PROGRESS.md) | Phase status, what is done/partial/blocked, dated updates | you want the honest overall state of the project |
| [plan.md](plan.md) | Original build brief (historical) | you want the original requirements |
| [architecture-next.md](architecture-next.md) | Architecture audit and baseline before the upgrade phase | you change server internals |
| [voice-system.md](voice-system.md) | Conditioning API, speaker registry and consent, cloning, routing, telephony, evaluation | you use or extend voices, cloning or streaming output |
| [research.md](research.md) | Model research, dataset and vendor-audio findings, licence-driven choices | before any commercial decision or new data source |
| [model-selection.md](model-selection.md) | Expressive/multi-voice model shortlist with evidence quality | you pick a GPU-tier or expressive engine |
| [licenses.md](licenses.md) | Licence verdict per model, dataset and library | before shipping anything commercially |
| [benchmarks.md](benchmarks.md) | Latency, throughput, quality and sizing measurements and methods | you size hardware or compare engines |
| `samples/step_N/` | 3 test wavs per custom-voice milestone | you want to listen |
| `samples/infer_grid/<config>/` | 3 wavs for the best and the default inference-parameter configs (`bench/infer_grid.py`) | you want to hear noise_scale / noise_w / length_scale differences |
| `../bench/hi_eval_50.txt`, `../bench/results/milestones.jsonl` | fixed 50-sentence Hindi eval set and one scored row per milestone (`bench/milestone_eval.py`) | you evaluate or compare checkpoints |

Other project docs: [../README.md](../README.md), [../training/kaggle/README.md](../training/kaggle/README.md), `.env.example`.

## Maintenance rules

- Put numbers next to their date, hardware and method; say "not measured" rather than guessing.
- Operational facts for the custom voice live in the runbook only; other docs link to it.
- Add every new doc to the table above.
