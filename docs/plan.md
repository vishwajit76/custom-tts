# Claude Code: Build a High-Performance Hindi TTS Engine for AI Calling

## Role

Act as a senior AI/ML engineer specializing in speech synthesis, real-time audio processing, model fine-tuning and low-latency inference.

Your objective is to research, develop, customize, optimize, test and deploy a fully self-hosted Hindi and Hinglish text-to-speech engine specifically designed for real-time AI calling.

**Do not just create a plan. Execute the entire implementation autonomously, validate every stage and deliver a working solution.**

## 1. Core requirements

Develop a TTS engine with the following capabilities:

* Natural, human-like Indian Hindi speech.
* Hindi and Hinglish support, including English words within Hindi sentences.
* Extremely low first-audio latency.
* Real-time streaming audio generation.
* Custom voice training and fine-tuning.
* Multiple voice profiles.
* CPU and GPU inference.
* Low RAM and VRAM consumption.
* High concurrency and efficient resource utilization.
* Immediate interruption and cancellation of speech.
* Fully offline inference without external API dependencies.
* Commercial deployment readiness.

Target performance:

| Metric              | Target                                   |
| ------------------- | ---------------------------------------- |
| First audio latency | Under 200 ms                             |
| Real-time factor    | Under 0.1                                |
| Initial concurrency | 10 simultaneous calls                    |
| Scalability target  | 200 simultaneous calls                   |
| Language            | Hindi and Hinglish                       |
| Audio               | Streaming PCM, configurable sample rates |

Treat these as performance targets, not guaranteed outcomes.

## 2. Research and select the model

Research current open-source Hindi TTS models, including:

* Piper TTS
* AI4Bharat Indic-TTS
* IndicF5
* Other suitable lightweight, high-performance speech synthesis architectures.

Evaluate pronunciation, naturalness, latency, streaming capabilities, model size, memory consumption, licensing and fine-tuning support.

Benchmark promising candidates and select an architecture based on measured results.

Verify the commercial-use rights of the selected model, pretrained weights and training datasets.

## 3. Build the training pipeline

Implement an automated dataset preparation and fine-tuning pipeline.

Support:

* Hindi speech datasets.
* Custom voice recordings from consenting speakers.
* Audio segmentation and noise removal.
* Transcript validation and alignment.
* Hindi text normalization.
* Training and validation splits.
* Speaker identification.
* Model checkpoints.
* Training resumption.
* Evaluation and model export.

Use pretrained checkpoints wherever appropriate rather than training from scratch.

If training data or hardware is unavailable, implement and validate the pipeline without claiming to have completed model training.

## 4. Optimize for real-time inference

Optimize the selected model for the lowest achievable latency without unacceptable speech-quality degradation.

Investigate and implement, where supported:

* ONNX Runtime.
* FP16 and INT8 quantization.
* Model warm-up.
* Persistent inference workers.
* Streaming and incremental synthesis.
* Sentence-level synthesis.
* Efficient audio buffering.
* CPU thread optimization.
* GPU acceleration.
* Dynamic batching where it improves latency.
* Concurrent inference scheduling.

Measure actual performance after every optimization.

## 5. Hindi and Hinglish speech processing

Build a robust text normalization and pronunciation pipeline.

Support Devanagari Hindi, Romanized Hindi, code-switching, Indian names, numbers, currencies, dates, abbreviations and common conversational expressions.

Ensure that the model produces natural speech with appropriate pronunciation, pacing and pauses.

Implement configurable speaking speed and voice selection. Add emotional or expressive speech controls only where the model genuinely supports them.

## 6. Real-time audio streaming

Develop a streaming inference API suitable for AI calling.

Requirements:

* WebSocket streaming.
* Optional HTTP streaming.
* PCM audio output.
* Configurable sample rates, including 8 kHz and 16 kHz.
* Audio chunking and buffering.
* Request IDs and session management.
* Immediate generation cancellation.
* Backpressure handling.
* Connection lifecycle management.
* Graceful error recovery.

Generate speech at the model's native sample rate and perform high-quality resampling when necessary.

Prioritize time to first audio and uninterrupted playback.

## 7. Concurrency and scalability

Design an inference architecture that supports multiple simultaneous conversations.

Implement bounded worker pools, request queues, session isolation, configurable concurrency limits, resource monitoring and overload protection.

Benchmark 1, 5 and 10 concurrent sessions. Evaluate higher concurrency using available infrastructure and report hardware requirements for scaling to 200 simultaneous sessions.

Avoid unnecessary architectural complexity.

## 8. Testing and evaluation

Create automated tests and benchmarks covering:

* Hindi pronunciation.
* Hinglish code-switching.
* Audio quality.
* First-audio latency.
* Real-time factor.
* Streaming stability.
* Concurrent inference.
* Memory consumption.
* Cancellation and interruption.
* Long-running reliability.
* CPU and GPU performance.

Measure p50, p95 and p99 latency. Create a reproducible benchmark suite and save the results.

## 9. Deployment

Package the engine as an independently deployable, self-hosted service.

Provide Docker configurations, environment templates, model caching, health checks, structured logging, resource limits and deployment documentation.

Ensure that the inference service can operate entirely offline once its dependencies and model weights are installed.

## 10. Execution instructions

1. Research and verify available models and their licenses.
2. Establish a working Hindi TTS baseline.
3. Benchmark latency and audio quality.
4. Select and optimize the inference architecture.
5. Implement the dataset preparation and fine-tuning pipeline.
6. Fine-tune a custom voice when authorized data and hardware are available.
7. Implement real-time audio streaming.
8. Build the inference API and concurrency management.
9. Execute functional, audio-quality and performance tests.
10. Package and document the production-ready system.

Maintain a progress checklist throughout implementation.

Automatically resolve errors, run tests and validate completed work before proceeding.

Ask for approval before paid infrastructure provisioning, destructive operations or substantial resource expenditure.

**Final deliverable:** A working, fully self-hosted Hindi/Hinglish TTS engine optimized for real-time AI calling, with reproducible benchmarks, custom voice training capabilities, streaming inference, deployment configurations and complete technical documentation.
