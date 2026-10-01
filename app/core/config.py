import os
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def cpu_count() -> int:
    """CPUs this process may use: the container's cgroup v2 quota if set (os.cpu_count() ignores it)."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


CPUS = cpu_count()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # comma list, all loaded side by side: piper (real-time, CPU) | supertonic | kokoro (requirements-engines.txt)
    # | qwen3 (zero-shot cloning, needs NVIDIA GPU; runs alone). Env ENGINES, or the older ENGINE.
    engines: str = Field("piper", validation_alias=AliasChoices("engines", "engine"))
    model_name: str = "hindi-tts-v2"  # public name exposed by /v1/models
    default_voice: str = "hi_IN-rohan-medium"  # what voice="default" resolves to
    log_level: str = "INFO"

    # --- piper ---
    models_dir: Path = Path("models/piper")  # every *.onnx + *.onnx.json here is a voice
    # Extra directories (comma list) scanned for voices in addition to models_dir, e.g. MODELS_EXTRA=voices picks up the
    # fine-tuned voices/hi_IN-custom-medium.onnx that training/export_latest.sh writes. Empty = nothing added; a stem
    # already loaded from an earlier directory wins, so extra directories can never shadow a bundled voice.
    models_extra: str = ""
    # Each worker thread runs one chunk at a time on a session shared per voice, using threads_per_worker
    # ONNX intra-op threads. 4 threads cut single-chunk latency ~2.3x vs 1 at equal throughput (docs/benchmarks.md).
    threads_per_worker: int = min(4, CPUS)
    workers: int = max(1, CPUS // min(4, CPUS))
    use_cuda: bool = False
    indian_english: bool = True  # English words -> Indian-English phonemes the Hindi voice knows (indian_english.py)
    # VITS sampling: noise_scale = pitch/energy variation, noise_w = rhythm variation. None = the voice's own config.
    noise_scale: float | None = None
    noise_w: float | None = None

    # --- supertonic / kokoro (optional, see requirements-engines.txt; weights via scripts/download_voices.py) ---
    supertonic_dir: Path = Path("models/supertonic3")
    supertonic_steps: int = 8  # flow-matching denoising steps: fewer = faster and rougher (package default 8)
    kokoro_dir: Path = Path("models/kokoro")

    # --- qwen3 (optional, see requirements-qwen.txt) ---
    model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"  # HF repo id or local dir
    model_subfolder: str = ""
    device: str = "auto"  # auto | cuda:0 | cpu
    language: str = "Auto"
    voices_dir: Path = Path("voices")  # qwen3 reference clips

    # --- speaker registry / cloning (docs/voice-system.md) ---
    speakers_dir: Path = Path("speakers")  # persistent registry: one dir per speaker (record + references + embeddings)
    keep_raw_reference: bool = True  # default retention for new speakers: keep uploaded reference audio on disk
    reference_delete_after_days: int | None = None  # default: purge raw references this many days after upload
    ref_max_bytes: int = 10_000_000
    ref_min_seconds: float = 3.0
    ref_max_seconds: float = 30.0
    ref_min_sample_rate: int = 16000
    upload_max_bytes: int = 11_000_000  # request body cap for upload routes (multipart overhead above ref_max_bytes)
    max_references_per_speaker: int = 10
    retention_sweep_minutes: float = 60  # how often expired raw references are purged while running; 0 = only at startup/on read
    speaker_encoder: str = "mfcc"  # mfcc (baseline, NOT neural) | resemblyzer | speechbrain (optional installs)

    # --- expressive engine / DSP / routing (docs/voice-system.md) ---
    expressive_engine: str = ""  # package.module:ClassName of an ExpressiveEngine subclass; used with ENGINES=...,expressive. Off by default.
    # Opt-in DSP post-processing for condition.pitch (semitones) / condition.energy (gain). Signal processing, not emotion:
    # reported as "pitch:dsp" / "energy:dsp". Off = pitch/energy are rejected/ignored like any unsupported control.
    dsp_prosody: bool = False
    # engine -> latency tier for routing_policy, from docs/benchmarks.md (fast = real-time at several streams on CPU,
    # balanced = ~2 real-time streams per 4 cores, slow = GPU / not real-time). Unlisted engines count as slow.
    engine_latency_tiers: str = "piper:fast,supertonic:balanced,kokoro:balanced,expressive:slow,qwen:slow"

    # --- streaming ---
    # output rate when a request doesn't set sample_rate. 24 kHz = OpenAI's pcm contract, which OpenAI-shaped
    # clients (e.g. the calling platform's custom adapter) assume. Resampling via soxr costs ~nothing.
    default_sample_rate: int = 24000
    first_chunk_chars: int = 60  # first chunk is split at a clause/word boundary for low time-to-first-audio
    max_chunk_chars: int = 120  # shorter chunks = shorter non-preemptible runs: p95 TTFA 293->215 ms at 20 calls
    max_input_chars: int = 4000
    lead_silence_ms: int = 30  # model pads ~250 ms of silence before each chunk; trim it to this
    # Engines end every chunk with an end-of-utterance silence (Kokoro ~400 ms, Supertonic ~500-700 ms). A chunk cut
    # mid-sentence (at a comma, or the short first chunk) would leave that whole pause inside the sentence: cap it.
    pause_ms: int = 150  # chunk that ends mid-sentence (Kokoro's own pause at a comma is ~120 ms)
    sentence_pause_ms: int = 350  # chunk that ends a sentence (। . ? !)
    cache_size: int = 256  # cached chunks (repeated calling phrases)

    # --- protection ---
    api_keys: str = ""  # comma separated; empty disables auth (dev only)
    rate_limit_per_minute: int = 600
    max_streams: int = 64  # concurrent synthesis streams per process; beyond this -> HTTP 503 / WS error "overloaded"

    # --- demo page (/demo: browser mic -> STT -> LLM -> our TTS). Vendor keys stay on the server. ---
    demo_enabled: bool = True
    demo_allow_open: bool = False  # with API_KEYS empty, /demo serves loopback clients only unless this is true
    demo_max_tokens: int = 300  # cap on every vendor LLM reply
    demo_turn_timeout_s: float = 60  # hard deadline for one turn (STT + LLM + TTS)
    demo_turns_per_minute: int = 20  # per WebSocket connection
    platform_openai_api_key: str = ""
    platform_gemini_api_key: str = ""
    platform_sarvam_api_key: str = ""
    platform_elevenlabs_api_key: str = ""
    demo_stt: str = ""  # sarvam | openai | elevenlabs; empty = first with a key
    demo_llm: str = ""  # openai | gemini | sarvam; empty = first with a key
    # model ids checked against each vendor's docs on 2026-09-29 (see app/services/providers.py)
    demo_sarvam_stt_model: str = "saaras:v3"
    demo_openai_stt_model: str = "gpt-transcribe"  # measured ~1.0 s vs ~1.9 s median for gpt-4o-mini-transcribe
    demo_elevenlabs_stt_model: str = "scribe_v2"
    demo_openai_llm_model: str = "gpt-6-luna"
    demo_gemini_llm_model: str = "gemini-3.5-flash-lite"
    demo_sarvam_llm_model: str = "sarvam-105b-conversations"
    demo_system_prompt: str = (
        "You are a polite Indian phone agent. Reply in natural spoken Hindi (Devanagari script), keeping common "
        "English words such as loan, payment and OTP in English, the way Indians speak. Answer in 1-2 short "
        "sentences. Never use markdown, lists or emojis."
    )

    def resolve_model_dir(self) -> str:
        if Path(self.model_path).is_dir():
            return str(Path(self.model_path) / self.model_subfolder)
        from huggingface_hub import snapshot_download

        root = snapshot_download(self.model_path, allow_patterns=[f"{self.model_subfolder}/**"] if self.model_subfolder else None)
        return str(Path(root) / self.model_subfolder) if self.model_subfolder else root

    @property
    def keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


settings = Settings()
