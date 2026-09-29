"""Minimal WebSocket client: speak one or more sentences, save a WAV, print time-to-first-audio.

  python scripts/ws_client.py "नमस्ते, मैं आपकी कैसे मदद कर सकती हूँ?" "Aapka payment kal tak ho jayega." \
      --url ws://localhost:8000/v1/audio/ws --key $KEY --sample-rate 8000 --out reply.wav [--barge-in-ms 800]
"""
import argparse
import asyncio
import json
import time
import wave

import websockets


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="+", help="one speak request per argument, played in order")
    p.add_argument("--url", default="ws://localhost:8000/v1/audio/ws")
    p.add_argument("--key", default="")
    p.add_argument("--voice", default="default")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--out", default="reply.wav")
    p.add_argument("--barge-in-ms", type=int, default=0, help="send cancel this long after the first audio (demo)")
    a = p.parse_args()

    url = f"{a.url}?api_key={a.key}" if a.key else a.url
    pcm, first, t0 = bytearray(), None, time.perf_counter()
    async with websockets.connect(url, max_size=None) as ws:
        for i, text in enumerate(a.text):  # sent back-to-back: the server queues and plays them in order
            await ws.send(json.dumps({"type": "speak", "id": f"r{i}", "text": text, "voice": a.voice, "sample_rate": a.sample_rate}))
        pending = len(a.text)
        cancel_at = None
        while pending:
            if cancel_at and time.perf_counter() >= cancel_at:
                await ws.send(json.dumps({"type": "cancel"}))  # barge-in: stop everything; also flush your player
                cancel_at = None
            msg = await ws.recv()
            if isinstance(msg, bytes):
                if first is None:
                    first = time.perf_counter() - t0
                    cancel_at = time.perf_counter() + a.barge_in_ms / 1000 if a.barge_in_ms else None
                pcm += msg
                continue
            ev = json.loads(msg)
            print(ev)
            if ev["type"] in ("end", "cancelled", "error"):
                pending -= 1
    with wave.open(a.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(a.sample_rate)
        w.writeframes(bytes(pcm))
    print(f"{a.out}: {len(pcm) / 2 / a.sample_rate:.2f}s audio, first audio after {first * 1000:.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())
