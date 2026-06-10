"""Live end-to-end voice test for the Vox assistant — your voice → spoken reply.

Run this on the Raspberry Pi (or any arm64 box where moonshine-voice loads
natively). It CANNOT run on an Intel Mac: moonshine-voice ships arm64-only
binaries.

What it measures, per scripted prompt:
  * STT accuracy — the production moonshine_voice.Transcriber is fed your live
    mic audio exactly as raspberry-pi/main.py does. Because you read the fixed
    script in prompts.txt, WER/CER are EXACT ground truth, not pseudo-reference.
  * STT finalize latency — stop talking → final transcript line; streaming RTF.
  * Question detection — your raspberry-pi/main.py heuristic on real lines.
  * END-TO-END latency — exercises the real LLM+TTS. The Pi forwards the prompt
    on the /pi websocket and the server returns synthesized speech on the
    /client websocket (see oracle-cloud/main.py Relay). The harness opens BOTH
    sockets, so it sends as the Pi and receives the spoken audio as the client,
    timing stop-talking → first audio out → full reply.

Outputs: voice_results.json + VOICE_REPORT.md

Setup on the Pi (reuses the model fetched by raspberry-pi/run.sh):
    pip install moonshine-voice sounddevice numpy jiwer websockets
    # run from a dir containing prompts.txt and cache/moonshine/tiny-streaming/
    python voice_test.py

Toggles (top of file): MEASURE_E2E to turn the LLM+TTS round-trip on/off;
FRESH_PER_PROMPT if transcripts bleed between prompts.
"""
import asyncio
import base64
import json
import statistics as st
import sys
import threading
import time
from collections import deque

import jiwer
import numpy as np

# Native real-time deps load only on arm64; guard so the file imports anywhere.
try:
    from moonshine_voice import Transcriber, TranscriptEventListener, ModelArch
    import sounddevice as sd
    HAVE_STT = True
except Exception as _e:  # pragma: no cover - platform dependent
    HAVE_STT = False
    _STT_IMPORT_ERROR = _e
    TranscriptEventListener = object  # lets Collector subclass below

try:
    import websockets
    HAVE_WS = True
except Exception as _e:  # pragma: no cover
    HAVE_WS = False
    _WS_IMPORT_ERROR = _e

SAMPLE_RATE = 48000
MODEL_PATH = "../raspberry-pi/cache/moonshine/tiny-streaming"   # matches raspberry-pi/config.toml
MODEL_ARCH = 2                                   # tiny streaming
CONTEXT_LENGTH = 20                              # question-detection context
CHUNK = 1280                                     # 80 ms mic blocks

MEASURE_E2E = True            # exercise the real LLM+TTS round-trip
FRESH_PER_PROMPT = False      # True if transcripts bleed between prompts
REMOTE_PI_URL = "wss://api.magnusfulton.com/cse481/ws/pi"      # from config.toml
REMOTE_CLIENT_URL = REMOTE_PI_URL.replace("/ws/pi", "/ws/client")
E2E_QUIET_TIMEOUT = 3.0       # s of silence after last TTS frame = reply done
E2E_HARD_TIMEOUT = 40.0       # s overall cap per prompt

# --- WER/CER normalization (identical to the dataset eval) -----------------
_W = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                    jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
                    jiwer.ReduceToListOfListOfWords()])
_C = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                    jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
                    jiwer.ReduceToListOfListOfChars()])


def score(ref, hyp):
    ref, hyp = (ref or "").strip(), (hyp or "").strip()
    if not ref:
        return {"wer": None, "cer": None}
    o = jiwer.process_words(ref, hyp, reference_transform=_W, hypothesis_transform=_W)
    return {"wer": o.wer,
            "cer": jiwer.cer(ref, hyp, reference_transform=_C, hypothesis_transform=_C),
            "sub": o.substitutions, "del": o.deletions, "ins": o.insertions}


# --- faithful copy of QuestionPipeline (raspberry-pi/main.py) --------------
def run_question_detection(lines):
    context = deque(maxlen=CONTEXT_LENGTH)
    fires = []
    for text in lines:
        context.append(text)
        if any(sep in text for sep in (",", ".", "!", "?")):
            payload = "".join(context)
            trigger = next(c for c in text[::-1] if c in ",.!?")
            fires.append({"payload_words": len(payload.split()), "trigger": trigger})
    return fires


class Collector(TranscriptEventListener):
    """Thread-safe sink for streaming transcript events from Moonshine."""

    def __init__(self):
        self.lines = []
        self.last_line_time = None
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.lines = []
            self.last_line_time = None

    def on_line_completed(self, event):
        with self.lock:
            self.lines.append(event.line.text.strip())
            self.last_line_time = time.time()

    def on_error(self, event):  # pragma: no cover
        print("  [moonshine error]", event, file=sys.stderr)

    def snapshot(self):
        with self.lock:
            return list(self.lines), self.last_line_time


def make_transcriber(collector):
    t = Transcriber(model_path=MODEL_PATH, model_arch=ModelArch(MODEL_ARCH))
    t.add_listener(collector)
    t.start()
    return t


def capture_one(transcriber, collector):
    """Stream the mic into Moonshine until the user presses Enter to stop.

    Returns (lines, audio_seconds, finalize_latency_s, stream_rtf, t_stop).
    """
    collector.reset()
    frames_fed = [0]

    def cb(indata, frames, time_info, status):  # runs on the audio thread
        if status:
            print("  [audio status]", status, file=sys.stderr)
        mono = indata[:, 0].astype(np.float32, copy=True)
        transcriber.add_audio(mono, SAMPLE_RATE)
        frames_fed[0] += len(mono)

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=CHUNK, callback=cb)
    input("    [Enter] then speak…")
    t0 = time.time()
    stream.start()
    input("    speaking — [Enter] when done.")
    t_stop = time.time()
    stream.stop()
    stream.close()

    try:
        transcriber.stop()  # finalize any pending line
    except Exception as e:  # pragma: no cover
        print("  [stop]", e, file=sys.stderr)

    deadline = time.time() + 2.5
    while time.time() < deadline:
        _, last = collector.snapshot()
        if last and last >= t_stop:
            break
        time.sleep(0.02)

    lines, last = collector.snapshot()
    audio_s = frames_fed[0] / SAMPLE_RATE
    finalize_latency = (last - t_stop) if last else None
    stream_rtf = ((t_stop - t0) / audio_s) if audio_s else None
    return lines, audio_s, finalize_latency, stream_rtf, t_stop


async def _e2e(prompt_text):
    """Send prompt as the Pi, receive TTS as the client; time the round-trip.

    Returns dict with send→first-audio, send→last-audio, and reply audio length,
    or None if no audio came back. Mirrors oracle-cloud/main.py's Relay routing.
    """
    async with websockets.connect(REMOTE_CLIENT_URL, max_size=None, ping_interval=20) as client, \
            websockets.connect(REMOTE_PI_URL, max_size=None, ping_interval=20) as pi:
        await client.send(json.dumps({"type": "register"}))

        t_send = time.time()
        await pi.send(json.dumps({"type": "prompt", "data": prompt_text}))

        first = last = None
        total_bytes = 0
        while True:
            remaining = E2E_HARD_TIMEOUT - (time.time() - t_send)
            if remaining <= 0:
                break
            try:
                msg = await asyncio.wait_for(client.recv(), timeout=E2E_QUIET_TIMEOUT)
            except asyncio.TimeoutError:
                break  # quiet → reply finished
            if isinstance(msg, (bytes, bytearray)):
                continue
            try:
                data = json.loads(msg)
            except Exception:
                continue
            if data.get("type") == "tts" and data.get("data"):
                now = time.time()
                first = first or now
                last = now
                total_bytes += len(base64.b64decode(data["data"]))

    if first is None:
        return None
    return {"t_send": t_send, "t_first": first,
            "send_to_first_s": first - t_send,
            "send_to_last_s": last - t_send,
            "reply_audio_s": (total_bytes / 2) / SAMPLE_RATE}  # int16 @16k


def measure_e2e(prompt_text):
    try:
        return asyncio.run(_e2e(prompt_text))
    except Exception as e:
        print(f"  [e2e unavailable] {e}", file=sys.stderr)
        return None


def main():
    if not HAVE_STT:
        print("ERROR: moonshine-voice / sounddevice did not load. Run on an arm64 "
              "box (Raspberry Pi or Apple-Silicon Mac), not Intel.\n"
              f"Import error: {_STT_IMPORT_ERROR}", file=sys.stderr)
        sys.exit(1)
    if MEASURE_E2E and not HAVE_WS:
        print("WARNING: websockets not installed; skipping end-to-end LLM+TTS "
              "timing. `pip install websockets` to enable.", file=sys.stderr)

    prompts = [l.strip() for l in open("prompts.txt")
               if l.strip() and not l.startswith("#")]
    print(f"Loaded {len(prompts)} prompts. Read each line aloud when asked.")
    if MEASURE_E2E and HAVE_WS:
        print(f"End-to-end LLM+TTS timing ON via {REMOTE_PI_URL}\n")

    collector = Collector()
    transcriber = None if FRESH_PER_PROMPT else make_transcriber(collector)

    rows = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Read this aloud:\n    >>> {prompt}")
        if FRESH_PER_PROMPT:
            transcriber = make_transcriber(collector)

        lines, audio_s, latency, rtf, t_stop = capture_one(transcriber, collector)
        hyp = " ".join(lines).strip()
        sc = score(prompt, hyp)
        fires = run_question_detection(lines)

        e2e = measure_e2e(hyp or prompt) if (MEASURE_E2E and HAVE_WS and hyp) else None

        row = {
            "idx": i, "prompt": prompt, "hyp": hyp,
            "wer": sc["wer"], "cer": sc["cer"],
            "audio_s": round(audio_s, 2),
            "stt_finalize_s": round(latency, 3) if latency is not None else None,
            "stream_rtf": round(rtf, 3) if rtf is not None else None,
            "stt_lines": len(lines),
            "qd_fires": len(fires),
            "qd_comma_fires": sum(1 for f in fires if f["trigger"] == ","),
            "llm_tts_first_s": round(e2e["send_to_first_s"], 3) if e2e else None,
            "llm_tts_full_s": round(e2e["send_to_last_s"], 3) if e2e else None,
            "reply_audio_s": round(e2e["reply_audio_s"], 2) if e2e else None,
            # voice → first spoken audio: stop talking → first TTS frame heard
            "e2e_first_audio_s": round(e2e["t_first"] - t_stop, 3) if e2e else None,
        }
        rows.append(row)

        if FRESH_PER_PROMPT:
            try:
                transcriber.close()
            except Exception:
                pass
            transcriber = None
        else:
            transcriber.start()  # fresh stream for the next prompt

        w = f"{100*sc['wer']:.0f}%" if sc["wer"] is not None else "na"
        lat = f"{latency:.2f}s" if latency is not None else "na"
        e2es = f"{row['e2e_first_audio_s']:.2f}s" if row["e2e_first_audio_s"] is not None else "na"
        print(f"    heard : {hyp!r}")
        print(f"    WER {w} | STT finalize {lat} | voice→reply {e2es} | fires {len(fires)}\n")

    if transcriber is not None:
        try:
            transcriber.close()
        except Exception:
            pass

    json.dump(rows, open("voice_results.json", "w"), indent=2)
    write_report(rows)
    print("wrote voice_results.json and VOICE_REPORT.md")


def _stats(vals):
    vals = [v for v in vals if v is not None]
    return vals


def write_report(rows):
    wers = _stats([r["wer"] for r in rows])
    fin = _stats([r["stt_finalize_s"] for r in rows])
    rtfs = _stats([r["stream_rtf"] for r in rows])
    e2e_first = _stats([r["e2e_first_audio_s"] for r in rows])
    llm_first = _stats([r["llm_tts_first_s"] for r in rows])
    llm_full = _stats([r["llm_tts_full_s"] for r in rows])

    refs = [r["prompt"] for r in rows]
    hyps = [r["hyp"] for r in rows]
    o = jiwer.process_words(refs, hyps, reference_transform=_W, hypothesis_transform=_W)
    corpus_cer = jiwer.cer(refs, hyps, reference_transform=_C, hypothesis_transform=_C)

    L = []
    L.append("# Vox — Live End-to-End Voice Test\n")
    L.append("Real production stack: your live mic → **moonshine-voice** STT → "
             "question detection → **LLM + Kokoro TTS** on oracle-cloud → spoken "
             "reply. Ground truth is the fixed script in `prompts.txt`, so WER/CER "
             "are exact.\n")

    L.append("## 1. STT accuracy (exact ground truth)\n")
    L.append(f"- **Corpus WER {100*o.wer:.1f}%, CER {100*corpus_cer:.1f}%** over "
             f"{len(rows)} prompts (sub {o.substitutions}, del {o.deletions}, "
             f"ins {o.insertions}).")
    if wers:
        L.append(f"- Per-prompt WER: median {100*st.median(wers):.0f}%, "
                 f"best {100*min(wers):.0f}%, worst {100*max(wers):.0f}%.\n")

    L.append("## 2. Latency\n")
    if fin:
        L.append(f"- **STT finalize** (stop talking → final transcript): mean "
                 f"**{st.mean(fin):.2f}s**, median {st.median(fin):.2f}s, max {max(fin):.2f}s.")
    if llm_first:
        L.append(f"- **LLM+TTS** (prompt sent → first audio): mean "
                 f"**{st.mean(llm_first):.2f}s** (full reply mean "
                 f"{st.mean(llm_full):.2f}s).")
    if e2e_first:
        L.append(f"- **End-to-end** (stop talking → first spoken audio): mean "
                 f"**{st.mean(e2e_first):.2f}s**, median {st.median(e2e_first):.2f}s, "
                 f"max {max(e2e_first):.2f}s.")
    if rtfs:
        L.append(f"- Streaming RTF (STT compute vs audio): mean {st.mean(rtfs):.2f} "
                 f"(<1 = keeps up in real time).")
    if not (llm_first or e2e_first):
        L.append("- (LLM+TTS timing unavailable — server unreachable or "
                 "MEASURE_E2E off.)")
    L.append("")

    L.append("## 3. Per-prompt detail\n")
    L.append("| # | prompt | heard | WER | STT fin | voice→reply | reply len |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        w = f"{100*r['wer']:.0f}%" if r["wer"] is not None else "na"
        f = f"{r['stt_finalize_s']:.2f}s" if r["stt_finalize_s"] is not None else "na"
        e = f"{r['e2e_first_audio_s']:.2f}s" if r["e2e_first_audio_s"] is not None else "na"
        rl = f"{r['reply_audio_s']:.1f}s" if r["reply_audio_s"] is not None else "na"
        L.append(f"| {r['idx']} | {r['prompt']} | {r['hyp']} | {w} | {f} | {e} | {rl} |")

    tot_fires = sum(r["qd_fires"] for r in rows)
    tot_comma = sum(r["qd_comma_fires"] for r in rows)
    L.append("\n## 4. Question detection (production-faithful)\n")
    L.append(f"- {tot_fires} LLM triggers across {len(rows)} short utterances "
             f"({tot_comma} fired on a comma mid-utterance).")

    open("VOICE_REPORT.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
