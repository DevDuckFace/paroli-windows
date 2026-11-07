#!/usr/bin/env python3
"""Drives paroli-stream: measures time-to-first-audio, real-time factor and
cancel latency, and saves what was spoken to a WAV file.

    python stream_test.py path/to/paroli-stream.exe voice_dir out.wav "texto"
"""
import json
import struct
import subprocess
import sys
import threading
import time
import wave

MAGIC = 0x494C5250
READY, AUDIO, DONE, ERROR = 1, 2, 3, 4


def read_frame(f):
    hdr = f.read(16)
    if len(hdr) < 16:
        return None
    magic, typ, ident, ln = struct.unpack("<IIII", hdr)
    assert magic == MAGIC, hex(magic)
    return typ, ident, f.read(ln)


def main():
    exe, voice, out_wav, text = sys.argv[1:5]
    t_start = time.perf_counter()
    p = subprocess.Popen(
        [exe, "--encoder", f"{voice}/encoder.onnx", "--decoder", f"{voice}/decoder.onnx",
         "-c", f"{voice}/model.json"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr)
    typ, _, payload = read_frame(p.stdout)
    assert typ == READY, (typ, payload)
    rate, ch = struct.unpack("<ii", payload)
    print(f"ready in {time.perf_counter() - t_start:.2f}s, {rate} Hz")

    def send(obj):
        p.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        p.stdin.flush()

    # 1) full utterance
    pcm = bytearray()
    t0 = time.perf_counter()
    send({"op": "speak", "id": 1, "text": text})
    first = None
    chunks = 0
    while True:
        typ, ident, payload = read_frame(p.stdout)
        if typ == AUDIO:
            if first is None:
                first = time.perf_counter() - t0
            chunks += 1
            pcm += payload
        elif typ == DONE:
            cancelled, rtf = struct.unpack("<Bf", payload)
            break
        elif typ == ERROR:
            print("ERROR", payload.decode("utf-8", "replace"))
            return
    total = time.perf_counter() - t0
    audio_s = len(pcm) / 2 / rate
    print(f"first audio after {first*1000:.0f} ms, {chunks} chunks, "
          f"synth {total:.2f}s for {audio_s:.2f}s of audio (RTF {total/audio_s:.3f}, engine {rtf:.3f})")

    # 2) cancel in the middle of a long text
    long_text = (text + " ") * 6
    t0 = time.perf_counter()
    send({"op": "speak", "id": 2, "text": long_text})
    send({"op": "speak", "id": 3, "text": "Esta frase nunca deve ser falada."})
    got = 0
    cancel_at = None
    while True:
        typ, ident, payload = read_frame(p.stdout)
        if typ == AUDIO:
            got += 1
            if got == 2 and cancel_at is None:
                cancel_at = time.perf_counter()
                send({"op": "cancel"})
        elif typ == DONE:
            cancelled, _ = struct.unpack("<Bf", payload)
            print(f"id {ident} done (cancelled={cancelled}) "
                  f"{(time.perf_counter() - cancel_at)*1000:.0f} ms after cancel")
            break
    # the queued id 3 must have been dropped: a new request answers next
    send({"op": "speak", "id": 4, "text": "Ok."})
    while True:
        typ, ident, payload = read_frame(p.stdout)
        if typ == DONE:
            print(f"next DONE id = {ident} (expect 4)")
            break

    send({"op": "quit"})
    p.stdin.close()
    p.wait(timeout=10)
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(pcm))
    print("wrote", out_wav)


if __name__ == "__main__":
    main()
