# paroli for Windows

A Windows port of [paroli](https://github.com/marty1885/paroli) (streaming
[Piper](https://github.com/rhasspy/piper) TTS), plus Brazilian Portuguese
voices converted to paroli's streaming format.

Plain Piper synthesizes a whole sentence before you hear anything. paroli
splits the VITS model into an **encoder** (text → latent, fast) and a
**decoder** run in small chunks, so audio starts playing while the rest of the
sentence is still being generated, and synthesis can be cancelled between
chunks. This port keeps that engine and makes it build and run natively on
Windows with MSVC.

Measured on a laptop (Intel CPU, no GPU used), voice `pt_BR-faber-medium`:

| | |
|---|---|
| Voice load | ~2.5 s (once) |
| First audio of a sentence | **~175 ms** |
| Synthesis speed | RTF ~0.13 (≈7× faster than real time) |
| Cancel (`{"op":"cancel"}`) to silence | **~70 ms** in the engine, ~10 ms in a waveOut client |

## What is in this repo

| Path | What |
|---|---|
| `piper/` | paroli's streaming Piper engine (library), ported to Windows |
| `paroli-cli/` | Command-line synthesizer (upstream tool) |
| `paroli-stream/` | **New**: streaming over stdin/stdout for embedding in other apps (no server deps) |
| `paroli-server/` | Upstream HTTP/WebSocket server (Drogon, Ogg/Opus) |
| `tools/export_streaming.py` | Converts a Piper training checkpoint (`.ckpt`) into `encoder.onnx` + `decoder.onnx` on current Python/PyTorch |
| `tools/stream_test.py` | Measures time-to-first-audio, RTF and cancel latency of `paroli-stream` |
| `tools/fetch-deps.ps1` | Downloads piper-phonemize (espeak-ng + onnxruntime 1.14.1) for Windows (CMake also does it automatically) |

### Changes from upstream paroli

- Builds with MSVC: ONNX Runtime wide-char model paths, no `pthread`/`dl` on
  Windows, `/utf-8`, xtensor ≥ 0.26 header layout (`piper/xt_compat.hpp`),
  dependencies through vcpkg manifest mode.
- Fixed: in `synthesize()`, short phrases (too small to chunk) **overwrote**
  the audio buffer instead of appending, dropping earlier phrases of the same
  sentence; their time was also missing from the real-time factor.
- Fixed: `textToWavFile()` passed its arguments shifted by one (`NULL` went
  into `speakerId`).
- New: `synthesize()` / `textToAudio()` take an optional
  `const std::atomic<bool>* cancel`, checked between phrases and decoder
  chunks.
- New: `paroli-stream` executable (protocol below).

## Building on Windows

Requirements: Visual Studio 2022 (Desktop C++), CMake ≥ 3.15,
[vcpkg](https://github.com/microsoft/vcpkg) with `VCPKG_ROOT` set.

Shortcut: `build.bat` does all of the steps below (`build.bat server` also
builds `paroli-server`). Manually:

```powershell
powershell -ExecutionPolicy Bypass -File tools\fetch-deps.ps1
cmake -B build -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" -DPAROLI_BUILD_SERVER=OFF
cmake --build build --config Release
```

`build\Release\` then has `paroli-cli.exe`, `paroli-stream.exe`, the DLLs and
`espeak-ng-data\` next to them, ready to copy elsewhere.

With the HTTP server (pulls Drogon, Opus, libopusenc and soxr from vcpkg, takes
a while the first time):

```powershell
cmake -B build-server -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" -DVCPKG_MANIFEST_FEATURES=server
cmake --build build-server --config Release
```

On Linux the original instructions still apply (set `ORT_ROOT` and
`PIPER_PHONEMIZE_ROOT`).

## Voices: Brazilian Portuguese

paroli needs each voice as an encoder/decoder pair, which Piper does not
publish. `tools/export_streaming.py` produces them from the official Piper
training checkpoints. Converted so far (all pt_BR, medium, 22.05 kHz, from
[rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main/pt/pt_BR),
CC0 datasets):

- `pt_BR-faber-medium`
- `pt_BR-cadu-medium`
- `pt_BR-jeff-medium`

Each voice folder contains `encoder.onnx` (~28 MB), `decoder.onnx` (~35 MB)
and `model.json` (the Piper voice config). Download them from
[caradepato/pt-br-voices-paroli](https://huggingface.co/caradepato/pt-br-voices-paroli).

### Converting a voice yourself

The upstream exporter (mush42's `piper_train.export_onnx_streaming`) needs
pytorch-lightning 1.7 and torch < 2, which do not install on current Python.
`export_streaming.py` rebuilds the generator from the checkpoint's saved
hyper-parameters instead, and loads the checkpoint with
`torch.load(weights_only=True)` so a tampered file cannot run code.

```powershell
git clone -b streaming https://github.com/mush42/piper.git tools\mush42-piper
python -m venv tools\.venv
tools\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
tools\.venv\Scripts\python -m pip install onnx numpy
tools\.venv\Scripts\python tools\export_streaming.py model.ckpt config.json out\pt_BR-faber-medium
```

(`config.json` is the voice config next to the checkpoint.)

## paroli-stream protocol

Start it once per voice:

```
paroli-stream --encoder encoder.onnx --decoder decoder.onnx -c model.json
              [--length_scale 1.0] [--noise_scale 0.667] [--noise_w 0.8]
              [--sentence_silence 0.2] [--speaker N] [--espeak_data DIR]
```

**stdin**: one JSON object per line (UTF-8).

```json
{"op":"speak","id":7,"text":"Olá! Tudo bem?"}
{"op":"cancel"}
{"op":"quit"}
```

`speak` may also carry `length_scale`, `noise_scale`, `noise_w`. `cancel`
stops the current utterance between chunks and drops everything queued.

**stdout**: binary frames, little endian: `uint32 magic 'PRLI' (0x494C5250)`,
`uint32 type`, `uint32 id`, `uint32 len`, then `len` bytes.

| type | meaning | payload |
|---|---|---|
| 1 READY | voice loaded | `int32 sample_rate, int32 channels` |
| 2 AUDIO | a chunk of the utterance `id` | 16-bit PCM mono |
| 3 DONE | utterance finished or cancelled | `uint8 cancelled, float32 rtf` |
| 4 ERROR | something failed | UTF-8 message |

Logs go to stderr. `tools/stream_test.py` is a complete client.

## License

MIT, like paroli (see `LICENSE`). Runtime components: espeak-ng is
**GPL-3.0** (shipped as a separate DLL from piper-phonemize; see
https://github.com/rhasspy/espeak-ng for its source), ONNX Runtime, fmt and
spdlog are MIT. The Piper voices are trained on CC0 datasets.

Credits: paroli by Martin Chang, Piper by Michael Hansen / Open Home
Foundation, streaming export by mush42.
