// paroli-stream: streaming Piper TTS over stdin/stdout, made to be driven by
// another program (a voice assistant, a game, an editor plugin...).
//
// The voice is loaded once; text comes in as JSON lines on stdin and audio
// goes out on stdout in small binary frames as soon as each decoder chunk is
// ready (paroli's streaming split: encoder + chunked decoder). A "cancel"
// command stops the current utterance between chunks (tens of ms) and drops
// everything queued, so a caller can implement barge-in.
//
// stdin  (UTF-8, one JSON object per line):
//   {"op":"speak","id":7,"text":"Olá!","length_scale":1.0,"noise_scale":0.667,"noise_w":0.8}
//   {"op":"cancel"}            stop the current utterance, drop the queue
//   {"op":"quit"}
// stdout (binary frames, little endian):
//   uint32 magic 'PRLI' (0x494C5250), uint32 type, uint32 id, uint32 len, len bytes
//   type 1 READY  id 0, payload: int32 sample_rate, int32 channels
//   type 2 AUDIO  payload: int16 PCM mono samples
//   type 3 DONE   payload: uint8 cancelled, float32 real-time factor
//   type 4 ERROR  payload: UTF-8 message
// Logs go to stderr.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#endif

#include <nlohmann/json.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "piper.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr uint32_t kMagic = 0x494C5250; // "PRLI"
enum FrameType : uint32_t { Ready = 1, Audio = 2, Done = 3, Error = 4 };

std::mutex g_outMutex;

void writeFrame(uint32_t type, uint32_t id, const void *data, uint32_t len) {
  std::lock_guard<std::mutex> lock(g_outMutex);
  uint32_t header[4] = {kMagic, type, id, len};
  fwrite(header, sizeof(header), 1, stdout);
  if (len) fwrite(data, 1, len, stdout);
  fflush(stdout);
}

struct Job {
  uint32_t id = 0;
  std::string text;
  std::optional<float> lengthScale, noiseScale, noiseW;
  uint64_t generation = 0;
};

std::mutex g_queueMutex;
std::condition_variable g_queueCv;
std::deque<Job> g_queue;
std::atomic<uint64_t> g_generation{0}; // bumped by every "cancel"
std::atomic<bool> g_currentCancel{false};
std::atomic<bool> g_quit{false};

fs::path exeDir() {
#ifdef _WIN32
  wchar_t buf[MAX_PATH] = {0};
  GetModuleFileNameW(nullptr, buf, MAX_PATH);
  return fs::path(buf).parent_path();
#else
  return fs::canonical("/proc/self/exe").parent_path();
#endif
}

void usage() {
  std::cerr << "usage: paroli-stream --encoder encoder.onnx --decoder decoder.onnx -c model.json\n"
               "                     [--espeak_data DIR] [--speaker N] [--length_scale F]\n"
               "                     [--noise_scale F] [--noise_w F] [--sentence_silence S]\n"
               "                     [--accelerator cuda] [--debug]\n"
               "Reads JSON lines on stdin, writes binary PCM frames on stdout (see source header).\n";
}

void readStdin() {
  std::string line;
  while (!g_quit.load() && std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    json j;
    try {
      j = json::parse(line);
    } catch (const std::exception &e) {
      std::string msg = std::string("bad json: ") + e.what();
      writeFrame(Error, 0, msg.data(), (uint32_t)msg.size());
      continue;
    }
    std::string op = j.value("op", "speak");
    if (op == "cancel") {
      std::lock_guard<std::mutex> lock(g_queueMutex);
      g_generation++;
      g_queue.clear();
      g_currentCancel = true;
    } else if (op == "quit") {
      break;
    } else if (op == "speak") {
      Job job;
      job.id = j.value("id", 0u);
      job.text = j.value("text", "");
      if (j.contains("length_scale")) job.lengthScale = j["length_scale"].get<float>();
      if (j.contains("noise_scale")) job.noiseScale = j["noise_scale"].get<float>();
      if (j.contains("noise_w")) job.noiseW = j["noise_w"].get<float>();
      std::lock_guard<std::mutex> lock(g_queueMutex);
      job.generation = g_generation.load();
      g_queue.push_back(std::move(job));
      g_queueCv.notify_one();
    }
  }
  // stdin closed (parent exited) or "quit": stop everything
  g_quit = true;
  g_currentCancel = true;
  g_queueCv.notify_all();
}

} // namespace

int main(int argc, char *argv[]) {
#ifdef _WIN32
  _setmode(_fileno(stdout), _O_BINARY);
#endif
  spdlog::set_default_logger(spdlog::stderr_color_mt("paroli-stream"));
  spdlog::set_level(spdlog::level::warn);

  std::string encoderPath, decoderPath, configPath, espeakData, accelerator;
  std::optional<piper::SpeakerId> speakerId;
  std::optional<float> lengthScale, noiseScale, noiseW, sentenceSilence;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { usage(); std::exit(1); }
      return argv[++i];
    };
    if (a == "--encoder") encoderPath = next();
    else if (a == "--decoder") decoderPath = next();
    else if (a == "-c" || a == "--config") configPath = next();
    else if (a == "--espeak_data") espeakData = next();
    else if (a == "--speaker") speakerId = std::stoll(next());
    else if (a == "--length_scale") lengthScale = std::stof(next());
    else if (a == "--noise_scale") noiseScale = std::stof(next());
    else if (a == "--noise_w") noiseW = std::stof(next());
    else if (a == "--sentence_silence") sentenceSilence = std::stof(next());
    else if (a == "--accelerator") accelerator = next();
    else if (a == "--debug") spdlog::set_level(spdlog::level::debug);
    else if (a == "-h" || a == "--help") { usage(); return 0; }
    else { usage(); return 1; }
  }
  if (encoderPath.empty() || decoderPath.empty() || configPath.empty()) {
    usage();
    return 1;
  }

  piper::PiperConfig config;
  piper::Voice voice;
  try {
    auto t0 = std::chrono::steady_clock::now();
    piper::loadVoice(config, "", encoderPath, decoderPath, configPath, voice, speakerId, accelerator);
    if (voice.phonemizeConfig.phonemeType == piper::eSpeakPhonemes) {
      config.eSpeakDataPath = espeakData.empty() ? (exeDir() / "espeak-ng-data").string() : espeakData;
    } else {
      config.useESpeak = false;
    }
    piper::initialize(config);
    spdlog::info("voice loaded in {:.2f} s", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  } catch (const std::exception &e) {
    std::string msg = std::string("failed to load voice: ") + e.what();
    spdlog::error(msg);
    writeFrame(Error, 0, msg.data(), (uint32_t)msg.size());
    return 2;
  }
  if (lengthScale) voice.synthesisConfig.lengthScale = *lengthScale;
  if (noiseScale) voice.synthesisConfig.noiseScale = *noiseScale;
  if (noiseW) voice.synthesisConfig.noiseW = *noiseW;
  if (sentenceSilence) voice.synthesisConfig.sentenceSilenceSeconds = *sentenceSilence;

  int32_t ready[2] = {voice.synthesisConfig.sampleRate, voice.synthesisConfig.channels};
  writeFrame(Ready, 0, ready, sizeof(ready));

  std::thread reader(readStdin);
  reader.detach(); // blocked in getline; the process exits when we return

  while (!g_quit.load()) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(g_queueMutex);
      g_queueCv.wait(lock, [] { return g_quit.load() || !g_queue.empty(); });
      if (g_quit.load()) break;
      job = std::move(g_queue.front());
      g_queue.pop_front();
      if (job.generation != g_generation.load()) continue; // cancelled while queued
      g_currentCancel = false;
    }

    std::vector<int16_t> audioBuffer;
    piper::SynthesisResult result{};
    bool failed = false;
    try {
      piper::textToAudio(
          config, voice, job.text, audioBuffer, result,
          [&]() {
            if (!audioBuffer.empty() && !g_currentCancel.load())
              writeFrame(Audio, job.id, audioBuffer.data(), (uint32_t)(audioBuffer.size() * sizeof(int16_t)));
          },
          std::nullopt, job.noiseScale, job.lengthScale, job.noiseW, &g_currentCancel);
    } catch (const std::exception &e) {
      failed = true;
      std::string msg = std::string("synthesis failed: ") + e.what();
      writeFrame(Error, job.id, msg.data(), (uint32_t)msg.size());
    }
    if (!failed) {
#pragma pack(push, 1)
      struct { uint8_t cancelled; float rtf; } done{(uint8_t)(g_currentCancel.load() ? 1 : 0), (float)result.realTimeFactor};
#pragma pack(pop)
      writeFrame(Done, job.id, &done, sizeof(done));
    }
  }

  piper::terminate(config);
  return 0;
}
