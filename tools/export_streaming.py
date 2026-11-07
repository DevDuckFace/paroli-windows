#!/usr/bin/env python3
"""Export a Piper (VITS) training checkpoint into the encoder.onnx +
decoder.onnx pair used by paroli's streaming synthesis.

Based on piper_train.export_onnx_streaming from mush42's piper fork
(streaming branch, MIT), but it does NOT need pytorch-lightning 1.7 or
torch < 2: the checkpoint is read with plain torch.load and the generator
(SynthesizerTrn) is rebuilt from the saved hyper-parameters. This lets the
export run on current Python/PyTorch (tested: Python 3.13, torch 2.x CPU).

Usage:
    python export_streaming.py model.ckpt config.json out_dir

out_dir receives encoder.onnx, decoder.onnx and model.json (the Piper voice
config, copied from config.json, which paroli reads with -c).
"""
import argparse
import json
import shutil
import sys
import types
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "mush42-piper" / "src" / "python"))

# The training-only Cython module (monotonic_align) is imported by
# vits.models but never used for inference: stub it out.
_ma = types.ModuleType("piper_train.vits.monotonic_align")
_ma.maximum_path = lambda *a, **k: None
sys.modules["piper_train.vits.monotonic_align"] = _ma

# Checkpoints pickle pytorch-lightning's AttributeDict inside
# hyper_parameters; provide a stand-in so torch.load can unpickle them.
class _AttributeDict(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

for name in ["pytorch_lightning", "pytorch_lightning.utilities",
             "pytorch_lightning.utilities.parsing", "lightning_fabric",
             "lightning_fabric.utilities", "lightning_fabric.utilities.data"]:
    sys.modules.setdefault(name, types.ModuleType(name))
for _mod in ("pytorch_lightning.utilities.parsing", "lightning_fabric.utilities.data"):
    _cls = type("AttributeDict", (_AttributeDict,), {"__module__": _mod})
    setattr(sys.modules[_mod], "AttributeDict", _cls)

from piper_train.vits import commons  # noqa: E402
from piper_train.vits.models import SynthesizerTrn  # noqa: E402

OPSET_VERSION = 15


class VitsEncoder(nn.Module):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def forward(self, x, x_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        gen = self.gen
        x, m_p, logs_p, x_mask = gen.enc_p(x, x_lengths)
        if gen.n_speakers > 1:
            g = gen.emb_g(sid).unsqueeze(-1)
        else:
            g = None
        if gen.use_sdp:
            logw = gen.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = gen.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_lengths.max()), 1).type_as(x_mask)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        if g is None:
            return z_p, y_mask
        return z_p, y_mask, g


class VitsDecoder(nn.Module):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def forward(self, z, y_mask, g=None):
        z = self.gen.flow(z, y_mask, g=g, reverse=True)
        return self.gen.dec((z * y_mask), g=g)


def build_generator(ckpt_path: Path) -> SynthesizerTrn:
    # weights_only=True: the unpickler only accepts tensors, plain
    # containers and the classes allow-listed here, so a tampered
    # checkpoint cannot run code while it is loaded.
    import collections
    import pathlib
    # Paths saved by Linux training runs (dataset paths in the hparams)
    # become plain strings: PosixPath cannot be created on Windows.
    class _PathStub(str):
        def __new__(cls, *parts):
            return str.__new__(cls, "/".join(str(p) for p in parts))
    safe = [collections.OrderedDict,
            (_PathStub, "pathlib.PosixPath"), (_PathStub, "pathlib._local.PosixPath"),
            (_PathStub, "pathlib.WindowsPath"), (_PathStub, "pathlib._local.WindowsPath")]
    for name in ("pytorch_lightning.utilities.parsing", "lightning_fabric.utilities.data"):
        safe.append(getattr(sys.modules[name], "AttributeDict"))
    torch.serialization.add_safe_globals(safe)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    hp = dict(ckpt["hyper_parameters"])
    gin = hp.get("gin_channels", 0)
    if hp["num_speakers"] > 1 and gin <= 0:
        gin = 512
    gen = SynthesizerTrn(
        n_vocab=hp["num_symbols"],
        spec_channels=hp.get("filter_length", 1024) // 2 + 1,
        segment_size=hp.get("segment_size", 8192) // hp.get("hop_length", 256),
        inter_channels=hp.get("inter_channels", 192),
        hidden_channels=hp.get("hidden_channels", 192),
        filter_channels=hp.get("filter_channels", 768),
        n_heads=hp.get("n_heads", 2),
        n_layers=hp.get("n_layers", 6),
        kernel_size=hp.get("kernel_size", 3),
        p_dropout=hp.get("p_dropout", 0.1),
        resblock=hp.get("resblock", "2"),
        resblock_kernel_sizes=hp.get("resblock_kernel_sizes", (3, 5, 7)),
        resblock_dilation_sizes=hp.get("resblock_dilation_sizes", ((1, 2), (2, 6), (3, 12))),
        upsample_rates=hp.get("upsample_rates", (8, 8, 4)),
        upsample_initial_channel=hp.get("upsample_initial_channel", 256),
        upsample_kernel_sizes=hp.get("upsample_kernel_sizes", (16, 16, 8)),
        n_speakers=hp["num_speakers"],
        gin_channels=gin,
        use_sdp=hp.get("use_sdp", True),
    )
    state = {k[len("model_g."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model_g.")}
    missing, unexpected = gen.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"checkpoint is missing generator weights: {missing[:5]}...")
    gen.eval()
    with torch.no_grad():
        gen.dec.remove_weight_norm()
    return gen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("config", help="Piper voice config.json from the checkpoint repo")
    ap.add_argument("output_dir")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1234)
    gen = build_generator(Path(args.checkpoint))

    multi = gen.n_speakers > 1
    seq = torch.randint(low=0, high=gen.n_vocab, size=(1, 50), dtype=torch.long)
    seq_len = torch.LongTensor([seq.size(1)])
    sid = torch.LongTensor([0]) if multi else None
    scales = torch.FloatTensor([0.667, 1.0, 0.8])

    enc = VitsEncoder(gen).eval()
    enc_outputs = ["z", "y_mask"] + (["g"] if multi else [])
    with torch.no_grad():
        torch.onnx.export(
            enc, (seq, seq_len, scales, sid), str(out / "encoder.onnx"),
            opset_version=OPSET_VERSION, dynamo=False,
            input_names=["input", "input_lengths", "scales"] + (["sid"] if multi else []),
            output_names=enc_outputs,
            dynamic_axes={
                "input": {0: "batch_size", 1: "phonemes"},
                "input_lengths": {0: "batch_size"},
                "z": {0: "batch_size", 2: "time"},
                "y_mask": {0: "batch_size", 2: "time"},
            },
        )
        dec_in = enc(seq, seq_len, scales, sid)
        dec = VitsDecoder(gen).eval()
        torch.onnx.export(
            dec, tuple(dec_in), str(out / "decoder.onnx"),
            opset_version=OPSET_VERSION, dynamo=False,
            input_names=["z", "y_mask"] + (["g"] if multi else []),
            output_names=["output"],
            dynamic_axes={
                "z": {0: "batch_size", 2: "time"},
                "y_mask": {0: "batch_size", 2: "time"},
                "output": {0: "batch_size", 2: "time"},
            },
        )
    shutil.copyfile(args.config, out / "model.json")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print(f"exported {out}: espeak={cfg.get('espeak', {}).get('voice')} "
          f"rate={cfg.get('audio', {}).get('sample_rate')} speakers={gen.n_speakers}")


if __name__ == "__main__":
    main()
