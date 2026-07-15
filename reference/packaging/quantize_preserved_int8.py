import argparse
from pathlib import Path

import torch


QUANT_MIN_NUMEL = 4096
QUANT_MIN_DIM = 2


def quantize_state(state):
    packed = {}
    counts = {"q": 0, "value": 0, "raw": 0}
    for key, tensor in state.items():
        if not torch.is_tensor(tensor):
            packed[key] = tensor
            counts["raw"] += 1
            continue
        tensor = tensor.detach().cpu()
        if not tensor.is_floating_point():
            packed[key] = tensor
            counts["raw"] += 1
        elif tensor.dim() >= QUANT_MIN_DIM and tensor.numel() >= QUANT_MIN_NUMEL:
            value = tensor.float()
            scale = value.abs().max().item() / 127.0
            if scale > 0:
                quantized = torch.clamp(torch.round(value / scale), -127, 127).to(torch.int8)
                packed[key] = {"q": quantized, "scale": float(scale)}
                counts["q"] += 1
            else:
                packed[key] = {"value": tensor.half()}
                counts["value"] += 1
        else:
            packed[key] = {"value": tensor.half()}
            counts["value"] += 1
    return packed, counts


def restore_state(payload):
    state = payload.get("state", payload)
    restored = {}
    for key, value in state.items():
        if isinstance(value, dict) and "q" in value and "scale" in value:
            restored[key] = value["q"].to(torch.float16) * float(value["scale"])
        elif isinstance(value, dict) and "value" in value:
            restored[key] = value["value"]
        else:
            restored[key] = value
    return restored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="FP32 state dict or directory containing model.pt")
    parser.add_argument("--dst", required=True, help="Output model_int8.pt path or output directory")
    args = parser.parse_args()

    src = Path(args.src)
    if src.is_dir():
        src = src / "model.pt"
    dst = Path(args.dst)
    if dst.suffix.lower() != ".pt":
        dst = dst / "model_int8.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)

    raw = torch.load(src, map_location="cpu")
    if isinstance(raw, dict) and "state" in raw:
        raw = raw["state"]
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a state dict, got {type(raw)!r}")

    packed, counts = quantize_state(raw)
    torch.save({"state": packed}, dst)
    restored = restore_state(torch.load(dst, map_location="cpu"))

    errors = []
    for key, original in raw.items():
        if not (torch.is_tensor(original) and original.is_floating_point()):
            continue
        delta = (restored[key].float() - original.float()).abs()
        denominator = original.float().abs().max().item() + 1e-12
        errors.append((delta.max().item() / denominator, key))
    errors.sort(reverse=True)
    keys_ok = set(restored) == set(raw)

    print(f"saved={dst}")
    print(f"bytes={dst.stat().st_size}")
    print(f"q={counts['q']} value={counts['value']} raw={counts['raw']}")
    print(f"worst_rel_max={errors[0][0] * 100:.4f}% key={errors[0][1]}")
    print(f"keys_ok={keys_ok}")


if __name__ == "__main__":
    main()
