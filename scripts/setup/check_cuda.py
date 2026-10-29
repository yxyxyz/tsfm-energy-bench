"""Check whether the active PyTorch build can run on the installed NVIDIA GPU."""

from __future__ import annotations

import json

import torch


def main() -> None:
    report: dict[str, object] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        report["status"] = "FAIL: this PyTorch build cannot access CUDA"
        print(json.dumps(report, indent=2))
        raise SystemExit(1)

    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    expected_arch = f"sm_{major}{minor}"
    supported_arches = torch.cuda.get_arch_list()
    report.update(
        {
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": f"{major}.{minor}",
            "total_vram_gib": round(
                torch.cuda.get_device_properties(device).total_memory / 2**30, 2
            ),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "wheel_architectures": supported_arches,
            "native_architecture_present": expected_arch in supported_arches,
        }
    )
    if expected_arch not in supported_arches:
        report["status"] = (
            f"FAIL: wheel lacks {expected_arch}; install a current CUDA 12.8+ "
            "PyTorch wheel for RTX 50-series"
        )
        print(json.dumps(report, indent=2))
        raise SystemExit(2)
    report["status"] = "OK"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

