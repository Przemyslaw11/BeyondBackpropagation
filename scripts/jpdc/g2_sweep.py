"""G2 E-S/E-D: per-step energy of BP, MF-Simul and MF-Seq-nom0 stages across batch x width.

One invocation = one repeat on one GPU: an idle baseline, then every (cell, arm) unit in an
order shuffled by the repeat seed, so paired contrasts share the physical GPU (E-P).
Each unit: build model, warm up, calibrate steps/s, then run a fixed step count between two
reads of the driver energy counter (primary) under the 5 Hz power trace (cross-check).
Steps mirror the production loops without their per-batch `.item()` syncs, equal across arms.
Output: one JSON with job metadata, the idle baseline and one record per unit.

    PYTHONPATH=. python scripts/jpdc/g2_sweep.py --dataset cifar10 --repeat 0 --out results/g2/r0.json
"""

import argparse
import json
import math
import os
import platform
import random
import socket
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.algorithms.mf import mf_local_loss_fn
from src.architectures.mf_mlp import MF_MLP
from src.baselines.bp_ds import (
    READOUT_PROJECTION,
    _deep_supervised_losses,
    _detached_intermediate_activations,
)
from src.data_utils.preprocessing import DATASET_STATS, get_transforms
from src.utils.monitoring import (
    GPUEnergyMonitor,
    get_gpu_energy_counter_j,
    get_gpu_handle,
    init_nvml,
)

TRAIN_SAMPLES_PER_PASS = 45000  # CIFAR train split after val_split 0.1, as in every W1 run
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


# ---------------------------------------------------------------- data (E-D)
def gpu_augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """GPU twin of RandomCrop(32, padding=4, reflect) + RandomHorizontalFlip on NCHW."""
    n, _, h, w = x.shape
    padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
    oy = torch.randint(0, 9, (n,), device=x.device, generator=gen)
    ox = torch.randint(0, 9, (n,), device=x.device, generator=gen)
    rows = (oy[:, None] + torch.arange(h, device=x.device))[:, None, :, None]
    cols = (ox[:, None] + torch.arange(w, device=x.device))[:, None, None, :]
    batch = torch.arange(n, device=x.device)[:, None, None, None]
    chans = torch.arange(x.shape[1], device=x.device)[None, :, None, None]
    out = padded[batch, chans, rows, cols]
    flip = torch.rand(n, device=x.device, generator=gen) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(-1), out)


def load_train_tensors(dataset: str, root: str, synthetic: bool, device: torch.device):
    """Normalised CIFAR train images (N,3,32,32) and labels, resident on `device`."""
    if synthetic:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(4096, 3, 32, 32, generator=g)
        y = torch.randint(0, NUM_CLASSES[dataset], (4096,), generator=g)
        return x.to(device), y.to(device)
    import torchvision

    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    ds = cls(root=root, train=True, download=False)
    stats = DATASET_STATS[dataset]
    x = torch.tensor(ds.data).permute(0, 3, 1, 2).float().div_(255)
    x = (x - torch.tensor(stats["mean"])[:, None, None]) / torch.tensor(stats["std"])[:, None, None]
    return x.to(device), torch.tensor(ds.targets).to(device)


def gpu_batches(x, y, batch: int, gen: torch.Generator):
    if batch > len(y):
        raise ValueError(f"batch {batch} exceeds {len(y)} resident samples")
    while True:
        perm = torch.randperm(len(y), device=y.device, generator=gen)
        for i in range(0, len(perm) - batch + 1, batch):
            idx = perm[i : i + batch]
            yield gpu_augment(x[idx], gen).flatten(1), y[idx]


def host_batches(dataset: str, root: str, batch: int, device: torch.device):
    """Bridge (H): the production host pipeline, torchvision transforms in 4 workers."""
    import torchvision

    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    ds = cls(root=root, train=True, download=False, transform=get_transforms(dataset, train=True))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4,
                                         pin_memory=True, drop_last=True, persistent_workers=True)
    while True:
        for images, labels in loader:
            yield images.to(device, non_blocking=True).flatten(1), labels.to(device, non_blocking=True)


# ---------------------------------------------------------------- arms
def _opt(cfg_path: str, params) -> torch.optim.Optimizer:
    cfg = yaml.safe_load(open(cfg_path))
    o = cfg.get("optimizer") or {}
    a = cfg.get("algorithm_params") or {}
    kind = o.get("type") or a.get("optimizer_type")
    lr, wd = o.get("lr", a.get("lr")), o.get("weight_decay", a.get("weight_decay"))
    return getattr(torch.optim, kind)(params, lr=lr, weight_decay=wd)


def make_arm(arm: str, dataset: str, width: int, depth: int, device) -> Callable:
    """Returns step(x, y) for one arm; optimiser settings from the W1 configs of that arm."""
    c, dims = NUM_CLASSES[dataset], [width] * depth
    if arm == "bp":
        layers: List[nn.Module] = []
        d = 3072
        for h in dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        model = nn.Sequential(*layers, nn.Linear(d, c)).to(device)
        opt = _opt(f"configs/bp_baselines/{dataset}_mlp_3x2000_bp.yaml", model.parameters())
        crit = nn.CrossEntropyLoss()

        def step(x, y):
            loss = crit(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        step.model = model
        return step

    model = MF_MLP(input_dim=3072, hidden_dims=dims, num_classes=c).to(device)
    crit = nn.CrossEntropyLoss()
    if arm == "mf_simul":
        cfg = f"configs/mf_simul/{dataset}_mlp_3x2000.yaml"
        aux = yaml.safe_load(open(cfg))["algorithm_params"]["aux_weight"]
        model.output_layer.requires_grad_(False)  # never read by the projection readout
        opt = _opt(cfg, [p for p in model.parameters() if p.requires_grad])

        def step(x, y):
            acts = _detached_intermediate_activations(model, x)
            loss, _, _ = _deep_supervised_losses(model, acts, y, crit, READOUT_PROJECTION, aux)
            opt.zero_grad()
            loss.backward()
            opt.step()
        step.model = model
        return step

    if arm.startswith("mf_stage"):  # MF-Seq-nom0 stage k: frozen prefix, train W_k and M_k
        k = int(arm.removeprefix("mf_stage"))
        lin, act, m = model.layers[(k - 1) * 2], model.layers[(k - 1) * 2 + 1], model.get_projection_matrix(k)
        opt = _opt(f"configs/mf_nom0/{dataset}_mlp_3x2000.yaml", list(lin.parameters()) + [m])

        def step(x, y):
            with torch.no_grad():
                for j in range(k - 1):
                    x = model.layers[j * 2 + 1](model.layers[j * 2](x))
            loss = mf_local_loss_fn(act(lin(x.detach())), m, y, crit)
            opt.zero_grad()
            loss.backward()
            opt.step()
        step.model = model
        return step
    raise ValueError(f"unknown arm {arm}")


# ---------------------------------------------------------------- measurement
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_unit(step, batches, device, handle, warmup_sec: float, measure_sec: float) -> Dict:
    t0 = time.monotonic()
    n = 0
    while time.monotonic() - t0 < warmup_sec or n < 3:
        step(*next(batches))
        n += 1
        if n % 10 == 0:
            _sync(device)
    _sync(device)
    rate = n / (time.monotonic() - t0)
    steps = max(5, math.ceil(rate * measure_sec))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    monitor = GPUEnergyMonitor(interval_sec=0.2) if handle else None
    if monitor:
        monitor.start()
    _sync(device)
    e0, t1 = get_gpu_energy_counter_j(handle), time.monotonic()
    for _ in range(steps):
        step(*next(batches))
    _sync(device)
    e1, t2 = get_gpu_energy_counter_j(handle), time.monotonic()
    trace_j = monitor.stop() if monitor else None
    counter_j = e1 - e0 if e0 is not None and e1 is not None else None
    return {
        "steps": steps, "seconds": t2 - t1, "counter_energy_j": counter_j, "trace_energy_j": trace_j,
        "peak_alloc_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
    }


def idle_baseline(handle, seconds: float) -> Dict:
    e0, t0 = get_gpu_energy_counter_j(handle), time.monotonic()
    time.sleep(seconds)
    e1, t1 = get_gpu_energy_counter_j(handle), time.monotonic()
    watts = (e1 - e0) / (t1 - t0) if e0 is not None and e1 is not None else None
    return {"seconds": t1 - t0, "idle_watts": watts}


def gpu_identity(device) -> Dict:
    info = {"host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "torch": torch.__version__, "python": platform.python_version()}
    if device.type == "cuda":
        import pynvml

        h = get_gpu_handle(0)
        info.update(gpu_uuid=pynvml.nvmlDeviceGetUUID(h), gpu_name=pynvml.nvmlDeviceGetName(h),
                    driver=pynvml.nvmlSystemGetDriverVersion())
    return info


def main(argv: Optional[List[str]] = None) -> Dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", choices=sorted(NUM_CLASSES), required=True)
    p.add_argument("--repeat", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[128, 512, 2048, 8192])
    p.add_argument("--widths", type=int, nargs="+", default=[2000, 4000, 8000])
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--arms", nargs="+", default=["bp", "mf_simul", "mf_stage1", "mf_stage2", "mf_stage3"])
    p.add_argument("--pipeline", choices=["gpu", "host"], default="gpu")
    p.add_argument("--idle-sec", type=float, default=300)
    p.add_argument("--warmup-sec", type=float, default=10)
    p.add_argument("--measure-sec", type=float, default=30)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--synthetic", action="store_true", help="random data; smoke tests only")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    handle = get_gpu_handle(0) if device.type == "cuda" and init_nvml() else None
    torch.manual_seed(args.repeat)
    if args.pipeline == "gpu":
        x, y = load_train_tensors(args.dataset, args.data_root, args.synthetic, device)
    _sync(device)
    record = {"args": vars(args), "identity": gpu_identity(device),
              "idle_start": idle_baseline(handle, args.idle_sec), "units": []}

    units = [(b, w, a) for b in args.batches for w in args.widths for a in args.arms]
    random.Random(args.repeat).shuffle(units)
    for order, (b, w, arm) in enumerate(units):
        gen = torch.Generator(device=device).manual_seed(args.repeat * 100003 + order)
        batches = (gpu_batches(x, y, b, gen) if args.pipeline == "gpu"
                   else host_batches(args.dataset, args.data_root, b, device))
        step = make_arm(arm, args.dataset, w, args.depth, device)
        unit = {"order": order, "batch": b, "width": w, "arm": arm,
                **measure_unit(step, batches, device, handle, args.warmup_sec, args.measure_sec)}
        unit["samples"] = unit["steps"] * b
        if unit["counter_energy_j"] is not None:
            unit["counter_j_per_pass"] = unit["counter_energy_j"] / unit["samples"] * TRAIN_SAMPLES_PER_PASS
        record["units"].append(unit)
        del step, batches
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps(unit), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1)
    return record


if __name__ == "__main__":
    main()
