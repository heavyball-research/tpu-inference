# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reproduce the v6e sustained-HBM-bandwidth drop that looks like a clock
throttle in e2e decode profiles.

Pure JAX, no vLLM or flywheel: one jitted step streams the Qwen3-4B decode
weights from HBM (36 layers of qkv, o_proj, gate_up and down at batch 16,
then the lm_head; ~8 GB per step, every layer with its own weights). Every
phase runs this one compiled step:

    sustained  steps back to back
    idle:MS    every step followed by MS ms of device idle

Each phase runs untraced for its wall-clock step time, then traced for the
device step times, the per-op spread of gate_up and lm_head, their per-layer
means and the TensorCore throttle counters of the trace window.

On v6e the sustained phase averages ~1.2 TB/s with gate_up p10 ~68 us and
p90 ~130 us, while every TC throttle counter stays 0: the bandwidth drops on
the memory side, not the clock. With 7 ms idle after each step the same step
runs steady at ~1.6 TB/s.

    python hbm_sustained_bandwidth.py run OUT_DIR [--phases P ...]
    python hbm_sustained_bandwidth.py analyze TRACE.json.gz \
        [--op LABEL=REGEX@HLO_SUBSTR ...] [--layers]

`analyze` reads any trace whose model step is a `jit_step_fun_impl` module,
a tpu-inference vLLM profile included; pass --op for its shapes.
"""

import argparse
import collections
import gzip
import json
import math
import pathlib
import re
import statistics
import time

import jax
import jax.numpy as jnp

NUM_LAYERS = 36
BATCH, HIDDEN, INTERMEDIATE, VOCAB = 16, 2560, 9728, 151936
Q_DIM, QKV_DIM = 32 * 128, (32 + 2 * 8) * 128
STEPS_PER_PHASE = 50
DEFAULT_PHASES = ["sustained", "idle:1", "idle:3", "idle:7", "sustained"]
# Ops by their output shape in the HLO text, as XLA fuses the rmsnorm into
# the matmul.
DEFAULT_OPS = [
    f"gate_up=^fusion@= bf16[{BATCH},{2 * INTERMEDIATE}]",
    f"lm_head=^fusion@= bf16[{BATCH},{VOCAB}]",
]
STEP_MODULE = "jit_step_fun_impl"
TC_THROTTLE = "PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_"
# A shape with its layout in the HLO text; S(1) marks a VMEM buffer.
HLO_SHAPE = re.compile(r"\w+\[([\d,]*)\]\{([^}]*)\}")
# A sustained gate_up p90 this far above its p10 reproduces the drop.
SPREAD_THRESHOLD = 1.3


def rmsnorm(x):
    xf = x.astype(jnp.float32)
    return (xf *
            jax.lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + 1e-6)).astype(
                jnp.bfloat16)


def step_fun_impl(weights, lm_head, x):
    """One decode step of weight matmuls; named like tpu-inference's model
    step so its trace module is `jit_step_fun_impl`."""
    for w_qkv, w_o, w_gate_up, w_down in weights:
        qkv = rmsnorm(x) @ w_qkv
        x = x + rmsnorm(qkv[:, :Q_DIM]) @ w_o
        gate_up = rmsnorm(x) @ w_gate_up
        x = x + (jax.nn.silu(gate_up[:, :INTERMEDIATE]) *
                 gate_up[:, INTERMEDIATE:]) @ w_down
    logits = rmsnorm(x) @ lm_head.T
    return rmsnorm(x), logits


def largest_in_vmem(long_name):
    """Whether an op's largest buffer (its weight, for a decode matmul) is
    in VMEM: XLA prefetched it, so the op does not stream it from HBM."""
    shapes = HLO_SHAPE.findall(long_name)
    if not shapes:
        raise RuntimeError(f"no shape in HLO text {long_name[:200]!r}")
    _, layout = max(
        shapes, key=lambda s: math.prod(int(d) for d in s[0].split(",") if d))
    return "S(1)" in layout


def load_trace(path):
    with gzip.open(path) as f:
        events = json.load(f)["traceEvents"]
    procs, threads = {}, {}
    for e in events:
        if e.get("ph") == "M" and e["name"] == "process_name":
            procs[e["pid"]] = e["args"]["name"]
        elif e.get("ph") == "M" and e["name"] == "thread_name":
            threads[(e["pid"], e["tid"])] = e["args"]["name"]
    devs = [p for p, n in procs.items() if n == "/device:TPU:0"]
    if not devs:
        raise RuntimeError(f"{path}: no /device:TPU:0 process")
    dev = devs[0]
    tid = {n: t for (p, t), n in threads.items() if p == dev}
    for name in ("XLA Modules", "XLA Ops", "counters_0"):
        if name not in tid:
            raise RuntimeError(f"{path}: device has no {name!r} thread")
    return events, dev, tid


def analyze_trace(path, op_specs, module_prefix=STEP_MODULE):
    """Step times, per-op spread and per-layer means of the ops in
    `op_specs` that stream from HBM, and the TC throttle counters of one
    trace."""
    events, dev, tid = load_trace(path)

    def on(e, thread):
        return (e.get("ph") == "X" and e.get("pid") == dev
                and e.get("tid") == tid[thread])

    # The first and last steps may be cut by the trace window.
    steps = sorted(
        (e for e in events
         if on(e, "XLA Modules") and e["name"].startswith(module_prefix)),
        key=lambda e: e["ts"])[1:-1]
    if len(steps) < 2:
        raise RuntimeError(f"{path}: fewer than 2 whole {module_prefix} steps")
    ops = sorted((e for e in events if on(e, "XLA Ops")),
                 key=lambda e: e["ts"])
    specs = []
    for spec in op_specs:
        label, pat = spec.split("=", 1)
        regex, _, sub = pat.partition("@")
        specs.append((label, re.compile(regex), sub))
    by_layer = {label: collections.defaultdict(list) for label, _, _ in specs}
    in_vmem = collections.Counter()
    j = 0
    for s in steps:
        t0, t1 = s["ts"], s["ts"] + s["dur"]
        while j < len(ops) and ops[j]["ts"] < t0:
            j += 1
        counts = collections.Counter()
        k = j
        while k < len(ops) and ops[k]["ts"] < t1:
            o = ops[k]
            long_name = o["args"].get("long_name", "")
            for label, regex, sub in specs:
                if regex.search(o["name"]) and sub in long_name:
                    # The layer index counts every match, VMEM ones too.
                    if largest_in_vmem(long_name):
                        in_vmem[label] += 1
                    else:
                        by_layer[label][counts[label]].append(o["dur"])
                    counts[label] += 1
            k += 1
    result = {"trace": str(path), "steps": len(steps), "ops": {}}
    span = steps[-1]["ts"] + steps[-1]["dur"] - steps[0]["ts"]
    result["step_durs_us"] = [s["dur"] for s in steps]
    result["step_us"] = statistics.fmean(result["step_durs_us"])
    result["device_busy"] = sum(result["step_durs_us"]) / span
    for label, _, _ in specs:
        layers = by_layer[label]
        if not layers:
            raise RuntimeError(f"{path}: no HBM-streaming op matches "
                               f"{label!r} ({in_vmem[label]} in VMEM)")
        every = [d for durs in layers.values() for d in durs]
        deciles = statistics.quantiles(every, n=10)
        result["ops"][label] = {
            "count": len(every),
            "in_vmem": in_vmem[label],
            "mean_us": statistics.fmean(every),
            "p10_us": deciles[0],
            "p50_us": statistics.median(every),
            "p90_us": deciles[-1],
            "layer_means_us": {
                i: statistics.fmean(layers[i])
                for i in sorted(layers)
            },
        }
    tc = {}
    for e in events:
        if (e.get("pid") == dev and e.get("tid") == tid["counters_0"]
                and TC_THROTTLE in e.get("name", "")):
            tc[e["name"].split(TC_THROTTLE,
                               1)[1]] = int(e["args"]["counter_value"])
    if "CLOCKS_SKIPPED" not in tc or "CYCLE_COUNT" not in tc:
        raise RuntimeError(f"{path}: no TC throttle counters in counters_0")
    result["tc_throttle"] = tc
    return result


def print_analysis(r, step_bytes=None):
    durs = r["step_durs_us"]

    def tbps(us):
        return f" ({step_bytes / us / 1e6:.2f} TB/s)" if step_bytes else ""

    print(f"  device step mean {r['step_us'] / 1e3:.3f} ms"
          f"{tbps(r['step_us'])}, min {min(durs) / 1e3:.3f}{tbps(min(durs))}"
          f", max {max(durs) / 1e3:.3f}{tbps(max(durs))}; device busy "
          f"{r['device_busy']:.0%} ({r['steps']} steps)")
    for label, s in r["ops"].items():
        print(f"  {label:>8}: mean {s['mean_us']:7.1f} us  p10 "
              f"{s['p10_us']:7.1f}  p50 {s['p50_us']:7.1f}  p90 "
              f"{s['p90_us']:7.1f}  p90/p10 {s['p90_us'] / s['p10_us']:.2f}"
              f"  ({s['count']} ops from HBM, {s['in_vmem']} from VMEM "
              f"left out)")
    tc = r["tc_throttle"]
    skipped = {k: v for k, v in tc.items() if v and k != "CYCLE_COUNT"}
    print(f"  TC throttle: CYCLE_COUNT {tc['CYCLE_COUNT']}, CLOCKS_SKIPPED "
          f"{tc['CLOCKS_SKIPPED']}, non-zero others: {skipped or 'none'}")


def print_layers(r):
    means = {k: v["layer_means_us"] for k, v in r["ops"].items()}
    layers = sorted({i for m in means.values() for i in m})
    print("  layer" + "".join(f"{label:>10}" for label in means))
    for i in layers:
        print(f"  {i:5d}" + "".join(f"{m[i]:10.1f}" if i in m else f"{'-':>10}"
                                    for m in means.values()))


def trace_file(phase_dir):
    found = sorted(phase_dir.glob("plugins/profile/*/*.trace.json.gz"))
    if len(found) != 1:
        raise RuntimeError(f"{phase_dir}: expected one trace, found {found}")
    return found[0]


def phase_idle_ms(name):
    if name == "sustained":
        return 0.0
    kind, _, ms = name.partition(":")
    if kind != "idle" or not ms:
        raise ValueError(f"phase {name!r} is neither sustained nor idle:MS")
    return float(ms)


def run(args):
    idle = {name: phase_idle_ms(name) for name in args.phases}
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"jax {jax.__version__}, devices {jax.devices()}", flush=True)
    key = jax.random.key(0)

    def weight(shape):
        nonlocal key
        key, sub = jax.random.split(key)
        return (jax.random.normal(sub, shape, jnp.float32) *
                shape[0]**-0.5).astype(jnp.bfloat16)

    weights = tuple(
        (weight((HIDDEN, QKV_DIM)), weight((Q_DIM, HIDDEN)),
         weight((HIDDEN, 2 * INTERMEDIATE)), weight((INTERMEDIATE, HIDDEN)))
        for _ in range(NUM_LAYERS))
    lm_head = weight((VOCAB, HIDDEN))
    x = weight((BATCH, HIDDEN))
    step_bytes = (sum(w.nbytes for layer in weights
                      for w in layer) + lm_head.nbytes)
    step = jax.jit(step_fun_impl)

    def run_steps(n, idle_ms):
        nonlocal x
        for _ in range(n):
            x, logits = step(weights, lm_head, x)
            if idle_ms:
                jax.block_until_ready(logits)
                time.sleep(idle_ms / 1e3)
        jax.block_until_ready(logits)

    run_steps(3, 0.0)
    print(f"step reads {step_bytes / 1e9:.2f} GB of weights", flush=True)
    results = []
    for i, name in enumerate(args.phases):
        start = time.perf_counter()
        run_steps(args.steps, idle[name])
        wall_ms = (time.perf_counter() - start) / args.steps * 1e3
        phase_dir = out_dir / f"{i}_{name.replace(':', '')}"
        with jax.profiler.trace(str(phase_dir)):
            run_steps(args.steps, idle[name])
        r = analyze_trace(trace_file(phase_dir), DEFAULT_OPS)
        r.update(phase=name,
                 idle_ms=idle[name],
                 step_bytes=step_bytes,
                 wall_ms_untraced=wall_ms)
        results.append(r)
        print(f"PHASE {i} {name}: untraced wall {wall_ms:.3f} ms/step",
              flush=True)
        print_analysis(r, step_bytes)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=1))
    first = next((r for r in results if r["phase"] == "sustained"), None)
    if first:
        print(f"per-layer means, first sustained phase ({first['trace']}):")
        print_layers(first)
    verdict(results)


def verdict(results):
    sustained = [r for r in results if r["phase"] == "sustained"]
    if not sustained:
        print("VERDICT: no sustained phase, nothing to judge")
        return
    spreads = [
        r["ops"]["gate_up"]["p90_us"] / r["ops"]["gate_up"]["p10_us"]
        for r in sustained
    ]
    skipped = sum(r["tc_throttle"]["CLOCKS_SKIPPED"] for r in results)
    reproduced = min(spreads) >= SPREAD_THRESHOLD
    print(f"VERDICT: sustained gate_up p90/p10 "
          f"{', '.join(f'{s:.2f}' for s in spreads)} (threshold "
          f"{SPREAD_THRESHOLD}); TC CLOCKS_SKIPPED over all phases {skipped}")
    for r in results:
        print(
            f"  {r['phase']:>9}: {r['step_bytes'] / r['step_us'] / 1e6:.2f}"
            f" TB/s per device step, gate_up p90/p10 "
            f"{r['ops']['gate_up']['p90_us'] / r['ops']['gate_up']['p10_us']:.2f}"
        )
    if reproduced and skipped == 0:
        print("REPRODUCED: sustained HBM reads slow down while the "
              "TensorCore clock is never throttled")
    elif reproduced:
        print("REPRODUCED WITH TC THROTTLE: the clock was throttled too, "
              "so this run differs from the e2e case")
    else:
        print("NOT REPRODUCED: the sustained phase ran at a steady speed")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="run the phases and analyze them")
    p_run.add_argument("out_dir")
    p_run.add_argument("--phases",
                       nargs="+",
                       default=DEFAULT_PHASES,
                       help="sustained or idle:MS, in run order")
    p_run.add_argument("--steps", type=int, default=STEPS_PER_PHASE)
    p_an = sub.add_parser("analyze", help="analyze an existing trace")
    p_an.add_argument("trace")
    p_an.add_argument("--op",
                      action="append",
                      help="LABEL=REGEX@HLO_SUBSTR; the k-th match in a "
                      "step is layer k (default: the Qwen3-4B b16 ops)")
    p_an.add_argument("--module-prefix", default=STEP_MODULE)
    p_an.add_argument("--layers", action="store_true")
    args = parser.parse_args()
    if args.cmd == "run":
        run(args)
        return
    r = analyze_trace(args.trace, args.op or DEFAULT_OPS, args.module_prefix)
    print_analysis(r)
    if args.layers:
        print_layers(r)


if __name__ == "__main__":
    main()
