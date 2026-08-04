# GPU power governor and generation rate limiter

Stock llama.cpp runs the GPU at 100% duty cycle for the whole of every request. For a server
that answers turns all day, that means sustained peak junction temperature and sustained peak
board power — the two things that actually wear a GPU out. There is no lever for either in
upstream: [#23891](https://github.com/ggml-org/llama.cpp/issues/23891) asked for one in May 2026
and was closed not-planned, and a `grep` for `throttl|thermal|power.?limit|duty.?cycle` over
`ggml/ src/ common/ tools/` finds only log-emission rate limiters.

This adds two setpoint sources feeding one actuator.

## The two levers

**A tokens-per-second ceiling** (`--max-gen-tps`, `--max-prompt-tps`). Feed-forward: no sensors,
nothing to oscillate. It refuses to run the card faster than the answer is consumed. If 20 tok/s
is faster than anyone reads, generating at 90 was only ever producing heat. This is the cheapest
and most effective of the two, and it is worth setting even if the governor stays off.

**A closed loop over the hardware's own sensors** (`--power-governor`). Reads junction and memory
temperature and board power draw, and modulates a compute duty cycle to hold configured targets.
This is the backstop for everything the rate ceiling does not cover: long prefills, a hot room, a
dusty heatsink, a desktop competing for the same GPU.

Whichever is more restrictive wins. They compose without double-counting, because the rate
limiter only ever sleeps for the time a step came in *under* its target interval — if the
governor already stretched the step, the rate limiter adds nothing.

## Where the work happens

Policy lives here, in `common/power-governor.{h,cpp}`. Mechanism lives in ggml.

ggml owns it because that is the only place it can work. `ggml-base` resolves a device to its
sysfs hwmon node through the PCI bus id every backend already reports
(`ggml_backend_dev_props::device_id`), so telemetry needs no backend-specific code and works for
amdgpu, i915 and xe alike. The Vulkan backend then applies the duty cycle at a submission
boundary — and **drains the queue before sleeping**. That drain is the load-bearing part: queue
submission is asynchronous, so sleeping straight after a submit merely lets the card work through
its backlog. Without it the duty cycle is fiction.

Both entry points are reached through `ggml_backend_reg_get_proc_address` rather than linked, so
there is no new exported symbol and no ABI break: against a ggml *without* the governor the
lookup returns `NULL` at runtime and the governor reports itself unavailable. It only hard-fails
if it was explicitly enabled — a governor that silently does nothing is worse than one that
refuses to start, because the hardware would be running unprotected while the configuration
claims otherwise.

The decoupling is at link and run time, not at compile time: `ggml-governor.h` still has to be on
the include path, since it declares the telemetry struct. The in-tree `ggml/` carries it, so the
default `-DLLAMA_USE_SYSTEM_GGML=OFF` build needs nothing extra. Building with
`-DLLAMA_USE_SYSTEM_GGML=ON` against a **stock installed** ggml will not compile — install the
governor-enabled ggml, or build in-tree.

## Options

| Flag | Env | Default |
|---|---|---|
| `--max-gen-tps N` | `LLAMA_ARG_MAX_GEN_TPS` | 0 = unlimited |
| `--max-prompt-tps N` | `LLAMA_ARG_MAX_PROMPT_TPS` | 0 = unlimited |
| `--power-governor` / `--no-power-governor` | `LLAMA_ARG_POWER_GOVERNOR` | off |
| `--power-max-temp N` (junction °C) | `LLAMA_ARG_POWER_MAX_TEMP` | 70 |
| `--power-max-mem-temp N` (VRAM °C, instantaneous) | `LLAMA_ARG_POWER_MAX_MEM_TEMP` | 85 |
| `--power-mem-temp-sustained N` (VRAM °C, long-window) | `LLAMA_ARG_POWER_MEM_TEMP_SUSTAINED` | 80 |
| `--power-sustained-window-s N` | `LLAMA_ARG_POWER_SUSTAINED_WINDOW_S` | 300 |
| `--power-max-mem-busy N` (% of the card) | `LLAMA_ARG_POWER_MAX_MEM_BUSY` | 0 = off |
| `--power-max-gpu-busy N` (% of the card) | `LLAMA_ARG_POWER_MAX_GPU_BUSY` | 0 = off |
| `--power-budget-pct N` (% of the board's default limit) | `LLAMA_ARG_POWER_BUDGET_PCT` | 60 |
| `--power-budget-watts N` (absolute override) | `LLAMA_ARG_POWER_BUDGET_WATTS` | 0 = use pct |
| `--power-min-duty N` (%) | `LLAMA_ARG_POWER_MIN_DUTY` | 15 |
| `--power-sample-interval-ms N` | `LLAMA_ARG_POWER_SAMPLE_MS` | 250 |
| `--power-governor-hwmon PATH` | `LLAMA_ARG_POWER_HWMON` | resolve from the PCI id |

Every one of these is also a valid `--models-preset` INI key, because the preset key table is
derived from the registered CLI options.

> **Precedence trap.** In router mode, llama-server merges its own CLI args *last*, on top of
> `[*]` and the per-model section both. A `--power-*` flag on the router command line silently
> overrides every per-model INI value. Set these in the INI, not on the unit.

## The loop

Each governed signal is normalised to a headroom-relative error, so one controller handles
temperature and power uniformly:

```
e_i = (x_i - target_i) / (ceiling_i - target_i)     # 0 at target, 1 at the hardware ceiling
e   = max_i(e_i)                                    # worst offender governs
```

Ceilings come from the hardware itself (`tempN_crit`, `power1_cap_max`), never hardcoded.
Likewise the sensors are located by reading `tempN_label`, because the numbering is not stable
across drivers or kernel versions.

The controller is PI with the derivative taken on the *measurement* rather than the error, so it
starts backing off while the card is still climbing instead of overshooting and correcting
afterwards. Three details matter more than the gains:

- **Asymmetric slew** — 1.0 duty/s falling, 0.10 duty/s rising. Thermal mass means the
  measurement lags the actuation by seconds, and a symmetric loop oscillates.
- **Deadband** at 0.05 normalised error, so the duty cycle does not dither. A dithering duty
  cycle is audible as fan and coil noise.
- **Anti-windup** clamping the integral to what `min_duty` can express. Without it the term grows
  while the output is saturated and the loop cannot recover when the card finally cools.

A swinging temperature is itself a wear mechanism, so holding the target *steadily* matters as
much as hitting it.

## VRAM, and sharing the card with a desktop

Core temperature is not the only thing that wears out, and inference is not the only thing that
wants the GPU. Three more signals, and one honest gap.

**Sustained VRAM temperature** (`--power-mem-temp-sustained`, default 80 °C over a 300 s window).
`--power-max-mem-temp` catches a spike; this catches a plateau. They are different failure modes:
30 s at 95 °C is survivable, three months at 84 °C is what actually degrades memory on a box
running agentic loops around the clock. The signal is an EWMA over the window, seeded from the
first reading rather than from zero so the loop does not spend the first five minutes believing
the memory is ice cold. It is normalised against the *instantaneous* ceiling rather than crit —
sitting as far above the long-run target as a momentary spike is allowed to reach is already a
full-scale error.

**Memory bandwidth** (`--power-max-mem-busy`) and **core utilisation** (`--power-max-gpu-busy`),
both off by default. `mem_busy_percent` and `gpu_busy_percent` are counters for the *whole card*,
not for this process, and that is exactly what makes them the right lever for sharing: when a
game or a browser starts using the GPU their load lands in the same number, the total rises past
the ceiling, and inference yields. When nothing else wants the card the totals stay low and
nothing is given up. They are off by default because throttling a card that nothing else is
competing for is pure loss — turn them on for a machine that is also a desktop.

**VRAM capacity is reported, not governed.** `gpu_vram_used_bytes` and `gpu_vram_total_bytes` are
on `/metrics`, but no amount of pacing frees a byte of it: measured across every arm below, VRAM
occupancy sat at 8139 MiB whether the duty cycle was 1.00 or 0.74. Capacity is bounded by
`models-max` and `sleep-idle-seconds`, which unload models — that is the knob to reach for if a
game needs the VRAM back, and the governor deliberately does not pretend to offer an alternative.

**PCIe bandwidth is not measurable here.** amdgpu exposes `pcie_bw` on some ASICs; Navi 48 is not
one of them, so there is no counter to govern against — only `current_link_speed` and
`current_link_width` (32 GT/s ×16 on this card). It matters less than it sounds: with
`-ngl all` the weights and KV cache live in VRAM and inference generates very little host↔device
traffic, so PCIe is not the contended resource. Model loading is, and that is bounded by
`models-max` rather than by pacing.

## Measured

Radeon RX 9070 XT (Navi 48), stock 304 W board limit, kernel 7.2.0-rc5.

### The actuator, in isolation

64 chained 2048x2048 matmuls, 25 s per arm, telemetry averaged over the second half of each arm
so the reading is a steady state rather than a transient, with a 20 s cooldown between arms:

| duty | rel. throughput | driver busy% | junction | power |
|---|---|---|---|---|
| 1.00 | 1.00 | 96 | 79.8 °C | 304 W |
| 0.75 | 0.64 | 70 | 60.5 °C | 161 W |
| 0.50 | 0.25 | 49 | 46.1 °C | 68.6 W |
| 0.25 | 0.11 | 25 | 44.0 °C | 53.2 W |

Driver-reported utilisation tracks the setpoint within a few percent. That is the measurement
that matters: it says the drain works and the duty cycle is real rather than a number in a log.

### End to end

`llama-server`, gemma-4-12b-it Q4, `-ngl 99`, 3 x 300-token streamed generations per arm. The
generation rate is measured client-side from the token stream, because llama.cpp's own eval-time
counter excludes the governor's sleeps and so reports the instantaneous decode rate rather than
the rate a caller actually sees:

| arm | gen tok/s | duty | junction | VRAM °C | power | busy% | mem busy% | VRAM MiB | fan RPM | tok/J |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 65.31 | 1.00 | 68.8 | 70.6 | 289 W | 95 | 70.4 | 8139 | 3112 | 0.226 |
| `--max-gen-tps 20` | **20.00** | 1.00 | 45.0 | 64.7 | 73 W | 39 | 21.7 | 8166 | 1822 | 0.274 |
| `--power-governor` (70 °C / 182 W) | 42.37 | 0.85 | 55.3 | — | 159 W | 69 | — | — | 2143 | 0.266 |
| `--power-governor`, 55 °C / 120 W | 35.97 | 0.73 | 49.2 | — | 102 W | 62 | — | — | 1889 | 0.354 |
| `--power-max-mem-busy 40` | 36.69 | 0.80 | 47.7 | 68.2 | 98 W | 62 | **37.2** | 8139 | 1877 | 0.374 |
| `--power-max-gpu-busy 60` | 29.18 | 0.74 | 46.3 | 66.6 | 86 W | **51** | 29.3 | 8139 | 1842 | 0.341 |
| both (rate + governor) | 20.00 | 1.00 | 46.0 | — | 83 W | 49 | — | — | 1833 | 0.242 |

Five things to read out of that table.

**The rate ceiling is exact.** 20.00 tok/s against a 20 tok/s target, for a 74% cut in board
power, 23 °C off the junction and 41% off the fan. Nothing about it is approximate or reactive,
which is why it is the lever to reach for first.

**The governor only takes what it needs.** At the shipped 70 °C / 182 W targets it settled on a
duty of 0.85 and held 65% of baseline throughput while cutting power by 45%. Tighten the targets
and it gives up more; leave the card cool and it does nothing at all.

**The arbiter behaves.** In the combined arm the rate ceiling already keeps the card at 46 °C, so
the governor correctly sits at duty 1.00 and adds nothing. The two setpoints do not fight.

**The utilisation ceilings hold.** Asked for at most 40% of the memory controller, the loop
settled on 37.2%; asked for at most 60% of the core, it settled on 51%. Both undershoot slightly
rather than overshoot, which is the safe direction — these exist to guarantee headroom for
something else, so erring towards more headroom is correct. In both arms the temperature and
power targets were parked out of reach, so the utilisation signal is demonstrably governing on
its own rather than riding on a thermal limit.

**VRAM occupancy does not move.** 8139 MiB at duty 1.00 and 8139 MiB at duty 0.74. This is the
measurement behind the claim above that capacity is not something pacing can give back.

### Pacing costs more throughput, and less energy, than the duty cycle suggests

Both are visible above, and they pull in opposite directions.

Throughput falls faster than the duty setpoint - 0.64 at duty 0.75, 0.25 at duty 0.50. Part is
the queue drain at every pace point giving up CPU/GPU overlap, and part is the card's own DPM
dropping clocks during the idle gaps and having to ramp back up. It is only paid while actually
throttling: at duty 1.0 the pace point is one relaxed atomic load.

But energy per token *improves* - by 18% at the shipped settings and 57% at the aggressive ones.
Those same idle gaps let the card fall to a lower voltage/frequency point, and the baseline is
pinned at the top of the V/f curve where efficiency is worst. So pacing is not "less work for
proportionally less power": it buys back some of what it costs.

## What this does not do

It does not write the GPU's power limit. Capping `power1_cap` lets the card's own DPM sit at a
lower point on the V/f curve continuously, instead of alternating between full tilt and idle, so
it should hold a temperature target for less throughput loss and without thermal cycling - and
thermal cycling is itself a wear mechanism. But `power1_cap` is `0644 root:root`, and the router
runs `DynamicUser=yes` with `ProtectKernelTunables=yes`. Duty-cycle pacing needs no privilege at
all, which is why it is what shipped. If you can set a power cap out of band, do - the two
compose, and the governor will simply find it has less work to do.

## Observability

A controller you cannot see is a controller you cannot trust.

- One line at startup naming the resolved device, its hwmon path, which sensors were found, and
  the derived budgets.
- `/props` carries a `power_governor` object with the configuration.
- `/metrics` carries the live values: `llamacpp:power_duty_cycle`,
  `llamacpp:gpu_temp_junction_celsius`, `llamacpp:gpu_temp_mem_celsius`,
  `llamacpp:gpu_temp_mem_sustained_celsius`, `llamacpp:gpu_temp_vrmem_celsius`,
  `llamacpp:gpu_power_watts`, `llamacpp:gpu_busy_percent`,
  `llamacpp:gpu_mem_busy_percent`, `llamacpp:gpu_vram_used_bytes`,
  `llamacpp:gpu_vram_total_bytes`, `llamacpp:power_throttle_seconds_total`.

`gpu_temp_mem_sustained_celsius` is the one to chart on a machine that runs for months — it is
the only number in the set that says anything about accumulated VRAM exposure, and it is
computed whether or not `--power-mem-temp-sustained` is enabled, so it can be watched before it
is acted on. Note that `power_throttle_seconds_total` counts only the rate limiter's own sleeps;
the thermal and utilisation duty cycle is applied inside ggml and shows up as
`power_duty_cycle`.

## Tests

`tests/test-power-governor.cpp` exercises the headroom error, the loop (deadband, anti-windup,
slew asymmetry, `min_duty` clamp, convergence against a synthetic first-order thermal model with
realistic lag) and the rate-interval arithmetic. No GPU, no sensors, no timing races.

## Known limitation

Under `--models-preset`, each child llama-server runs its own governor against the *same* card.
If two models decode concurrently, both see the combined thermal load and both back off,
compounding into more throttling than either asked for. With `parallel = 1` the overlap is rare
and the error is in the safe direction. A budget shared across children is the fix if it matters.
