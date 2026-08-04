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
| `--power-max-mem-temp N` (°C) | `LLAMA_ARG_POWER_MAX_MEM_TEMP` | 85 |
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

## Measured

Radeon RX 9070 XT, 64 chained 2048×2048 matmuls × 40 iterations, duty cycle set directly:

| duty | seconds | rel. throughput | driver busy% | junction | power |
|---|---|---|---|---|---|
| 1.00 | 0.70 | 1.00 | 95 | 61 °C | 148 W |
| 0.75 | 1.04 | 0.67 | 71 | 58 °C | 99 W |
| 0.50 | 1.58 | 0.44 | 49 | 55 °C | 87 W |
| 0.25 | 3.56 | 0.20 | 26 | 51 °C | 58 W |

Driver-reported utilisation tracks the setpoint within a few percent, which is what says the
drain works and the duty cycle is real.

Throughput falls slightly faster than the duty cycle (0.67 at duty 0.75, not 0.75). That is the
cost of draining the queue at every pace point: it gives up the CPU/GPU overlap. It is only paid
while actually throttling — at duty 1.0 the pace point is one relaxed atomic load.

## What this does not do

It does not write the GPU's power limit. Capping `power1_cap` would be the better actuator —
power scales with V²f, so a 70% cap costs perhaps 10% throughput rather than 30%, and it holds
temperature steady instead of cycling it. But it is `0644 root:root`, and the router runs
`DynamicUser=yes` with `ProtectKernelTunables=yes`. Duty-cycle pacing needs no privilege at all,
which is why it is what shipped. If you can set a power cap out of band, do — the two compose.

## Observability

A controller you cannot see is a controller you cannot trust.

- One line at startup naming the resolved device, its hwmon path, which sensors were found, and
  the derived budgets.
- `/props` carries a `power_governor` object with the configuration.
- `/metrics` carries the live values: `llamacpp:power_duty_cycle`,
  `llamacpp:gpu_temp_junction_celsius`, `llamacpp:gpu_temp_mem_celsius`,
  `llamacpp:gpu_power_watts`, `llamacpp:gpu_busy_percent`,
  `llamacpp:power_throttle_seconds_total`.

## Tests

`tests/test-power-governor.cpp` exercises the headroom error, the loop (deadband, anti-windup,
slew asymmetry, `min_duty` clamp, convergence against a synthetic first-order thermal model with
realistic lag) and the rate-interval arithmetic. No GPU, no sensors, no timing races.

## Known limitation

Under `--models-preset`, each child llama-server runs its own governor against the *same* card.
If two models decode concurrently, both see the combined thermal load and both back off,
compounding into more throttling than either asked for. With `parallel = 1` the overlap is rare
and the error is in the safe direction. A budget shared across children is the fix if it matters.
