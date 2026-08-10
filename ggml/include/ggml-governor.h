#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

//
// GPU power governor support.
//
// This provides two orthogonal primitives:
//
//   1. Telemetry - read physical sensors (temperature, power draw, utilisation) for a ggml
//      device. Implemented generically on Linux by resolving the device's PCI bus id
//      (ggml_backend_dev_props::device_id) to its sysfs hwmon node, so it works for any
//      backend driving an amdgpu / i915 / xe device without backend-specific code.
//
//   2. Pacing - a duty cycle applied at backend submission boundaries. A backend calls
//      ggml_governor_pace_point() between command submissions; when a duty cycle below 1.0
//      is set for the device, the call sleeps long enough to hold that duty.
//
// The policy (what the target is, and how the duty cycle is derived from telemetry) lives in
// the caller. ggml only provides the measurement and the actuator.
//
// Both entry points are also exposed through ggml_backend_reg_get_proc_address() under the
// names below, so callers can link against a ggml build without governor support and detect
// its absence at runtime rather than at link time.
//

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_GOVERNOR_PROC_GET_TELEMETRY    "ggml_backend_dev_get_telemetry"
#define GGML_GOVERNOR_PROC_SET_PACE         "ggml_backend_dev_set_pace"
#define GGML_GOVERNOR_PROC_SET_PACE_EVERY_N "ggml_backend_dev_set_pace_every_n"
#define GGML_GOVERNOR_PROC_SET_SOFT_START   "ggml_backend_dev_set_soft_start"

    // Physical sensor readings for a device. Fields that could not be read are NAN (floats),
    // -1 (signed integers) or 0 (sizes). Temperatures are degrees Celsius, power is Watts.
    //
    // The caller MUST set `size` to sizeof(struct ggml_governor_telemetry) before the call.
    // ggml writes at most that many bytes, so a caller compiled against an older header is not
    // overflowed when the library is upgraded underneath it - which is a real case here,
    // because ggml and its consumers ship as separate packages.
    struct ggml_governor_telemetry {
        uint32_t size;

        float temp_edge_c;
        float temp_junction_c;
        float temp_mem_c;         // VRAM
        float temp_vrmem_c;       // VRAM voltage regulator

        // hardware-declared critical thresholds, for deriving headroom
        float temp_edge_crit_c;
        float temp_junction_crit_c;
        float temp_mem_crit_c;

        float power_w;            // current board power draw
        float power_limit_w;      // the board's default power limit
        float power_limit_max_w;  // the highest limit the board will accept

        int32_t busy_pct;         // GPU core utilisation, -1 if unknown
        int32_t mem_busy_pct;     // memory controller utilisation, -1 if unknown

        // Capacity, in bytes. Reported for observability only: a compute duty cycle cannot
        // free VRAM, so nothing the governor does moves these numbers.
        uint64_t vram_used;
        uint64_t vram_total;

        // Pacing activity, cumulative since process start.
        //
        // Each pace point takes the device from loaded to idle and back, which is a load step
        // seen by the power supply. The peak is unchanged - pacing never raises draw above what
        // an unpaced card would pull - but the RATE of those steps is what a duty cycle adds,
        // and it is otherwise invisible: board power is only reported as a ~100 ms rolling
        // average, three orders of magnitude too slow to see a transient. Take the derivative
        // of pace_points_total to get load steps per second.
        uint64_t pace_points_total;
        uint64_t pace_sleep_us_total;

        // Soft-start ramps begun, i.e. returns to load after an idle gap. The count going up
        // is the only direct evidence the ramp is firing, since the transient it targets is
        // far too fast for any sensor on the board to show.
        uint64_t soft_start_ramps_total;
    };

    // Zero a telemetry struct and stamp its size. Always use this rather than initialising by
    // hand: `size` is what bounds how much the library writes back.
    static inline void ggml_governor_telemetry_init(struct ggml_governor_telemetry * t) {
        memset(t, 0, sizeof(*t));
        t->size = (uint32_t) sizeof(*t);
    }

    // Fill *out with the current sensor readings for dev.
    // Returns false if the device has no resolvable sensors, leaving *out untouched.
    GGML_API bool ggml_backend_dev_get_telemetry(ggml_backend_dev_t dev, struct ggml_governor_telemetry * out);

    // Set the compute duty cycle for dev. duty is clamped to (0.0, 1.0]; 1.0 disables pacing.
    // Safe to call from a different thread than the one running the backend.
    GGML_API void ggml_backend_dev_set_pace(ggml_backend_dev_t dev, float duty);

    // Ramp the duty cycle up over `ramp_ms` when the device emerges from idle, starting from
    // `start_duty`, instead of going straight to full load.
    //
    // This targets a different thing from the duty cycle proper. Going from idle to a full
    // prompt-processing graph steps the board current from near nothing to its maximum in
    // milliseconds, and no closed loop sampling at hundreds of milliseconds can intervene in
    // time. The ramp does not work by lowering current directly - during the loaded part of
    // any duty cycle the card is at full tilt regardless - it works because the GPU's own DPM
    // needs sustained load to climb its boost curve. Short bursts do not give it that time, so
    // clocks, and therefore current, rise gradually rather than stepping.
    //
    // ramp_ms of 0 disables it. While a ramp is in progress the every-n granularity is ignored
    // and every boundary is a pace point, since short bursts are the entire mechanism.
    GGML_API void ggml_backend_dev_set_soft_start(ggml_backend_dev_t dev, uint32_t ramp_ms, float start_duty);

    // Pace at most once every n submission boundaries, instead of at every one.
    //
    // The duty cycle is unchanged - work simply accumulates across the skipped boundaries and
    // is paid off in one longer sleep - so this trades how smoothly the duty is spread for how
    // often the device is switched between loaded and idle. Raise it to reduce load-step
    // frequency on the power supply; leave it at 1 for the smoothest thermal behaviour.
    // Values below 1 are treated as 1.
    GGML_API void ggml_backend_dev_set_pace_every_n(ggml_backend_dev_t dev, uint32_t n);

    typedef bool (*ggml_backend_dev_get_telemetry_t)   (ggml_backend_dev_t dev, struct ggml_governor_telemetry * out);
    typedef void (*ggml_backend_dev_set_pace_t)        (ggml_backend_dev_t dev, float duty);
    typedef void (*ggml_backend_dev_set_pace_every_n_t)(ggml_backend_dev_t dev, uint32_t n);
    typedef void (*ggml_backend_dev_set_soft_start_t)  (ggml_backend_dev_t dev, uint32_t ramp_ms, float start_duty);

    // Called by a backend at a submission boundary to ask whether this one is a pace point.
    // Counts the boundary and applies the every-n granularity, so the backend can skip the
    // expensive part - draining the queue - on boundaries that will not be paced.
    // A no-op returning false, costing one relaxed atomic load, when nothing is being paced.
    GGML_API bool ggml_governor_pace_due(ggml_backend_dev_t dev);

    // Called by a backend once ggml_governor_pace_due() has returned true AND the queue has
    // been drained. Measures the work submitted since the previous pace point on this device
    // and sleeps to hold the configured duty cycle.
    GGML_API void ggml_governor_pace_point(ggml_backend_dev_t dev);

    // True when dev currently has a duty cycle below 1.0. Backends can use this to choose a
    // finer submission granularity while pacing is active.
    GGML_API bool ggml_governor_pace_active(ggml_backend_dev_t dev);

    // Discard the accumulated work measurement for dev. A backend should call this when it is
    // about to block on unrelated work (a long host transfer, an idle wait), so that the gap
    // is not mistaken for compute and paid for with a sleep.
    GGML_API void ggml_governor_pace_reset(ggml_backend_dev_t dev);

#ifdef __cplusplus
}
#endif
