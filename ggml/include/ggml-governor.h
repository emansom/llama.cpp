#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stdbool.h>
#include <stdint.h>

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

#define GGML_GOVERNOR_PROC_GET_TELEMETRY "ggml_backend_dev_get_telemetry"
#define GGML_GOVERNOR_PROC_SET_PACE      "ggml_backend_dev_set_pace"

    // Physical sensor readings for a device. Fields that could not be read are NAN (floats)
    // or -1 (integers). Temperatures are degrees Celsius, power is Watts.
    struct ggml_governor_telemetry {
        float temp_edge_c;
        float temp_junction_c;
        float temp_mem_c;

        // hardware-declared critical thresholds, for deriving headroom
        float temp_edge_crit_c;
        float temp_junction_crit_c;
        float temp_mem_crit_c;

        float power_w;            // current board power draw
        float power_limit_w;      // the board's default power limit
        float power_limit_max_w;  // the highest limit the board will accept

        int32_t busy_pct;         // driver-reported utilisation, -1 if unknown
    };

    // Fill *out with the current sensor readings for dev.
    // Returns false if the device has no resolvable sensors, leaving *out untouched.
    GGML_API bool ggml_backend_dev_get_telemetry(ggml_backend_dev_t dev, struct ggml_governor_telemetry * out);

    // Set the compute duty cycle for dev. duty is clamped to (0.0, 1.0]; 1.0 disables pacing.
    // Safe to call from a different thread than the one running the backend.
    GGML_API void ggml_backend_dev_set_pace(ggml_backend_dev_t dev, float duty);

    typedef bool (*ggml_backend_dev_get_telemetry_t)(ggml_backend_dev_t dev, struct ggml_governor_telemetry * out);
    typedef void (*ggml_backend_dev_set_pace_t)     (ggml_backend_dev_t dev, float duty);

    // Called by a backend at a submission boundary. Measures the work submitted since the
    // previous pace point on this device and sleeps to hold the configured duty cycle.
    // A no-op, costing one relaxed atomic load, when no device is being paced.
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
