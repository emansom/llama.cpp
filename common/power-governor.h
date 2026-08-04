#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>

//
// GPU power governor and generation rate limiter.
//
// Two independent setpoint sources feed one actuator:
//
//   * a closed loop over physical sensors (junction/memory temperature, board power), which
//     holds the hardware at a configured target rather than at whatever the silicon will
//     tolerate. The duty cycle it produces is pushed down to ggml, which applies it at
//     backend submission boundaries.
//
//   * a tokens-per-second ceiling, which is feed-forward: it needs no sensors and bounds the
//     work rate at the source. Applied here, as a minimum interval between decode steps.
//
// Whichever is more restrictive wins. The two compose without double-counting because the
// rate limiter only ever sleeps for the time a step came in *under* its target interval.
//

struct common_power_params {
    bool enabled = false;   // the sensor-driven governor; the tok/s ceilings are independent

    // targets
    float   max_temp_c        = 70.0f;   // junction/hotspot
    float   max_mem_temp_c    = 85.0f;   // memory
    int32_t budget_pct        = 60;      // % of the board's default power limit
    float   budget_watts      = 0.0f;    // absolute override; 0 means derive from budget_pct

    // loop behaviour
    int32_t min_duty_pct      = 15;
    int32_t sample_interval_ms = 250;

    std::string hwmon_path;              // explicit override; empty means resolve via PCI id

    // rate ceilings, 0 means unlimited
    float max_gen_tps    = 0.0f;         // per-sequence generation rate
    float max_prompt_tps = 0.0f;         // prompt processing rate
};

// A snapshot for logging and metrics. Every field is a plain value so callers can render it
// without touching governor internals. Unavailable readings are NAN / -1.
struct common_power_status {
    bool enabled     = false;
    bool rate_capped = false;

    float duty = 1.0f;   // current compute duty cycle, 1.0 = unthrottled

    float   temp_junction_c = NAN;
    float   temp_mem_c      = NAN;
    float   temp_edge_c     = NAN;
    float   power_w         = NAN;
    int32_t busy_pct        = -1;

    // resolved targets, after budget_pct was applied to the board's limit
    float target_temp_c     = NAN;
    float target_mem_temp_c = NAN;
    float target_power_w    = NAN;

    // Cumulative time this process slept to hold the tokens/s ceilings. The thermal duty cycle
    // is applied by ggml at submission boundaries, so it does not show up here - `duty` is the
    // observable for that half.
    double throttled_seconds = 0.0;
};

struct common_power_governor;

void common_power_governor_free(common_power_governor * gov);

struct common_power_governor_deleter {
    void operator()(common_power_governor * gov) const { common_power_governor_free(gov); }
};

using common_power_governor_ptr = std::unique_ptr<common_power_governor, common_power_governor_deleter>;

// Resolve devices and start the loop.
//
// Throws std::runtime_error when params.enabled is set but no device exposes usable sensors.
// That is deliberate: a governor that silently does nothing is worse than one that refuses to
// start, because the hardware would run unprotected while the configuration claims otherwise.
// Returns a governor with the sensor loop inert (but rate ceilings live) when enabled is false.
common_power_governor_ptr common_power_governor_init(const common_power_params & params);

// Call immediately after each llama_decode.
//
//   n_prompt_tokens - tokens being prefilled in this step
//   n_gen_tokens    - sequences that advanced by one token in this step
//   work_us         - wall-clock duration of the decode call
//
// Samples sensors when the interval has elapsed, updates the duty cycle, pushes it to ggml,
// and sleeps for whatever the rate ceilings still require.
void common_power_governor_on_decode(
        common_power_governor * gov,
        int32_t                 n_prompt_tokens,
        int32_t                 n_gen_tokens,
        int64_t                 work_us);

common_power_status common_power_governor_status(const common_power_governor * gov);

// Human-readable summary of what was resolved, for the startup log.
std::string common_power_governor_describe(const common_power_governor * gov);

//
// The pieces below are the loop's arithmetic, exposed so they can be exercised directly
// against a synthetic thermal model - no GPU, no sensors, no timing races.
//

// Normalised headroom error: 0 at the target, 1 at the hardware ceiling, negative when there is
// headroom to spare. NAN when the reading or the ceiling is unavailable, so the caller can skip
// the signal rather than treat a missing sensor as a cold one.
float common_power_headroom_error(float value, float target, float ceiling);

struct common_power_loop_state {
    float duty     = 1.0f;
    float integral = 0.0f;
    float err_prev = 0.0f;
    bool  has_prev = false;
};

// Advance the controller by dt seconds against the worst-case normalised error, returning the
// new duty cycle. Pure: same inputs, same outputs.
float common_power_loop_step(common_power_loop_state & st, float err, float dt, float min_duty);

// Minimum wall-clock interval a decode step should occupy to respect the rate ceilings.
int64_t common_power_rate_interval_us(
        const common_power_params & params,
        int32_t                     n_prompt_tokens,
        int32_t                     n_gen_tokens);
