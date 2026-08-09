#include "power-governor.h"

#include "log.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-governor.h"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

// Controller gains. The integral term does the steady-state work - it is what holds a duty
// cycle below 1.0 while the error sits at zero - so the proportional and derivative terms only
// need to shape the approach. The derivative acts on the measured error, i.e. on the
// temperature slope, which is what lets the loop start backing off while the card is still
// below target instead of overshooting and correcting afterwards.
static const float GOV_KP = 0.50f;
static const float GOV_KI = 0.40f;   // per second
static const float GOV_KD = 0.15f;   // seconds

// Below this normalised error the loop stops integrating. Without it the duty cycle dithers
// around the setpoint, and a dithering duty cycle is audible as fan and coil noise.
static const float GOV_DEADBAND = 0.05f;

// Asymmetric slew, in duty units per second. Thermal mass means the measurement lags the
// actuation by seconds, so a symmetric loop oscillates: back off quickly, recover slowly.
static const float GOV_MAX_FALL_PER_S = 1.00f;   // 1 s to go from full speed to a standstill
static const float GOV_MAX_RISE_PER_S = 0.10f;   // 10 s to climb back to full speed

// Largest dt the controller will integrate or slew over in one step, in seconds.
//
// The slew limits are per second, so stepping the loop with a dt of two minutes would permit a
// rise of 12.0 duty units and snap straight to the target - defeating the asymmetric slew
// entirely. That is not hypothetical: before the loop was put on a timer it only advanced on
// decode, so the first request after an idle period stepped with the whole idle gap as its dt.
// Clamping here bounds one step to the same authority it would have had if the tick had not
// been missed. The sustained VRAM average deliberately does NOT use this - it wants true
// elapsed time, because it is a time-average rather than a rate limit.
static const float GOV_MAX_STEP_DT_S = 1.0f;

// Rate-limiter sleeps are served in slices so that a cancelled request or a Ctrl-C is not
// stuck behind a long nanosleep.
static const int64_t GOV_SLICE_US = 25 * 1000;

// A single rate-limiter sleep never exceeds this. A step that lands far under its target
// interval is paced towards the target over several steps rather than in one long stall.
static const int64_t GOV_MAX_SLEEP_US = 250 * 1000;

namespace {

struct gov_device {
    ggml_backend_dev_t dev = nullptr;

    ggml_backend_dev_get_telemetry_t get_telemetry = nullptr;
    ggml_backend_dev_set_pace_t      set_pace      = nullptr;

    std::string name;
    std::string id;

    // resolved once from the hardware's own declared limits
    float crit_junction_c = NAN;
    float crit_mem_c      = NAN;
    float power_limit_w   = NAN;
    float power_ceiling_w = NAN;
    float target_power_w  = NAN;
};

void sleep_us_sliced(int64_t total_us) {
    while (total_us > 0) {
        const int64_t slice = std::min(total_us, GOV_SLICE_US);
        std::this_thread::sleep_for(std::chrono::microseconds(slice));
        total_us -= slice;
    }
}

} // namespace

float common_power_headroom_error(float value, float target, float ceiling) {
    if (std::isnan(value) || std::isnan(target) || std::isnan(ceiling)) {
        return NAN;
    }
    if (ceiling <= target) {
        return NAN;
    }
    return (value - target) / (ceiling - target);
}

float common_power_loop_step(common_power_loop_state & st, float err, float dt, float min_duty) {
    dt       = std::max(dt, 1e-3f);
    min_duty = std::clamp(min_duty, 0.01f, 1.0f);

    // Integrate outside the deadband only, and clamp so the integral can never demand more
    // reduction than min_duty allows. Without the clamp the term keeps growing while the output
    // is saturated, and the loop cannot recover when the card finally cools.
    if (std::fabs(err) > GOV_DEADBAND) {
        st.integral += err * dt;
    }
    st.integral = std::clamp(st.integral, 0.0f, (1.0f - min_duty) / GOV_KI);

    const float derivative = st.has_prev ? (err - st.err_prev) / dt : 0.0f;
    st.err_prev = err;
    st.has_prev = true;

    const float correction = GOV_KP * err + GOV_KI * st.integral + GOV_KD * derivative;

    float target_duty = std::clamp(1.0f - correction, min_duty, 1.0f);

    // Slew limit, asymmetric: back off quickly, recover slowly.
    target_duty = std::clamp(target_duty,
            st.duty - GOV_MAX_FALL_PER_S * dt,
            st.duty + GOV_MAX_RISE_PER_S * dt);

    st.duty = std::clamp(target_duty, min_duty, 1.0f);

    return st.duty;
}

int64_t common_power_rate_interval_us(
        const common_power_params & params,
        int32_t                     n_prompt_tokens,
        int32_t                     n_gen_tokens) {
    int64_t target_us = 0;

    // Scaled by the tokens actually committed, not charged once per step. With one token per
    // step the two are identical, which is why this read as `1e6f / max_gen_tps` for as long
    // as that was the only case. A speculative step commits up to n_draft+1 tokens for the
    // same single decode, and each of them is a token the caller receives, so each of them
    // has to be paid for at the ceiling's rate.
    if (params.max_gen_tps > 0.0f && n_gen_tokens > 0) {
        target_us = std::max(target_us, (int64_t) ((float) n_gen_tokens * 1e6f / params.max_gen_tps));
    }

    if (params.max_prompt_tps > 0.0f && n_prompt_tokens > 0) {
        target_us = std::max(target_us, (int64_t) ((float) n_prompt_tokens * 1e6f / params.max_prompt_tps));
    }

    return target_us;
}

struct common_power_governor {
    common_power_params params;

    std::vector<gov_device> devices;

    mutable std::mutex mutex;

    // controller state
    common_power_loop_state loop;
    int64_t                 t_last_sample_us = 0;

    // Long-window average VRAM temperature. Seeded on the first reading rather than from zero,
    // so the loop does not spend the first window believing the memory is ice cold.
    float   mem_temp_ewma    = NAN;
    int64_t t_last_ewma_us   = 0;   // whichever path last advanced the average

    // Snapshot for status reporting, refreshed on its own timer. Kept separate from
    // t_last_sample_us so that a frequent /metrics scrape cannot consume the controller's
    // sample slots and starve the loop.
    ggml_governor_telemetry last{};
    bool                    has_last = false;
    int64_t                 t_last_snapshot_us = 0;

    // rate limiter state
    int64_t t_last_step_us   = 0;
    double  throttled_seconds = 0.0;

    float min_duty = 0.15f;

    // The loop has to advance on a timer, not on traffic. Driving it from the decode path
    // alone meant a server that throttled once stayed throttled until the next request:
    // measured duty frozen at 0.67 across two minutes of idle on a 35 C card drawing 26 W,
    // recovering only when a request arrived. Thermal state keeps changing while nothing is
    // being decoded, so the controller has to keep looking.
    std::thread             ticker;
    std::condition_variable cv;
    bool                    stopping = false;
};

// Record a telemetry reading: keep it as the reportable snapshot and advance the long-window
// VRAM average. Called from both the decode path and a /metrics scrape, so the average is a
// true time-average of memory temperature rather than one biased towards whenever we happened
// to be decoding. Seeded from the first reading so it starts at the truth.
static void gov_record_sample(common_power_governor * gov,
                              const ggml_governor_telemetry & t,
                              int64_t now_us) {
    gov->last     = t;
    gov->has_last = true;

    if (!std::isnan(t.temp_mem_c)) {
        const float tau = std::max((float) gov->params.sustained_window_s, 1.0f);

        if (std::isnan(gov->mem_temp_ewma) || gov->t_last_ewma_us == 0) {
            gov->mem_temp_ewma = t.temp_mem_c;
        } else {
            const float dt = (float) (now_us - gov->t_last_ewma_us) / 1e6f;
            if (dt > 0.0f) {
                gov->mem_temp_ewma += (t.temp_mem_c - gov->mem_temp_ewma) * std::min(dt / tau, 1.0f);
            }
        }
        gov->t_last_ewma_us = now_us;
    }
}

static void gov_push_duty(common_power_governor * gov, float duty) {
    for (auto & d : gov->devices) {
        if (d.set_pace) {
            d.set_pace(d.dev, duty);
        }
    }
}

// Sample sensors and advance the controller by dt. Returns the new duty cycle.
static void gov_update_thermal(common_power_governor * gov, int64_t now_us) {
    if (!gov->params.enabled || gov->devices.empty()) {
        return;
    }

    const int64_t interval_us = (int64_t) gov->params.sample_interval_ms * 1000;
    if (gov->t_last_sample_us != 0 && now_us - gov->t_last_sample_us < interval_us) {
        return;
    }

    const float dt_elapsed = gov->t_last_sample_us == 0
        ? (float) gov->params.sample_interval_ms / 1000.0f
        : (float) (now_us - gov->t_last_sample_us) / 1e6f;
    gov->t_last_sample_us = now_us;

    // See GOV_MAX_STEP_DT_S: one step must not be handed the authority of a missed hour.
    const float dt = std::min(dt_elapsed, GOV_MAX_STEP_DT_S);

    // Worst offender governs: sample every device and every signal, take the largest
    // normalised error. One hot card in a multi-GPU box should throttle the whole process.
    // `have_err` rather than a sentinel value in `err`. An earlier version seeded err at -1.0
    // to mean "nothing read" and tested `err < -0.5` for it, which collides with reality: a
    // cold idle card produces a worst-case error near -0.8, and every signal on it is more
    // negative still. The loop therefore refused to step exactly when it should have been
    // recovering the duty cycle, and duty stayed frozen wherever the last busy period left it.
    float err      = 0.0f;
    bool  have_err = false;
    bool  any      = false;

    for (const auto & d : gov->devices) {
        ggml_governor_telemetry t;
        ggml_governor_telemetry_init(&t);
        if (!d.get_telemetry(d.dev, &t)) {
            continue;
        }

        gov_record_sample(gov, t, now_us);
        any = true;

        const float e_junction = common_power_headroom_error(t.temp_junction_c, gov->params.max_temp_c,     d.crit_junction_c);
        const float e_mem      = common_power_headroom_error(t.temp_mem_c,      gov->params.max_mem_temp_c, d.crit_mem_c);
        const float e_power    = common_power_headroom_error(t.power_w,         d.target_power_w,           d.power_ceiling_w);

        // Sustained VRAM temperature, normalised against the instantaneous ceiling rather
        // than crit: exceeding the long-run target by as much as the spike ceiling allows is
        // already a full-scale error, because this signal is about months, not seconds.
        const float e_mem_sustained = gov->params.mem_temp_sustained_c > 0.0f
            ? common_power_headroom_error(gov->mem_temp_ewma, gov->params.mem_temp_sustained_c,
                    std::max(gov->params.max_mem_temp_c, gov->params.mem_temp_sustained_c + 1.0f))
            : NAN;

        // Utilisation ceilings. Both counters cover the whole card, so this is what yields to
        // a game or a compositor: their load lands in the same number and pushes the error up.
        const float e_gpu_busy = gov->params.max_gpu_busy_pct > 0 && t.busy_pct >= 0
            ? common_power_headroom_error((float) t.busy_pct, (float) gov->params.max_gpu_busy_pct, 100.0f)
            : NAN;
        const float e_mem_busy = gov->params.max_mem_busy_pct > 0 && t.mem_busy_pct >= 0
            ? common_power_headroom_error((float) t.mem_busy_pct, (float) gov->params.max_mem_busy_pct, 100.0f)
            : NAN;

        for (const float e : { e_junction, e_mem, e_power, e_mem_sustained, e_gpu_busy, e_mem_busy }) {
            if (!std::isnan(e)) {
                err      = have_err ? std::max(err, e) : e;
                have_err = true;
            }
        }
    }

    if (!any || !have_err) {
        return;   // nothing readable this round; hold the current duty
    }

    gov_push_duty(gov, common_power_loop_step(gov->loop, err, dt, gov->min_duty));
}

// Advance the loop on a timer, whatever the traffic is doing.
//
// Runs with the governor mutex held except while waiting, which is the same discipline the
// decode and status paths use. The wait is a condition variable rather than a sleep so that
// shutdown does not have to sit through a whole interval.
static void gov_ticker(common_power_governor * gov) {
    std::unique_lock<std::mutex> lock(gov->mutex);

    while (!gov->stopping) {
        gov->cv.wait_for(lock,
                std::chrono::milliseconds(gov->params.sample_interval_ms),
                [gov] { return gov->stopping; });

        if (gov->stopping) {
            break;
        }

        gov_update_thermal(gov, ggml_time_us());
    }
}

common_power_governor_ptr common_power_governor_init(const common_power_params & params) {
    common_power_governor_ptr gov(new common_power_governor());

    gov->params   = params;
    gov->min_duty = std::clamp((float) params.min_duty_pct / 100.0f, 0.01f, 1.0f);

    if (params.sample_interval_ms <= 0) {
        gov->params.sample_interval_ms = 250;
    }

    // The explicit hwmon override is handed to ggml through the environment, which is the only
    // channel the sysfs resolver reads. Set it before any device is touched.
    if (!params.hwmon_path.empty()) {
        setenv("GGML_GOVERNOR_HWMON", params.hwmon_path.c_str(), 1);
    }

    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (reg == nullptr) {
            continue;
        }

        // Resolved by name rather than linked: a build against a ggml without governor
        // support gets nullptr here and reports the feature as unavailable, instead of
        // failing to link.
        auto get_telemetry = (ggml_backend_dev_get_telemetry_t)
            ggml_backend_reg_get_proc_address(reg, GGML_GOVERNOR_PROC_GET_TELEMETRY);
        auto set_pace = (ggml_backend_dev_set_pace_t)
            ggml_backend_reg_get_proc_address(reg, GGML_GOVERNOR_PROC_SET_PACE);

        if (get_telemetry == nullptr || set_pace == nullptr) {
            continue;
        }

        ggml_governor_telemetry t;
        ggml_governor_telemetry_init(&t);
        if (!get_telemetry(dev, &t)) {
            continue;
        }

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);

        gov_device d;
        d.dev             = dev;
        d.get_telemetry   = get_telemetry;
        d.set_pace        = set_pace;
        d.name            = ggml_backend_dev_name(dev);
        d.id              = props.device_id ? props.device_id : "unknown";
        d.crit_junction_c = t.temp_junction_crit_c;
        d.crit_mem_c      = t.temp_mem_crit_c;
        d.power_limit_w   = t.power_limit_w;

        // The power budget is a fraction of what the board ships at, and the ceiling for the
        // error normalisation is the highest limit the board would accept.
        if (params.budget_watts > 0.0f) {
            d.target_power_w = params.budget_watts;
        } else if (!std::isnan(t.power_limit_w)) {
            d.target_power_w = t.power_limit_w * (float) params.budget_pct / 100.0f;
        }

        d.power_ceiling_w = !std::isnan(t.power_limit_max_w) ? t.power_limit_max_w : t.power_limit_w;

        gov->devices.push_back(std::move(d));
    }

    if (params.enabled && gov->devices.empty()) {
        throw std::runtime_error(
            "power governor enabled but no GPU exposes usable sensors: either this ggml build "
            "has no governor support, or the device's hwmon node could not be resolved. Point "
            "--power-governor-hwmon at the correct hwmon directory, or disable the governor.");
    }

    // Only when there is something to govern. With the sensor loop off the ticker would wake
    // every interval to do nothing, and the tokens-per-second ceilings need no timer - they are
    // feed-forward and act on the decode path itself.
    if (gov->params.enabled && !gov->devices.empty()) {
        gov->ticker = std::thread(gov_ticker, gov.get());
    }

    return gov;
}

void common_power_governor_free(common_power_governor * gov) {
    if (gov == nullptr) {
        return;
    }

    // Stop the ticker before anything else: it touches every field below.
    {
        std::lock_guard<std::mutex> lock(gov->mutex);
        gov->stopping = true;
    }
    gov->cv.notify_all();
    if (gov->ticker.joinable()) {
        gov->ticker.join();
    }

    // Leave the hardware unpaced; the process may outlive the governor.
    gov_push_duty(gov, 1.0f);
    delete gov;
}

void common_power_governor_on_decode(
        common_power_governor * gov,
        int32_t                 n_prompt_tokens,
        int32_t                 n_gen_tokens,
        int64_t                 work_us) {
    if (gov == nullptr) {
        return;
    }

    int64_t sleep_us = 0;

    {
        std::lock_guard<std::mutex> lock(gov->mutex);

        const int64_t now_us = ggml_time_us();

        gov_update_thermal(gov, now_us);

        const int64_t target_us = common_power_rate_interval_us(gov->params, n_prompt_tokens, n_gen_tokens);
        if (target_us > 0) {
            // Measure from the start of this step, not from its end, so that time already
            // spent - including any sleeping ggml did to hold the thermal duty cycle - counts
            // towards the interval. This is what keeps the two setpoint sources from
            // double-counting: whichever demanded more, the other adds nothing on top.
            const int64_t step_start_us = now_us - work_us;
            const int64_t elapsed_us    = gov->t_last_step_us > 0
                ? step_start_us - gov->t_last_step_us + work_us
                : work_us;

            sleep_us = std::clamp<int64_t>(target_us - elapsed_us, 0, GOV_MAX_SLEEP_US);
        }

        gov->t_last_step_us = now_us + sleep_us;
        gov->throttled_seconds += (double) sleep_us / 1e6;
    }

    if (sleep_us > 0) {
        sleep_us_sliced(sleep_us);
    }
}

common_power_status common_power_governor_status(common_power_governor * gov) {
    common_power_status st;

    if (gov == nullptr) {
        return st;
    }

    std::lock_guard<std::mutex> lock(gov->mutex);

    // Refresh if the snapshot has gone stale. Without this the readings only advance while
    // decoding, so an idle server reports zeros for every sensor - which reads as broken
    // hardware rather than an idle one, and leaves the cumulative VRAM average frozen exactly
    // when there is nothing to make it move.
    const int64_t now_us = ggml_time_us();
    if (!gov->devices.empty() &&
        (gov->t_last_snapshot_us == 0 ||
         now_us - gov->t_last_snapshot_us >= (int64_t) gov->params.sample_interval_ms * 1000)) {
        gov->t_last_snapshot_us = now_us;

        for (const auto & d : gov->devices) {
            ggml_governor_telemetry t;
            ggml_governor_telemetry_init(&t);
            if (d.get_telemetry(d.dev, &t)) {
                gov_record_sample(gov, t, now_us);
                break;
            }
        }
    }

    st.enabled           = gov->params.enabled;
    st.rate_capped       = gov->params.max_gen_tps > 0.0f || gov->params.max_prompt_tps > 0.0f;
    st.duty              = gov->loop.duty;
    st.throttled_seconds = gov->throttled_seconds;

    st.target_temp_c     = gov->params.max_temp_c;
    st.target_mem_temp_c = gov->params.max_mem_temp_c;

    if (!gov->devices.empty()) {
        st.target_power_w = gov->devices.front().target_power_w;
    }

    st.mem_temp_sustained_c = gov->mem_temp_ewma;

    if (gov->has_last) {
        st.temp_junction_c = gov->last.temp_junction_c;
        st.temp_mem_c      = gov->last.temp_mem_c;
        st.temp_edge_c     = gov->last.temp_edge_c;
        st.temp_vrmem_c    = gov->last.temp_vrmem_c;
        st.power_w         = gov->last.power_w;
        st.busy_pct        = gov->last.busy_pct;
        st.mem_busy_pct    = gov->last.mem_busy_pct;
        st.vram_used       = gov->last.vram_used;
        st.vram_total      = gov->last.vram_total;
    }

    return st;
}

std::string common_power_governor_describe(const common_power_governor * gov) {
    if (gov == nullptr) {
        return "power governor: not initialised";
    }

    std::lock_guard<std::mutex> lock(gov->mutex);

    std::string out = "power governor: ";

    if (!gov->params.enabled) {
        out += "sensor loop off";
    } else if (gov->devices.empty()) {
        out += "sensor loop off (no devices)";
    } else {
        char buf[320];
        for (const auto & d : gov->devices) {
            snprintf(buf, sizeof(buf),
                    "%s (%s) junction<=%.0fC (crit %.0fC), mem<=%.0fC (crit %.0fC), power<=%.0fW of %.0fW; ",
                    d.name.c_str(), d.id.c_str(),
                    gov->params.max_temp_c, d.crit_junction_c,
                    gov->params.max_mem_temp_c, d.crit_mem_c,
                    d.target_power_w, d.power_limit_w);
            out += buf;
        }
        if (gov->params.mem_temp_sustained_c > 0.0f) {
            snprintf(buf, sizeof(buf), "mem<=%.0fC averaged over %ds; ",
                    gov->params.mem_temp_sustained_c, gov->params.sustained_window_s);
            out += buf;
        }
        if (gov->params.max_gpu_busy_pct > 0) {
            snprintf(buf, sizeof(buf), "gpu busy<=%d%%; ", gov->params.max_gpu_busy_pct);
            out += buf;
        }
        if (gov->params.max_mem_busy_pct > 0) {
            snprintf(buf, sizeof(buf), "mem busy<=%d%%; ", gov->params.max_mem_busy_pct);
            out += buf;
        }
        snprintf(buf, sizeof(buf), "min duty %d%%, sampling every %dms",
                gov->params.min_duty_pct, gov->params.sample_interval_ms);
        out += buf;
    }

    if (gov->params.max_gen_tps > 0.0f) {
        char buf[64];
        snprintf(buf, sizeof(buf), "; generation capped at %.1f tok/s", gov->params.max_gen_tps);
        out += buf;
    }
    if (gov->params.max_prompt_tps > 0.0f) {
        char buf[64];
        snprintf(buf, sizeof(buf), "; prompt capped at %.1f tok/s", gov->params.max_prompt_tps);
        out += buf;
    }

    return out;
}
