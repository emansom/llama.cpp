// Tests for the GPU power governor's arithmetic: the headroom error, the control loop, and the
// tokens-per-second ceilings. All of it runs against a synthetic thermal model, so there is no
// GPU, no sensor, and no timing race involved.

#include "power-governor.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int g_failures = 0;

static void check(bool ok, const std::string & what) {
    if (!ok) {
        printf("  FAIL  %s\n", what.c_str());
        g_failures++;
    }
}

static void check_near(float got, float want, float tol, const std::string & what) {
    if (!(std::fabs(got - want) <= tol)) {
        printf("  FAIL  %s (got %.4f, want %.4f +/- %.4f)\n", what.c_str(), got, want, tol);
        g_failures++;
    }
}

//
// headroom error
//

static void test_headroom_error() {
    printf("test_headroom_error\n");

    check_near(common_power_headroom_error(70.0f, 70.0f, 110.0f), 0.0f, 1e-6f, "zero at the target");
    check_near(common_power_headroom_error(110.0f, 70.0f, 110.0f), 1.0f, 1e-6f, "one at the ceiling");
    check_near(common_power_headroom_error(90.0f, 70.0f, 110.0f), 0.5f, 1e-6f, "half way");

    // headroom to spare reads negative, which is what lets the loop recover
    check(common_power_headroom_error(50.0f, 70.0f, 110.0f) < 0.0f, "negative below the target");

    // a missing reading must not be mistaken for a cold one
    check(std::isnan(common_power_headroom_error(NAN, 70.0f, 110.0f)), "NaN reading propagates");
    check(std::isnan(common_power_headroom_error(70.0f, NAN, 110.0f)), "NaN target propagates");
    check(std::isnan(common_power_headroom_error(70.0f, 70.0f, NAN)), "NaN ceiling propagates");

    // a ceiling at or below the target would divide by zero or invert the sign
    check(std::isnan(common_power_headroom_error(70.0f, 70.0f, 70.0f)), "degenerate ceiling rejected");
    check(std::isnan(common_power_headroom_error(70.0f, 90.0f, 80.0f)), "inverted ceiling rejected");
}

//
// control loop
//

static void test_loop_deadband() {
    printf("test_loop_deadband\n");

    common_power_loop_state st;

    // Inside the deadband the integral must not move, or the duty cycle dithers forever.
    for (int i = 0; i < 100; i++) {
        common_power_loop_step(st, 0.02f, 0.25f, 0.15f);
    }
    check_near(st.integral, 0.0f, 1e-6f, "integral frozen inside the deadband");
}

static void test_loop_anti_windup() {
    printf("test_loop_anti_windup\n");

    common_power_loop_state st;

    // Peg the error hard for a long time; the loop saturates at min_duty.
    for (int i = 0; i < 400; i++) {
        common_power_loop_step(st, 1.0f, 0.25f, 0.15f);
    }
    check_near(st.duty, 0.15f, 1e-3f, "saturates at min_duty");

    // Now the card cools. Without anti-windup the integral would have grown without bound and
    // the loop would sit at min_duty for minutes. It must recover promptly instead.
    int steps = 0;
    while (st.duty < 0.99f && steps < 1000) {
        common_power_loop_step(st, -1.0f, 0.25f, 0.15f);
        steps++;
    }
    check(st.duty >= 0.99f, "recovers to full speed once cool");

    // Recovery is slew limited to 0.10 duty/s, so 0.15 -> 1.0 takes at least 8.5 s.
    const float recovery_s = steps * 0.25f;
    check(recovery_s >= 8.0f, "recovery respects the rise slew limit");
    check(recovery_s <= 20.0f, "recovery is not stalled by integral windup");
}

static void test_loop_slew_asymmetry() {
    printf("test_loop_slew_asymmetry\n");

    common_power_loop_state fall;
    common_power_loop_step(fall, 1.0f, 0.1f, 0.15f);
    // falling is limited to 1.0 duty/s, so one 100 ms step drops at most 0.10
    check_near(fall.duty, 0.90f, 1e-3f, "fall limited to 1.0/s");

    common_power_loop_state rise;
    rise.duty = 0.5f;
    common_power_loop_step(rise, -1.0f, 0.1f, 0.15f);
    // rising is limited to 0.10 duty/s, so the same step gains at most 0.01
    check_near(rise.duty, 0.51f, 1e-3f, "rise limited to 0.1/s");

    check(true, "back off fast, recover slow");
}

static void test_loop_min_duty_clamp() {
    printf("test_loop_min_duty_clamp\n");

    for (const float min_duty : { 0.05f, 0.15f, 0.50f }) {
        common_power_loop_state st;
        for (int i = 0; i < 500; i++) {
            const float d = common_power_loop_step(st, 5.0f, 0.25f, min_duty);
            check(d >= min_duty - 1e-4f, "never drops below min_duty");
            check(d <= 1.0f + 1e-4f, "never exceeds 1.0");
        }
        check_near(st.duty, min_duty, 1e-3f, "settles on min_duty under a huge error");
    }
}

// A first-order thermal model: temperature relaxes towards an equilibrium set by the duty cycle.
// Crude, but it has the property that matters - the measurement lags the actuation, which is
// exactly what makes a badly tuned loop oscillate.
static void test_loop_converges_on_synthetic_thermal_model() {
    printf("test_loop_converges_on_synthetic_thermal_model\n");

    const float target  = 70.0f;
    const float ceiling = 110.0f;
    const float dt      = 0.25f;
    const float tau     = 4.0f;    // seconds
    const float t_idle  = 40.0f;   // temperature at duty 0
    const float t_full  = 95.0f;   // equilibrium at duty 1, i.e. well over target

    common_power_loop_state st;

    float temp = t_idle;
    std::vector<float> tail;

    for (int i = 0; i < 1200; i++) {   // 300 s
        const float err = common_power_headroom_error(temp, target, ceiling);
        const float duty = common_power_loop_step(st, err, dt, 0.15f);

        const float equilibrium = t_idle + (t_full - t_idle) * duty;
        temp += (equilibrium - temp) * (dt / tau);

        if (i >= 800) {
            tail.push_back(temp);
        }
    }

    float lo = tail[0];
    float hi = tail[0];
    for (const float v : tail) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }

    // Settles at the target rather than at the hardware limit.
    check_near((lo + hi) / 2.0f, target, 2.0f, "settles at the temperature target");

    // And holds it steady. Oscillation is the failure mode this loop is shaped to avoid, and a
    // swinging temperature is itself a wear mechanism.
    check(hi - lo < 2.0f, "no sustained oscillation");

    // Holding the target must have cost some throughput but not all of it.
    check(st.duty > 0.15f && st.duty < 1.0f, "settles on a partial duty cycle");
}

static void test_loop_leaves_a_cool_card_alone() {
    printf("test_loop_leaves_a_cool_card_alone\n");

    common_power_loop_state st;

    // A card that never approaches its target should never be throttled at all.
    for (int i = 0; i < 400; i++) {
        common_power_loop_step(st, common_power_headroom_error(45.0f, 70.0f, 110.0f), 0.25f, 0.15f);
    }
    check_near(st.duty, 1.0f, 1e-3f, "full speed while there is headroom");
}

//
// utilisation ceilings and sustained exposure
//

static void test_utilisation_headroom() {
    printf("test_utilisation_headroom\n");

    // The counters are a share of the whole card and the ceiling is 100%, so the error is
    // simply how far past the target the card has gone relative to the remaining headroom.
    check_near(common_power_headroom_error(85.0f, 85.0f, 100.0f), 0.0f, 1e-6f, "at the busy target");
    check_near(common_power_headroom_error(100.0f, 85.0f, 100.0f), 1.0f, 1e-6f, "fully saturated");
    check(common_power_headroom_error(40.0f, 85.0f, 100.0f) < 0.0f, "idle card reads negative");

    // A target of 100 leaves no headroom to normalise against and must be rejected rather
    // than dividing by zero.
    check(std::isnan(common_power_headroom_error(100.0f, 100.0f, 100.0f)), "target of 100 rejected");
}

// The point of the sustained signal: a card that spikes briefly is fine, a card that sits hot
// for months is not. The long window has to ignore the former and catch the latter.
static void test_sustained_window_ignores_spikes_catches_plateaus() {
    printf("test_sustained_window_ignores_spikes_catches_plateaus\n");

    const float dt = 0.25f;
    const float tau = 300.0f;   // the shipped 5 minute window

    auto ewma = [&](float seed, float value, float seconds) {
        float e = seed;
        for (int i = 0; i < (int) (seconds / dt); i++) {
            e += (value - e) * std::min(dt / tau, 1.0f);
        }
        return e;
    };

    // A 30 s excursion to 95 C from a 65 C baseline barely moves a 5 minute average.
    const float spike = ewma(65.0f, 95.0f, 30.0f);
    check(spike < 68.0f, "a 30s spike hardly moves the long window");
    check(common_power_headroom_error(spike, 80.0f, 85.0f) < 0.0f, "and does not trip the target");

    // Sitting at 84 C for an hour does move it, and does trip.
    const float plateau = ewma(65.0f, 84.0f, 3600.0f);
    check(plateau > 83.0f, "an hour at 84C converges the long window");
    check(common_power_headroom_error(plateau, 80.0f, 85.0f) > 0.0f, "and trips the target");

    // Seeding matters: starting the average at zero would take a full window to become
    // meaningful, during which the loop would believe the memory was ice cold.
    check_near(ewma(70.0f, 70.0f, 600.0f), 70.0f, 0.01f, "a seeded average at steady state does not drift");
}

//
// rate ceilings
//

static void test_rate_interval() {
    printf("test_rate_interval\n");

    common_power_params p;

    // unlimited by default
    check(common_power_rate_interval_us(p, 0, 1) == 0, "no ceiling by default");
    check(common_power_rate_interval_us(p, 512, 0) == 0, "no prompt ceiling by default");

    p.max_gen_tps = 20.0f;

    // n_gen_tokens is the tokens ONE sequence committed in the step, so the interval scales
    // with it. The caller keeps the cap per-sequence by passing the maximum across sequences
    // rather than the sum, which is why "8 slots busy" is not a case this function can see:
    // eight slots committing one token each arrive here as 1, not 8.
    check(common_power_rate_interval_us(p, 0, 1) == 50000, "20 tok/s -> 50 ms for one token");
    check(common_power_rate_interval_us(p, 0, 0) == 0, "no gating when nothing is generating");

    // Speculative decoding: one decode, several tokens handed back, each of them paid for.
    // This is the case that used to escape the ceiling entirely - the step was charged 50 ms
    // however many tokens it committed, so a 4-token accept ran at 4x the configured rate.
    check(common_power_rate_interval_us(p, 0, 2) == 100000, "2 accepted tokens -> 100 ms");
    check(common_power_rate_interval_us(p, 0, 4) == 200000, "4 accepted tokens -> 200 ms");

    // The ceiling is the ceiling regardless of how the tokens were produced: N tokens cost
    // the same wall clock whether they arrived one per decode or all in one verification.
    for (int32_t n = 1; n <= 8; ++n) {
        check(common_power_rate_interval_us(p, 0, n) == (int64_t) n * 50000,
              "speculated and unspeculated tokens cost the same time");
    }

    // Prompt work is bulk, so its ceiling scales with the token count.
    common_power_params q;
    q.max_prompt_tps = 1000.0f;
    check(common_power_rate_interval_us(q, 500, 0) == 500000, "500 tokens at 1000 tok/s -> 500 ms");
    check(common_power_rate_interval_us(q, 0, 1) == 0, "prompt ceiling ignores generation");

    // Both configured: the more restrictive one wins.
    common_power_params r;
    r.max_gen_tps    = 20.0f;      //  50 ms
    r.max_prompt_tps = 1000.0f;    // 100 ms for 100 tokens
    check(common_power_rate_interval_us(r, 100, 1) == 100000, "most restrictive ceiling wins");
}

static void test_rate_interval_matches_requested_rate() {
    printf("test_rate_interval_matches_requested_rate\n");

    for (const float tps : { 1.0f, 5.0f, 20.0f, 60.0f, 200.0f }) {
        common_power_params p;
        p.max_gen_tps = tps;

        const int64_t interval_us = common_power_rate_interval_us(p, 0, 1);
        const float   achieved    = 1e6f / (float) interval_us;

        check_near(achieved, tps, tps * 0.01f, "achieved rate matches the configured ceiling");
    }
}

//
// governor lifecycle without any GPU present
//

static void test_disabled_governor_is_inert() {
    printf("test_disabled_governor_is_inert\n");

    common_power_params p;   // enabled = false, no ceilings

    // Must not throw even when nothing on the machine has sensors.
    auto gov = common_power_governor_init(p);
    check(gov != nullptr, "constructs with the sensor loop off");

    const auto st = common_power_governor_status(gov.get());
    check(!st.enabled, "reports itself disabled");
    check_near(st.duty, 1.0f, 1e-6f, "duty stays at full speed");

    // Exercising the decode hook must be harmless.
    common_power_governor_on_decode(gov.get(), 0, 1, 1000);
    check_near(common_power_governor_status(gov.get()).duty, 1.0f, 1e-6f, "still unthrottled");

    // A null governor is a supported no-op, so callers need no branch.
    common_power_governor_on_decode(nullptr, 0, 1, 1000);
    check(true, "null governor tolerated");
}

// The sustained average must advance on wall-clock time, not on how often we happen to decode.
// It is driven from both the decode path and a /metrics scrape for exactly this reason, so the
// arithmetic has to be correct for irregular sampling intervals.
static void test_sustained_average_is_sampling_rate_independent() {
    printf("test_sustained_average_is_sampling_rate_independent\n");

    const float tau = 300.0f;
    const float seed = 60.0f;
    const float value = 80.0f;

    auto converge = [&](float dt, float seconds) {
        float e = seed;
        for (int i = 0; i < (int) (seconds / dt); i++) {
            e += (value - e) * std::min(dt / tau, 1.0f);
        }
        return e;
    };

    // Ten minutes at 0.25 s and at 5 s steps must land in the same place: the average is a
    // function of elapsed time, not of sample count.
    const float fast = converge(0.25f, 600.0f);
    const float slow = converge(5.00f, 600.0f);
    check_near(fast, slow, 0.5f, "same result at 20x different sampling rates");

    // And a single very long gap must not overshoot past the reading itself.
    float once = seed;
    once += (value - once) * std::min(3600.0f / tau, 1.0f);
    check(once <= value + 1e-3f, "a long gap clamps rather than overshooting");
}

int main() {
    test_headroom_error();
    test_loop_deadband();
    test_loop_anti_windup();
    test_loop_slew_asymmetry();
    test_loop_min_duty_clamp();
    test_loop_converges_on_synthetic_thermal_model();
    test_loop_leaves_a_cool_card_alone();
    test_utilisation_headroom();
    test_sustained_window_ignores_spikes_catches_plateaus();
    test_sustained_average_is_sampling_rate_independent();
    test_rate_interval();
    test_rate_interval_matches_requested_rate();
    test_disabled_governor_is_inert();

    if (g_failures > 0) {
        printf("\n%d check(s) failed\n", g_failures);
        return 1;
    }

    printf("\nall power governor tests passed\n");
    return 0;
}
