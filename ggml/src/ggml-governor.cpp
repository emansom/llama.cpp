#include "ggml-governor.h"
#include "ggml-impl.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#if defined(__linux__)
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#endif

// Longest gap between two pace points that is still counted as compute. A larger gap means the
// backend was idle (waiting for a request, a host transfer), and charging a duty-cycle sleep for
// it would stall the next submission for no thermal benefit.
static const int64_t GGML_GOVERNOR_MAX_WORK_US = 100 * 1000;   // 100 ms

// Longest single sleep. Bounds the latency cost of one pace point, so cancellation and shutdown
// stay responsive even at a very low duty cycle.
static const int64_t GGML_GOVERNOR_MAX_SLEEP_US = 250 * 1000;  // 250 ms

// Sleeps are served in slices of this length. Nothing polls between slices today, but the slicing
// keeps any single nanosleep short enough that a signal is never swallowed for long.
static const int64_t GGML_GOVERNOR_SLICE_US = 25 * 1000;       // 25 ms

// Number of devices currently paced. The pace point fast path is a single relaxed load of this,
// so a build with pacing disabled pays essentially nothing.
static std::atomic<int> g_ggml_governor_paced{0};

namespace {

struct governor_device {
    // pacing state
    std::atomic<float>   duty{1.0f};
    std::atomic<int64_t> last_pace_us{0};

    // sensor state, resolved once
    std::mutex  sensor_mutex;
    bool        resolved = false;
    bool        usable   = false;

#if defined(__linux__)
    // open file descriptors, -1 when the sensor is absent
    int fd_temp_edge      = -1;
    int fd_temp_junction  = -1;
    int fd_temp_mem       = -1;
    int fd_temp_vrmem     = -1;
    int fd_power          = -1;
    int fd_busy           = -1;
    int fd_mem_busy       = -1;
    int fd_vram_used      = -1;
    int fd_vram_total     = -1;

    // static values, read once at resolution time
    float temp_edge_crit_c     = NAN;
    float temp_junction_crit_c = NAN;
    float temp_mem_crit_c      = NAN;
    float power_limit_w        = NAN;
    float power_limit_max_w    = NAN;

    std::string hwmon_path;
#endif
};

std::mutex                                            g_devices_mutex;
std::map<ggml_backend_dev_t, governor_device *>       g_devices;

governor_device * governor_get(ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_devices_mutex);

    auto it = g_devices.find(dev);
    if (it != g_devices.end()) {
        return it->second;
    }

    // Never freed. One entry per physical device for the lifetime of the process, and the
    // pace point reads it without holding g_devices_mutex.
    governor_device * gd = new governor_device();
    g_devices[dev] = gd;
    return gd;
}

#if defined(__linux__)

// Read a whole sysfs attribute. sysfs supports pread from offset 0 on an already-open fd, which
// lets the hot path avoid re-walking the directory tree on every sample.
bool read_fd_str(int fd, std::string & out) {
    if (fd < 0) {
        return false;
    }

    char buf[64];
    ssize_t n;
    do {
        n = pread(fd, buf, sizeof(buf) - 1, 0);
    } while (n < 0 && errno == EINTR);

    if (n <= 0) {
        return false;
    }

    buf[n] = '\0';
    out.assign(buf);
    return true;
}

bool read_fd_long(int fd, long & out) {
    std::string s;
    if (!read_fd_str(fd, s)) {
        return false;
    }

    errno = 0;
    char * end = nullptr;
    const long v = strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || errno != 0) {
        return false;
    }

    out = v;
    return true;
}

bool read_path_str(const std::string & path, std::string & out) {
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return false;
    }
    const bool ok = read_fd_str(fd, out);
    close(fd);
    return ok;
}

bool read_path_long(const std::string & path, long & out) {
    std::string s;
    if (!read_path_str(path, s)) {
        return false;
    }

    errno = 0;
    char * end = nullptr;
    const long v = strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || errno != 0) {
        return false;
    }

    out = v;
    return true;
}

std::string trimmed(const std::string & s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && isspace((unsigned char) s[b])) { b++; }
    while (e > b && isspace((unsigned char) s[e - 1])) { e--; }
    return s.substr(b, e - b);
}

// Locate the hwmon directory belonging to a PCI device, e.g.
//   /sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon3
std::string find_hwmon(const std::string & pci_base) {
    const std::string hwmon_root = pci_base + "/hwmon";

    DIR * d = opendir(hwmon_root.c_str());
    if (d == nullptr) {
        return "";
    }

    std::string result;
    while (const struct dirent * ent = readdir(d)) {
        if (strncmp(ent->d_name, "hwmon", 5) == 0) {
            result = hwmon_root + "/" + ent->d_name;
            break;
        }
    }
    closedir(d);

    return result;
}

// Resolve which tempN_* group corresponds to which physical sensor. The numbering is not stable
// across drivers or even across kernel versions, so it is always read from tempN_label rather
// than assumed.
void resolve_temp_sensors(governor_device * gd) {
    for (int i = 1; i <= 16; i++) {
        const std::string prefix = gd->hwmon_path + "/temp" + std::to_string(i);

        std::string label;
        if (!read_path_str(prefix + "_label", label)) {
            continue;
        }
        label = trimmed(label);

        int *   fd_slot   = nullptr;
        float * crit_slot = nullptr;

        static float ignored_crit = NAN;

        if (label == "edge") {
            fd_slot   = &gd->fd_temp_edge;
            crit_slot = &gd->temp_edge_crit_c;
        } else if (label == "junction" || label == "hotspot") {
            fd_slot   = &gd->fd_temp_junction;
            crit_slot = &gd->temp_junction_crit_c;
        } else if (label == "mem" || label == "vram") {
            fd_slot   = &gd->fd_temp_mem;
            crit_slot = &gd->temp_mem_crit_c;
        } else if (label == "vrmem") {
            // The VRAM voltage regulator. amdgpu exposes no _crit for it, so it is reported
            // but not governed - there is no hardware-declared ceiling to normalise against.
            fd_slot   = &gd->fd_temp_vrmem;
            crit_slot = &ignored_crit;
        } else {
            continue;
        }

        if (*fd_slot >= 0) {
            continue; // already resolved by an earlier index
        }

        const int fd = open((prefix + "_input").c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            continue;
        }
        *fd_slot = fd;

        long crit = 0;
        if (read_path_long(prefix + "_crit", crit)) {
            *crit_slot = crit / 1000.0f;
        }
    }
}

void resolve_power_sensors(governor_device * gd) {
    // power1_average is the board power (PPT) on amdgpu; power1_input is the equivalent on some
    // other drivers. Prefer the averaged reading, which is what the board regulates against.
    for (const char * name : { "/power1_average", "/power1_input" }) {
        const int fd = open((gd->hwmon_path + name).c_str(), O_RDONLY | O_CLOEXEC);
        if (fd >= 0) {
            gd->fd_power = fd;
            break;
        }
    }

    long cap = 0;
    if (read_path_long(gd->hwmon_path + "/power1_cap_default", cap)) {
        gd->power_limit_w = cap / 1000000.0f;
    }
    if (read_path_long(gd->hwmon_path + "/power1_cap_max", cap)) {
        gd->power_limit_max_w = cap / 1000000.0f;
    }
}

void resolve_sensors(governor_device * gd, ggml_backend_dev_t dev) {
    if (gd->resolved) {
        return;
    }
    gd->resolved = true;

    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);

    // An explicit override wins over discovery, and is honoured even for a device that reports
    // no PCI id - the whole point of the override is to cover the cases discovery cannot reach.
    const char * override_path = getenv("GGML_GOVERNOR_HWMON");

    if (override_path != nullptr && override_path[0] != '\0') {
        gd->hwmon_path = override_path;
    } else {
        if (props.device_id == nullptr) {
            GGML_LOG_DEBUG("%s: device '%s' reports no PCI id, no sensors available\n",
                    __func__, ggml_backend_dev_name(dev));
            return;
        }

        const std::string pci_base = std::string("/sys/bus/pci/devices/") + props.device_id;
        gd->hwmon_path = find_hwmon(pci_base);

        // Utilisation and capacity hang off the PCI node rather than hwmon. They are
        // system-wide counters, not per-process, which is what makes the busy figures useful
        // for yielding to whatever else is on the card.
        struct { const char * name; int * fd; } extras[] = {
            { "/gpu_busy_percent",    &gd->fd_busy       },
            { "/mem_busy_percent",    &gd->fd_mem_busy   },
            { "/mem_info_vram_used",  &gd->fd_vram_used  },
            { "/mem_info_vram_total", &gd->fd_vram_total },
        };
        for (const auto & e : extras) {
            const int fd = open((pci_base + e.name).c_str(), O_RDONLY | O_CLOEXEC);
            if (fd >= 0) {
                *e.fd = fd;
            }
        }
    }

    if (gd->hwmon_path.empty()) {
        GGML_LOG_DEBUG("%s: device '%s' (%s) has no hwmon node\n",
                __func__, ggml_backend_dev_name(dev), props.device_id);
        return;
    }

    resolve_temp_sensors(gd);
    resolve_power_sensors(gd);

    gd->usable = gd->fd_temp_edge >= 0 || gd->fd_temp_junction >= 0 ||
                 gd->fd_temp_mem  >= 0 || gd->fd_power         >= 0 ||
                 gd->fd_busy      >= 0 || gd->fd_mem_busy      >= 0;

    if (!gd->usable) {
        GGML_LOG_DEBUG("%s: device '%s' (%s) exposes no usable sensors under %s\n",
                __func__, ggml_backend_dev_name(dev), props.device_id, gd->hwmon_path.c_str());
        return;
    }

    GGML_LOG_INFO("%s: device '%s' (%s) sensors at %s:%s%s%s%s%s%s%s%s\n",
            __func__, ggml_backend_dev_name(dev), props.device_id, gd->hwmon_path.c_str(),
            gd->fd_temp_edge     >= 0 ? " edge"     : "",
            gd->fd_temp_junction >= 0 ? " junction" : "",
            gd->fd_temp_mem      >= 0 ? " mem"      : "",
            gd->fd_temp_vrmem    >= 0 ? " vrmem"    : "",
            gd->fd_power         >= 0 ? " power"    : "",
            gd->fd_busy          >= 0 ? " busy"     : "",
            gd->fd_mem_busy      >= 0 ? " mem-busy" : "",
            gd->fd_vram_used     >= 0 ? " vram"     : "");
}

float read_temp_c(int fd) {
    long v = 0;
    if (!read_fd_long(fd, v)) {
        return NAN;
    }
    return v / 1000.0f;
}

void sleep_us_sliced(int64_t total_us) {
    while (total_us > 0) {
        const int64_t slice = total_us < GGML_GOVERNOR_SLICE_US ? total_us : GGML_GOVERNOR_SLICE_US;

        struct timespec ts;
        ts.tv_sec  = (time_t) (slice / 1000000);
        ts.tv_nsec = (long)   ((slice % 1000000) * 1000);

        // Deliberately not resumed on EINTR: an interrupt should cut through a throttling
        // sleep rather than queue behind it.
        if (nanosleep(&ts, nullptr) != 0) {
            return;
        }

        total_us -= slice;
    }
}

#endif // __linux__

} // namespace

bool ggml_backend_dev_get_telemetry(ggml_backend_dev_t dev, struct ggml_governor_telemetry * out) {
    if (out == nullptr || out->size < sizeof(uint32_t)) {
        return false;
    }

#if defined(__linux__)
    governor_device * gd = governor_get(dev);
    if (gd == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(gd->sensor_mutex);

    resolve_sensors(gd, dev);
    if (!gd->usable) {
        return false;
    }

    // Fill a full struct, then copy back only as much as the caller declared room for. A
    // caller built against an older header gets the prefix it understands and nothing else.
    ggml_governor_telemetry t;
    ggml_governor_telemetry_init(&t);

    t.temp_edge_c          = read_temp_c(gd->fd_temp_edge);
    t.temp_junction_c      = read_temp_c(gd->fd_temp_junction);
    t.temp_mem_c           = read_temp_c(gd->fd_temp_mem);
    t.temp_vrmem_c         = read_temp_c(gd->fd_temp_vrmem);
    t.temp_edge_crit_c     = gd->temp_edge_crit_c;
    t.temp_junction_crit_c = gd->temp_junction_crit_c;
    t.temp_mem_crit_c      = gd->temp_mem_crit_c;
    t.power_limit_w        = gd->power_limit_w;
    t.power_limit_max_w    = gd->power_limit_max_w;

    long v = 0;
    t.power_w      = read_fd_long(gd->fd_power,     v) ? v / 1000000.0f : NAN;
    t.busy_pct     = read_fd_long(gd->fd_busy,      v) ? (int32_t) v : -1;
    t.mem_busy_pct = read_fd_long(gd->fd_mem_busy,  v) ? (int32_t) v : -1;
    t.vram_used    = read_fd_long(gd->fd_vram_used,  v) ? (uint64_t) v : 0;
    t.vram_total   = read_fd_long(gd->fd_vram_total, v) ? (uint64_t) v : 0;

    const uint32_t want = out->size;
    memcpy(out, &t, want < sizeof(t) ? want : sizeof(t));
    out->size = want;

    return true;
#else
    GGML_UNUSED(dev);
    return false;
#endif
}

void ggml_backend_dev_set_pace(ggml_backend_dev_t dev, float duty) {
    governor_device * gd = governor_get(dev);
    if (gd == nullptr) {
        return;
    }

    if (!(duty > 0.0f)) {      // also catches NaN
        duty = 1.0f;
    }
    if (duty > 1.0f) {
        duty = 1.0f;
    }

    const float prev = gd->duty.exchange(duty, std::memory_order_relaxed);

    const bool was_pacing = prev < 1.0f;
    const bool is_pacing  = duty < 1.0f;

    if (was_pacing != is_pacing) {
        g_ggml_governor_paced.fetch_add(is_pacing ? 1 : -1, std::memory_order_release);
    }

    if (!was_pacing && is_pacing) {
        // Starting fresh: do not charge for whatever happened before pacing was enabled.
        gd->last_pace_us.store(0, std::memory_order_relaxed);
    }
}

bool ggml_governor_pace_active(ggml_backend_dev_t dev) {
    if (g_ggml_governor_paced.load(std::memory_order_acquire) == 0) {
        return false;
    }

    governor_device * gd = governor_get(dev);
    return gd != nullptr && gd->duty.load(std::memory_order_relaxed) < 1.0f;
}

void ggml_governor_pace_reset(ggml_backend_dev_t dev) {
    if (g_ggml_governor_paced.load(std::memory_order_acquire) == 0) {
        return;
    }

    governor_device * gd = governor_get(dev);
    if (gd != nullptr) {
        gd->last_pace_us.store(0, std::memory_order_relaxed);
    }
}

void ggml_governor_pace_point(ggml_backend_dev_t dev) {
#if defined(__linux__)
    // Fast path: one relaxed load when nothing is being paced.
    if (g_ggml_governor_paced.load(std::memory_order_acquire) == 0) {
        return;
    }

    governor_device * gd = governor_get(dev);
    if (gd == nullptr) {
        return;
    }

    const float duty = gd->duty.load(std::memory_order_relaxed);
    if (duty >= 1.0f) {
        return;
    }

    const int64_t now  = ggml_time_us();
    const int64_t last = gd->last_pace_us.exchange(now, std::memory_order_relaxed);

    if (last == 0) {
        return; // first pace point since pacing began, nothing measured yet
    }

    int64_t work_us = now - last;
    if (work_us <= 0) {
        return;
    }
    if (work_us > GGML_GOVERNOR_MAX_WORK_US) {
        work_us = GGML_GOVERNOR_MAX_WORK_US;
    }

    // Hold a duty cycle of `duty`: for every unit of work, idle (1/duty - 1) units.
    int64_t sleep_us = (int64_t) ((double) work_us * (1.0 / (double) duty - 1.0));
    if (sleep_us <= 0) {
        return;
    }
    if (sleep_us > GGML_GOVERNOR_MAX_SLEEP_US) {
        sleep_us = GGML_GOVERNOR_MAX_SLEEP_US;
    }

    sleep_us_sliced(sleep_us);

    // Exclude the sleep itself from the next interval's work measurement.
    gd->last_pace_us.store(ggml_time_us(), std::memory_order_relaxed);
#else
    GGML_UNUSED(dev);
#endif
}
