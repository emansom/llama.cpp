// Tests that each per-format writer renders prompts byte-for-byte identical
// to the Jinja path (the "golden" path until Jinja is removed in Package F).
//
// For a given canonical message list + flags, we render the prompt two ways:
//   1. Via the existing Jinja path (`common_chat_templates_apply`).
//   2. Via the per-format writer (`common_chat_<format>_render`).
// and assert byte equality.

#include "chat.h"
#include "chat-formats/deepseek-v3-2-format.h"
#include "chat-formats/functionary-v3-2-format.h"
#include "chat-formats/gemma4-format.h"
#include "chat-formats/gigachat-v3-format.h"
#include "chat-formats/glm-4-7-flash-format.h"
#include "chat-formats/gpt-oss-format.h"
#include "chat-formats/hermes-format.h"
#include "chat-formats/qwq-format.h"
#include "chat-formats/kimi-k2-format.h"
#include "chat-formats/lfm2-5-format.h"
#include "chat-formats/lfm2-format.h"
#include "chat-formats/ministral-3-format.h"
#include "testing.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

using ordered_json = nlohmann::ordered_json;

// Reads a file's contents into a string. Used to load Jinja template sources.
std::string read_file(const std::string & path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        return {};
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Reads a Jinja template file, parses it, and returns a chat-templates handle
// suitable for `common_chat_templates_apply`.
common_chat_templates_ptr load_template(const std::string & path) {
    std::string source = read_file(path);
    if (source.empty()) {
        return nullptr;
    }
    return common_chat_templates_init(/*model=*/nullptr,
                                      source,
                                      /*bos_token_override=*/"",
                                      /*eos_token_override=*/"");
}

// Builds `common_chat_templates_inputs` matching `inputs`. Used to drive the
// Jinja path. The two structs have the same semantic fields but slightly
// different shapes (templates_inputs uses common_chat_msg, generation_params
// uses raw JSON).
common_chat_templates_inputs to_templates_inputs(const ordered_json & messages,
                                                 const ordered_json & tools,
                                                 bool add_generation_prompt,
                                                 bool enable_thinking) {
    common_chat_templates_inputs ti;
    ti.messages              = common_chat_msgs_parse_oaicompat(messages);
    ti.tools                 = common_chat_tools_parse_oaicompat(tools);
    ti.add_generation_prompt = add_generation_prompt;
    ti.enable_thinking       = enable_thinking;
    ti.use_jinja             = true;
    return ti;
}

// Builds `autoparser::generation_params` matching the same canonical input,
// for direct invocation of the per-format writer.
autoparser::generation_params to_generation_params(const ordered_json & messages,
                                                    const ordered_json & tools,
                                                    bool add_generation_prompt,
                                                    bool enable_thinking) {
    autoparser::generation_params gp;
    gp.messages              = messages;
    gp.tools                 = tools;
    gp.add_generation_prompt = add_generation_prompt;
    gp.enable_thinking       = enable_thinking;
    return gp;
}

// Renders the prompt via the Jinja path. Returns empty string on failure.
std::string render_via_jinja(common_chat_templates * tmpls,
                             const ordered_json & messages,
                             const ordered_json & tools,
                             bool add_generation_prompt,
                             bool enable_thinking) {
    auto ti     = to_templates_inputs(messages, tools, add_generation_prompt, enable_thinking);
    auto params = common_chat_templates_apply(tmpls, ti);
    return params.prompt;
}

}  // namespace

// ──────────────────────────────────────────────────────────────────────────────
// Kimi-K2 writer parity tests
// ──────────────────────────────────────────────────────────────────────────────

static void test_kimi_k2_render(testing & t) {
    auto tmpls = load_template("models/templates/moonshotai-Kimi-K2.jinja");
    t.assert_true("Kimi-K2 template loads", tmpls != nullptr);
    if (!tmpls) {
        return;
    }

    auto check = [&](const std::string & name,
                     const ordered_json & messages,
                     const ordered_json & tools,
                     bool add_generation_prompt) {
        const std::string jinja_out = render_via_jinja(
            tmpls.get(), messages, tools, add_generation_prompt, /*enable_thinking=*/true);
        auto gp = to_generation_params(messages, tools, add_generation_prompt, /*enable_thinking=*/true);
        const std::string writer_out = common_chat_kimi_k2_render(gp);
        if (jinja_out == writer_out) {
            t.assert_true(name + " bytes match", true);
        } else {
            // On mismatch, dump both sides for diagnosis.
            std::cerr << "\n=== Kimi-K2 mismatch: " << name << " ===\n";
            std::cerr << "Jinja  bytes: " << jinja_out  << "\n";
            std::cerr << "Writer bytes: " << writer_out << "\n";
            t.assert_true(name + " bytes match", false);
        }
    };

    t.test("user-only with generation prompt", [&](testing & t) {
        (void) t;
        check("user_only",
              ordered_json::array({ {{"role", "user"}, {"content", "Hello"}} }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("system + user", [&](testing & t) {
        (void) t;
        check("system_user",
              ordered_json::array({
                  {{"role", "system"}, {"content", "You are X"}},
                  {{"role", "user"},   {"content", "Hi"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("multi-turn user/assistant", [&](testing & t) {
        (void) t;
        check("multi_turn",
              ordered_json::array({
                  {{"role", "user"},      {"content", "Hello"}},
                  {{"role", "assistant"}, {"content", "Hi there"}},
                  {{"role", "user"},      {"content", "Goodbye"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("assistant with tool calls", [&](testing & t) {
        (void) t;
        ordered_json tool_call = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"arguments", {{"city", "Paris"}}},
            }},
        };
        check("tool_call",
              ordered_json::array({
                  {{"role", "user"}, {"content", "What's the weather?"}},
                  {{"role", "assistant"}, {"content", ""}, {"tool_calls", ordered_json::array({tool_call})}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/false);
    });

    t.test("tool result", [&](testing & t) {
        (void) t;
        ordered_json tool_call = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"arguments", {{"city", "Paris"}}},
            }},
        };
        check("tool_result",
              ordered_json::array({
                  {{"role", "user"}, {"content", "What's the weather?"}},
                  {{"role", "assistant"}, {"content", ""}, {"tool_calls", ordered_json::array({tool_call})}},
                  {{"role", "tool"}, {"tool_call_id", "call_0"}, {"content", "{\"temp\":18}"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("with tools declaration", [&](testing & t) {
        (void) t;
        ordered_json tool = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"description", "Get the weather"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {{"city", {{"type", "string"}}}}},
                    {"required", ordered_json::array({"city"})},
                }},
            }},
        };
        check("with_tools",
              ordered_json::array({{{"role", "user"}, {"content", "Hi"}}}),
              ordered_json::array({tool}),
              /*add_generation_prompt=*/true);
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// Functionary v3.2 writer parity tests
// ──────────────────────────────────────────────────────────────────────────────

static void test_functionary_v3_2_render(testing & t) {
    auto tmpls = load_template("models/templates/meetkai-functionary-medium-v3.2.jinja");
    t.assert_true("Functionary v3.2 template loads", tmpls != nullptr);
    if (!tmpls) {
        return;
    }

    auto check = [&](const std::string & name,
                     const ordered_json & messages,
                     const ordered_json & tools,
                     bool add_generation_prompt) {
        const std::string jinja_out = render_via_jinja(
            tmpls.get(), messages, tools, add_generation_prompt, /*enable_thinking=*/true);
        auto gp = to_generation_params(messages, tools, add_generation_prompt, /*enable_thinking=*/true);
        // Functionary's template emits `bos_token`. The Jinja path resolves it
        // from the loaded template metadata; we have no direct accessor here,
        // so we extract the BOS by stripping the writer-side header off the
        // Jinja output's prefix. For these tests we use empty BOS — both
        // sides receive empty consistently because load_template above does
        // not pass a tokenizer-derived BOS.
        const std::string bos_token;
        const std::string writer_out = common_chat_functionary_v3_2_render(gp, bos_token);
        if (jinja_out == writer_out) {
            t.assert_true(name + " bytes match", true);
        } else {
            std::cerr << "\n=== Functionary v3.2 mismatch: " << name << " ===\n";
            std::cerr << "Jinja  bytes: " << jinja_out  << "\n";
            std::cerr << "Writer bytes: " << writer_out << "\n";
            t.assert_true(name + " bytes match", false);
        }
    };

    t.test("user-only no tools", [&](testing & t) {
        (void) t;
        check("user_only",
              ordered_json::array({{{"role", "user"}, {"content", "Hello"}}}),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("system + user", [&](testing & t) {
        (void) t;
        check("system_user",
              ordered_json::array({
                  {{"role", "system"}, {"content", "You are X"}},
                  {{"role", "user"},   {"content", "Hi"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("multi-turn user/assistant", [&](testing & t) {
        (void) t;
        check("multi_turn",
              ordered_json::array({
                  {{"role", "user"},      {"content", "Hello"}},
                  {{"role", "assistant"}, {"content", "Hi there"}},
                  {{"role", "user"},      {"content", "Goodbye"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("with single-arg function tool", [&](testing & t) {
        (void) t;
        ordered_json tool = {
            {"type", "function"},
            {"function", {
                {"name", "special_function"},
                {"description", "I'm special"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {{"arg1", {{"type", "integer"}, {"description", "The arg."}}}}},
                    {"required", ordered_json::array({"arg1"})},
                }},
            }},
        };
        check("with_function",
              ordered_json::array({{{"role", "user"}, {"content", "Hi"}}}),
              ordered_json::array({tool}),
              /*add_generation_prompt=*/true);
    });

    t.test("assistant tool call + tool result", [&](testing & t) {
        (void) t;
        ordered_json tool_call = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"arguments", "{\"city\":\"Paris\"}"},  // string args
            }},
        };
        check("tool_cycle",
              ordered_json::array({
                  {{"role", "user"},      {"content", "What's the weather?"}},
                  {{"role", "assistant"}, {"content", ""}, {"tool_calls", ordered_json::array({tool_call})}},
                  {{"role", "tool"},      {"tool_call_id", "call_0"}, {"content", "{\"temp\":18}"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// GigaChat v3 writer parity tests (older `function call` separate-turn variant)
// ──────────────────────────────────────────────────────────────────────────────

static void test_gigachat_v3_render(testing & t) {
    auto tmpls = load_template("models/templates/GigaChat3-10B-A1.8B.jinja");
    t.assert_true("GigaChat v3 template loads", tmpls != nullptr);
    if (!tmpls) {
        return;
    }

    auto check = [&](const std::string & name,
                     const ordered_json & messages,
                     const ordered_json & tools,
                     bool add_generation_prompt) {
        const std::string jinja_out = render_via_jinja(
            tmpls.get(), messages, tools, add_generation_prompt, /*enable_thinking=*/true);
        auto gp = to_generation_params(messages, tools, add_generation_prompt, /*enable_thinking=*/true);
        const std::string bos_token;  // template emits bos_token verbatim; both sides empty.
        const std::string writer_out = common_chat_gigachat_v3_render(gp, bos_token);
        if (jinja_out == writer_out) {
            t.assert_true(name + " bytes match", true);
        } else {
            std::cerr << "\n=== GigaChat v3 mismatch: " << name << " ===\n";
            std::cerr << "Jinja  bytes (" << jinja_out.size() << "): " << jinja_out  << "\n";
            std::cerr << "Writer bytes (" << writer_out.size() << "): " << writer_out << "\n";
            t.assert_true(name + " bytes match", false);
        }
    };

    t.test("user-only no tools", [&](testing & t) {
        (void) t;
        check("user_only",
              ordered_json::array({{{"role", "user"}, {"content", "Hello"}}}),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("system + user", [&](testing & t) {
        (void) t;
        check("system_user",
              ordered_json::array({
                  {{"role", "system"}, {"content", "You are X"}},
                  {{"role", "user"},   {"content", "Hi"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("multi-turn user/assistant", [&](testing & t) {
        (void) t;
        check("multi_turn",
              ordered_json::array({
                  {{"role", "user"},      {"content", "Hello"}},
                  {{"role", "assistant"}, {"content", "Hi there"}},
                  {{"role", "user"},      {"content", "Goodbye"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("with single-arg function tool", [&](testing & t) {
        (void) t;
        ordered_json tool = {
            {"type", "function"},
            {"function", {
                {"name", "special_function"},
                {"description", "I'm special"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {{"arg1", {{"type", "integer"}, {"description", "The arg."}}}}},
                    {"required", ordered_json::array({"arg1"})},
                }},
            }},
        };
        check("with_function",
              ordered_json::array({{{"role", "user"}, {"content", "Hi"}}}),
              ordered_json::array({tool}),
              /*add_generation_prompt=*/true);
    });

    t.test("assistant tool call + tool result", [&](testing & t) {
        (void) t;
        ordered_json tool_call = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"arguments", {{"city", "Paris"}}},
            }},
        };
        check("tool_cycle",
              ordered_json::array({
                  {{"role", "user"},      {"content", "What's the weather?"}},
                  {{"role", "assistant"}, {"content", ""}, {"tool_calls", ordered_json::array({tool_call})}},
                  {{"role", "tool"},      {"tool_call_id", "call_0"}, {"content", "{\"temp\":18}"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// GLM-4.7-Flash writer parity tests
// ──────────────────────────────────────────────────────────────────────────────

static void test_glm_4_7_flash_render(testing & t) {
    auto tmpls = load_template("models/templates/GLM-4.7-Flash.jinja");
    t.assert_true("GLM-4.7-Flash template loads", tmpls != nullptr);
    if (!tmpls) {
        return;
    }

    auto check = [&](const std::string & name,
                     const ordered_json & messages,
                     const ordered_json & tools,
                     bool add_generation_prompt,
                     bool enable_thinking) {
        const std::string jinja_out = render_via_jinja(
            tmpls.get(), messages, tools, add_generation_prompt, enable_thinking);
        auto gp = to_generation_params(messages, tools, add_generation_prompt, enable_thinking);
        const std::string writer_out = common_chat_glm_4_7_flash_render(gp);
        if (jinja_out == writer_out) {
            t.assert_true(name + " bytes match", true);
        } else {
            std::cerr << "\n=== GLM-4.7-Flash mismatch: " << name << " ===\n";
            std::cerr << "Jinja  bytes (" << jinja_out.size() << "): " << jinja_out  << "\n";
            std::cerr << "Writer bytes (" << writer_out.size() << "): " << writer_out << "\n";
            t.assert_true(name + " bytes match", false);
        }
    };

    t.test("user-only thinking on", [&](testing & t) {
        (void) t;
        check("user_only_think",
              ordered_json::array({{{"role", "user"}, {"content", "Hello"}}}),
              ordered_json::array(),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/true);
    });

    t.test("user-only thinking off", [&](testing & t) {
        (void) t;
        check("user_only_no_think",
              ordered_json::array({{{"role", "user"}, {"content", "Hello"}}}),
              ordered_json::array(),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/false);
    });

    t.test("system + user", [&](testing & t) {
        (void) t;
        check("system_user",
              ordered_json::array({
                  {{"role", "system"}, {"content", "You are X"}},
                  {{"role", "user"},   {"content", "Hi"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/true);
    });

    t.test("multi-turn", [&](testing & t) {
        (void) t;
        check("multi_turn",
              ordered_json::array({
                  {{"role", "user"},      {"content", "Hello"}},
                  {{"role", "assistant"}, {"content", "Hi there"}},
                  {{"role", "user"},      {"content", "Goodbye"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/true);
    });

    t.test("with weather tool", [&](testing & t) {
        (void) t;
        ordered_json tool = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"description", "Get the weather"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {{"city", {{"type", "string"}}}}},
                    {"required", ordered_json::array({"city"})},
                }},
            }},
        };
        check("with_tool",
              ordered_json::array({{{"role", "user"}, {"content", "Hi"}}}),
              ordered_json::array({tool}),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/true);
    });

    t.test("assistant with tool calls", [&](testing & t) {
        (void) t;
        ordered_json tool = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"description", "Get the weather"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {{"city", {{"type", "string"}}}}},
                    {"required", ordered_json::array({"city"})},
                }},
            }},
        };
        ordered_json tool_call = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"arguments", {{"city", "Paris"}}},
            }},
        };
        check("assistant_tool_call",
              ordered_json::array({
                  {{"role", "user"},      {"content", "What's the weather?"}},
                  {{"role", "assistant"}, {"content", ""}, {"tool_calls", ordered_json::array({tool_call})}},
              }),
              ordered_json::array({tool}),
              /*add_generation_prompt=*/false,
              /*enable_thinking=*/true);
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// LFM2 writer parity tests
// ──────────────────────────────────────────────────────────────────────────────

static void test_lfm2_render(testing & t) {
    auto tmpls = load_template("models/templates/LFM2-8B-A1B.jinja");
    t.assert_true("LFM2 template loads", tmpls != nullptr);
    if (!tmpls) {
        return;
    }

    auto check = [&](const std::string & name,
                     const ordered_json & messages,
                     const ordered_json & tools,
                     bool add_generation_prompt) {
        const std::string jinja_out = render_via_jinja(
            tmpls.get(), messages, tools, add_generation_prompt, /*enable_thinking=*/true);
        auto gp = to_generation_params(messages, tools, add_generation_prompt, /*enable_thinking=*/true);
        const std::string bos_token;  // template emits bos_token verbatim; both sides empty.
        const std::string writer_out = common_chat_lfm2_render(gp, bos_token);
        if (jinja_out == writer_out) {
            t.assert_true(name + " bytes match", true);
        } else {
            std::cerr << "\n=== LFM2 mismatch: " << name << " ===\n";
            std::cerr << "Jinja  bytes (" << jinja_out.size() << "): " << jinja_out  << "\n";
            std::cerr << "Writer bytes (" << writer_out.size() << "): " << writer_out << "\n";
            t.assert_true(name + " bytes match", false);
        }
    };

    t.test("user-only no tools", [&](testing & t) {
        (void) t;
        check("user_only",
              ordered_json::array({{{"role", "user"}, {"content", "Hello"}}}),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("system + user", [&](testing & t) {
        (void) t;
        check("system_user",
              ordered_json::array({
                  {{"role", "system"}, {"content", "You are X"}},
                  {{"role", "user"},   {"content", "Hi"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("multi-turn", [&](testing & t) {
        (void) t;
        check("multi_turn",
              ordered_json::array({
                  {{"role", "user"},      {"content", "Hello"}},
                  {{"role", "assistant"}, {"content", "Hi there"}},
                  {{"role", "user"},      {"content", "Goodbye"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("with tools (no system)", [&](testing & t) {
        (void) t;
        ordered_json tool = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"description", "Get the weather"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {{"city", {{"type", "string"}}}}},
                    {"required", ordered_json::array({"city"})},
                }},
            }},
        };
        check("with_tools_no_sys",
              ordered_json::array({{{"role", "user"}, {"content", "Hi"}}}),
              ordered_json::array({tool}),
              /*add_generation_prompt=*/true);
    });

    t.test("with tools + system", [&](testing & t) {
        (void) t;
        ordered_json tool = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"description", "Get the weather"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {{"city", {{"type", "string"}}}}},
                    {"required", ordered_json::array({"city"})},
                }},
            }},
        };
        check("with_tools_and_sys",
              ordered_json::array({
                  {{"role", "system"}, {"content", "You are helpful"}},
                  {{"role", "user"},   {"content", "Hi"}},
              }),
              ordered_json::array({tool}),
              /*add_generation_prompt=*/true);
    });

    t.test("tool result", [&](testing & t) {
        (void) t;
        check("tool_result",
              ordered_json::array({
                  {{"role", "user"}, {"content", "What's the weather?"}},
                  {{"role", "tool"}, {"tool_call_id", "call_0"}, {"content", "{\"temp\":18}"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// LFM2.5 writer parity tests
// ──────────────────────────────────────────────────────────────────────────────

static void test_lfm2_5_render(testing & t) {
    auto tmpls = load_template("models/templates/LFM2.5-Instruct.jinja");
    t.assert_true("LFM2.5 template loads", tmpls != nullptr);
    if (!tmpls) {
        return;
    }

    auto check = [&](const std::string & name,
                     const ordered_json & messages,
                     const ordered_json & tools,
                     bool add_generation_prompt) {
        const std::string jinja_out = render_via_jinja(
            tmpls.get(), messages, tools, add_generation_prompt, /*enable_thinking=*/true);
        auto gp = to_generation_params(messages, tools, add_generation_prompt, /*enable_thinking=*/true);
        const std::string bos_token;
        const std::string writer_out = common_chat_lfm2_5_render(gp, bos_token);
        if (jinja_out == writer_out) {
            t.assert_true(name + " bytes match", true);
        } else {
            std::cerr << "\n=== LFM2.5 mismatch: " << name << " ===\n";
            std::cerr << "Jinja  bytes (" << jinja_out.size() << "): " << jinja_out  << "\n";
            std::cerr << "Writer bytes (" << writer_out.size() << "): " << writer_out << "\n";
            t.assert_true(name + " bytes match", false);
        }
    };

    t.test("user-only no tools", [&](testing & t) {
        (void) t;
        check("user_only",
              ordered_json::array({{{"role", "user"}, {"content", "Hello"}}}),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("system + user", [&](testing & t) {
        (void) t;
        check("system_user",
              ordered_json::array({
                  {{"role", "system"}, {"content", "You are X"}},
                  {{"role", "user"},   {"content", "Hi"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("multi-turn", [&](testing & t) {
        (void) t;
        check("multi_turn",
              ordered_json::array({
                  {{"role", "user"},      {"content", "Hello"}},
                  {{"role", "assistant"}, {"content", "Hi there"}},
                  {{"role", "user"},      {"content", "Goodbye"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("with tools (no system)", [&](testing & t) {
        (void) t;
        ordered_json tool = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"description", "Get the weather"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {{"city", {{"type", "string"}}}}},
                    {"required", ordered_json::array({"city"})},
                }},
            }},
        };
        check("with_tools_no_sys",
              ordered_json::array({{{"role", "user"}, {"content", "Hi"}}}),
              ordered_json::array({tool}),
              /*add_generation_prompt=*/true);
    });

    t.test("past-assistant thinking stripped", [&](testing & t) {
        (void) t;
        // Two assistant messages: the earlier one's <think>...</think> should
        // be stripped; the latest one's reasoning is preserved.
        check("past_thinking_stripped",
              ordered_json::array({
                  {{"role", "user"},      {"content", "First"}},
                  {{"role", "assistant"}, {"content", "<think>past reasoning</think>past answer"}},
                  {{"role", "user"},      {"content", "Second"}},
                  {{"role", "assistant"}, {"content", "<think>current reasoning</think>current answer"}},
                  {{"role", "user"},      {"content", "Third"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// Ministral-3 writer parity tests
// ──────────────────────────────────────────────────────────────────────────────

static void test_ministral_3_render(testing & t) {
    auto tmpls = load_template("models/templates/mistralai-Ministral-3-14B-Reasoning-2512.jinja");
    t.assert_true("Ministral-3 template loads", tmpls != nullptr);
    if (!tmpls) {
        return;
    }

    auto check = [&](const std::string & name,
                     const ordered_json & messages,
                     const ordered_json & tools,
                     bool add_generation_prompt) {
        const std::string jinja_out = render_via_jinja(
            tmpls.get(), messages, tools, add_generation_prompt, /*enable_thinking=*/true);
        auto gp = to_generation_params(messages, tools, add_generation_prompt, /*enable_thinking=*/true);
        // Both bos and eos pass empty here: load_template() doesn't supply a
        // tokenizer, so the Jinja path renders an empty `bos_token`/`eos_token`.
        // Production callers pass `tmpl.bos_token()` / `tmpl.eos_token()`.
        const std::string bos_token;
        const std::string eos_token;
        const std::string writer_out = common_chat_ministral_3_render(gp, bos_token, eos_token);
        if (jinja_out == writer_out) {
            t.assert_true(name + " bytes match", true);
        } else {
            std::cerr << "\n=== Ministral-3 mismatch: " << name << " ===\n";
            std::cerr << "Jinja  bytes (" << jinja_out.size() << "): " << jinja_out  << "\n";
            std::cerr << "Writer bytes (" << writer_out.size() << "): " << writer_out << "\n";
            t.assert_true(name + " bytes match", false);
        }
    };

    t.test("user-only no tools", [&](testing & t) {
        (void) t;
        check("user_only",
              ordered_json::array({{{"role", "user"}, {"content", "Hello"}}}),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("system + user", [&](testing & t) {
        (void) t;
        check("system_user",
              ordered_json::array({
                  {{"role", "system"}, {"content", "You are X"}},
                  {{"role", "user"},   {"content", "Hi"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("multi-turn user/assistant", [&](testing & t) {
        (void) t;
        check("multi_turn",
              ordered_json::array({
                  {{"role", "user"},      {"content", "Hello"}},
                  {{"role", "assistant"}, {"content", "Hi there"}},
                  {{"role", "user"},      {"content", "Goodbye"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("with tools", [&](testing & t) {
        (void) t;
        ordered_json tool = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"description", "Get the weather"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {{"city", {{"type", "string"}}}}},
                    {"required", ordered_json::array({"city"})},
                }},
            }},
        };
        check("with_tools",
              ordered_json::array({{{"role", "user"}, {"content", "Hi"}}}),
              ordered_json::array({tool}),
              /*add_generation_prompt=*/true);
    });

    t.test("assistant tool call + tool result", [&](testing & t) {
        (void) t;
        ordered_json tool_call = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"arguments", {{"city", "Paris"}}},
            }},
        };
        check("tool_cycle",
              ordered_json::array({
                  {{"role", "user"},      {"content", "What's the weather?"}},
                  {{"role", "assistant"}, {"content", ""}, {"tool_calls", ordered_json::array({tool_call})}},
                  {{"role", "tool"},      {"tool_call_id", "call_0"}, {"content", "{\"temp\":18}"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("assistant with reasoning_content", [&](testing & t) {
        (void) t;
        // reasoning_content is preprocessed into a `thinking` typed-content
        // block, which the Jinja template renders as `[THINK]...[/THINK]`.
        check("with_reasoning",
              ordered_json::array({
                  {{"role", "user"},      {"content", "Solve 2+2"}},
                  {{"role", "assistant"}, {"content", "4"}, {"reasoning_content", "Adding 2 and 2"}},
                  {{"role", "user"},      {"content", "Thanks"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// Gemma 4 writer parity tests
// ──────────────────────────────────────────────────────────────────────────────

static void test_gemma4_render(testing & t) {
    auto tmpls = load_template("models/templates/google-gemma-4-31B-it.jinja");
    t.assert_true("Gemma 4 template loads", tmpls != nullptr);
    if (!tmpls) {
        return;
    }

    auto check = [&](const std::string & name,
                     const ordered_json & messages,
                     const ordered_json & tools,
                     bool add_generation_prompt,
                     bool enable_thinking) {
        const std::string jinja_out = render_via_jinja(
            tmpls.get(), messages, tools, add_generation_prompt, enable_thinking);
        auto gp = to_generation_params(messages, tools, add_generation_prompt, enable_thinking);
        const std::string bos_token;
        const std::string writer_out = common_chat_gemma4_render(gp, bos_token);
        if (jinja_out == writer_out) {
            t.assert_true(name + " bytes match", true);
        } else {
            std::cerr << "\n=== Gemma 4 mismatch: " << name << " ===\n";
            std::cerr << "Jinja  bytes (" << jinja_out.size() << "): " << jinja_out  << "\n";
            std::cerr << "Writer bytes (" << writer_out.size() << "): " << writer_out << "\n";
            t.assert_true(name + " bytes match", false);
        }
    };

    t.test("user-only no thinking", [&](testing & t) {
        (void) t;
        check("user_only",
              ordered_json::array({{{"role", "user"}, {"content", "Hello"}}}),
              ordered_json::array(),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/false);
    });

    t.test("user-only with thinking", [&](testing & t) {
        (void) t;
        check("user_only_think",
              ordered_json::array({{{"role", "user"}, {"content", "Hello"}}}),
              ordered_json::array(),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/true);
    });

    t.test("system + user", [&](testing & t) {
        (void) t;
        check("system_user",
              ordered_json::array({
                  {{"role", "system"}, {"content", "You are X"}},
                  {{"role", "user"},   {"content", "Hi"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/false);
    });

    t.test("multi-turn", [&](testing & t) {
        (void) t;
        check("multi_turn",
              ordered_json::array({
                  {{"role", "user"},      {"content", "Hello"}},
                  {{"role", "assistant"}, {"content", "Hi there"}},
                  {{"role", "user"},      {"content", "Goodbye"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/false);
    });

    t.test("with single-arg tool", [&](testing & t) {
        (void) t;
        ordered_json tool = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"description", "Get the weather"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {{"city", {{"type", "string"}, {"description", "The city"}}}}},
                    {"required", ordered_json::array({"city"})},
                }},
            }},
        };
        check("with_tool",
              ordered_json::array({{{"role", "user"}, {"content", "Hi"}}}),
              ordered_json::array({tool}),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/false);
    });

    t.test("assistant tool call + tool result", [&](testing & t) {
        (void) t;
        ordered_json tool = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"description", "Get the weather"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {{"city", {{"type", "string"}}}}},
                    {"required", ordered_json::array({"city"})},
                }},
            }},
        };
        ordered_json tool_call = {
            {"type", "function"},
            {"id", "call_0"},
            {"function", {
                {"name", "get_weather"},
                {"arguments", {{"city", "Paris"}}},
            }},
        };
        check("tool_cycle",
              ordered_json::array({
                  {{"role", "user"},      {"content", "What's the weather?"}},
                  {{"role", "assistant"}, {"content", ""}, {"tool_calls", ordered_json::array({tool_call})}},
                  {{"role", "tool"},      {"tool_call_id", "call_0"}, {"name", "get_weather"}, {"content", "{\"temp\":18}"}},
              }),
              ordered_json::array({tool}),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/false);
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// DeepSeek-V3.2 writer parity tests
// ──────────────────────────────────────────────────────────────────────────────

static void test_deepseek_v3_2_render(testing & t) {
    auto tmpls = load_template("models/templates/deepseek-ai-DeepSeek-V3.2.jinja");
    t.assert_true("DeepSeek-V3.2 template loads", tmpls != nullptr);
    if (!tmpls) {
        return;
    }

    auto check = [&](const std::string & name,
                     const ordered_json & messages,
                     const ordered_json & tools,
                     bool add_generation_prompt,
                     bool enable_thinking) {
        const std::string jinja_out = render_via_jinja(
            tmpls.get(), messages, tools, add_generation_prompt, enable_thinking);
        auto gp = to_generation_params(messages, tools, add_generation_prompt, enable_thinking);
        const std::string bos_token;
        const std::string writer_out = common_chat_deepseek_v3_2_render(gp, bos_token);
        if (jinja_out == writer_out) {
            t.assert_true(name + " bytes match", true);
        } else {
            std::cerr << "\n=== DeepSeek-V3.2 mismatch: " << name << " ===\n";
            std::cerr << "Jinja  bytes (" << jinja_out.size() << "): " << jinja_out  << "\n";
            std::cerr << "Writer bytes (" << writer_out.size() << "): " << writer_out << "\n";
            t.assert_true(name + " bytes match", false);
        }
    };

    t.test("user-only no thinking", [&](testing & t) {
        (void) t;
        check("user_only",
              ordered_json::array({{{"role", "user"}, {"content", "Hello"}}}),
              ordered_json::array(),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/false);
    });

    t.test("user-only with thinking", [&](testing & t) {
        (void) t;
        check("user_only_think",
              ordered_json::array({{{"role", "user"}, {"content", "Hello"}}}),
              ordered_json::array(),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/true);
    });

    t.test("system + user", [&](testing & t) {
        (void) t;
        check("system_user",
              ordered_json::array({
                  {{"role", "system"}, {"content", "You are X"}},
                  {{"role", "user"},   {"content", "Hi"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/false);
    });

    t.test("multi-turn", [&](testing & t) {
        (void) t;
        check("multi_turn",
              ordered_json::array({
                  {{"role", "user"},      {"content", "Hello"}},
                  {{"role", "assistant"}, {"content", "Hi there"}},
                  {{"role", "user"},      {"content", "Goodbye"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/false);
    });

    t.test("with tools", [&](testing & t) {
        (void) t;
        ordered_json tool = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"description", "Get the weather"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {{"city", {{"type", "string"}}}}},
                    {"required", ordered_json::array({"city"})},
                }},
            }},
        };
        check("with_tools",
              ordered_json::array({{{"role", "user"}, {"content", "Hi"}}}),
              ordered_json::array({tool}),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/false);
    });

    t.test("tool call + tool result", [&](testing & t) {
        (void) t;
        ordered_json tool = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"description", "Get the weather"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {{"city", {{"type", "string"}}}}},
                    {"required", ordered_json::array({"city"})},
                }},
            }},
        };
        ordered_json tool_call = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"arguments", {{"city", "Paris"}}},
            }},
        };
        check("tool_cycle",
              ordered_json::array({
                  {{"role", "user"},      {"content", "What's the weather?"}},
                  {{"role", "assistant"}, {"content", ""}, {"tool_calls", ordered_json::array({tool_call})}},
                  {{"role", "tool"},      {"content", "{\"temp\":18}"}},
              }),
              ordered_json::array({tool}),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/false);
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// GPT-OSS writer parity tests
// ──────────────────────────────────────────────────────────────────────────────

static void test_gpt_oss_render(testing & t) {
    auto tmpls = load_template("models/templates/openai-gpt-oss-120b.jinja");
    t.assert_true("GPT-OSS template loads", tmpls != nullptr);
    if (!tmpls) {
        return;
    }

    auto check = [&](const std::string & name,
                     const ordered_json & messages,
                     const ordered_json & tools,
                     bool add_generation_prompt) {
        const std::string jinja_out = render_via_jinja(
            tmpls.get(), messages, tools, add_generation_prompt, /*enable_thinking=*/false);
        auto gp = to_generation_params(messages, tools, add_generation_prompt, /*enable_thinking=*/false);
        const std::string writer_out = common_chat_gpt_oss_render(gp);
        if (jinja_out == writer_out) {
            t.assert_true(name + " bytes match", true);
        } else {
            std::cerr << "\n=== GPT-OSS mismatch: " << name << " ===\n";
            std::cerr << "Jinja  bytes (" << jinja_out.size() << "): " << jinja_out  << "\n";
            std::cerr << "Writer bytes (" << writer_out.size() << "): " << writer_out << "\n";
            t.assert_true(name + " bytes match", false);
        }
    };

    t.test("user-only", [&](testing & t) {
        (void) t;
        check("user_only",
              ordered_json::array({{{"role", "user"}, {"content", "Hello"}}}),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("system + user", [&](testing & t) {
        (void) t;
        check("system_user",
              ordered_json::array({
                  {{"role", "system"}, {"content", "You are X"}},
                  {{"role", "user"},   {"content", "Hi"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("multi-turn user/assistant", [&](testing & t) {
        (void) t;
        check("multi_turn",
              ordered_json::array({
                  {{"role", "user"},      {"content", "Hello"}},
                  {{"role", "assistant"}, {"content", "Hi there"}},
                  {{"role", "user"},      {"content", "Goodbye"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("with tool", [&](testing & t) {
        (void) t;
        ordered_json tool = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"description", "Get the weather"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {{"city", {{"type", "string"}, {"description", "The city"}}}}},
                    {"required", ordered_json::array({"city"})},
                }},
            }},
        };
        check("with_tool",
              ordered_json::array({{{"role", "user"}, {"content", "Hi"}}}),
              ordered_json::array({tool}),
              /*add_generation_prompt=*/true);
    });

    t.test("assistant tool call + tool result", [&](testing & t) {
        (void) t;
        ordered_json tool = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"description", "Get the weather"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {{"city", {{"type", "string"}}}}},
                    {"required", ordered_json::array({"city"})},
                }},
            }},
        };
        ordered_json tool_call = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"arguments", {{"city", "Paris"}}},
            }},
        };
        check("tool_cycle",
              ordered_json::array({
                  {{"role", "user"},      {"content", "What's the weather?"}},
                  {{"role", "assistant"}, {"content", "Let me check"}, {"tool_calls", ordered_json::array({tool_call})}},
                  {{"role", "tool"},      {"content", "{\"temp\":18}"}},
              }),
              ordered_json::array({tool}),
              /*add_generation_prompt=*/true);
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// Hermes writer parity tests
// ──────────────────────────────────────────────────────────────────────────────

static void test_hermes_render(testing & t) {
    auto tmpls = load_template("models/templates/NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja");
    t.assert_true("Hermes template loads", tmpls != nullptr);
    if (!tmpls) {
        return;
    }

    auto check = [&](const std::string & name,
                     const ordered_json & messages,
                     const ordered_json & tools,
                     bool add_generation_prompt) {
        const std::string jinja_out = render_via_jinja(
            tmpls.get(), messages, tools, add_generation_prompt, /*enable_thinking=*/true);
        auto gp = to_generation_params(messages, tools, add_generation_prompt, /*enable_thinking=*/true);
        const std::string bos_token;
        const std::string writer_out = common_chat_hermes_render(gp, bos_token);
        if (jinja_out == writer_out) {
            t.assert_true(name + " bytes match", true);
        } else {
            std::cerr << "\n=== Hermes mismatch: " << name << " ===\n";
            std::cerr << "Jinja  bytes (" << jinja_out.size() << "): " << jinja_out  << "\n";
            std::cerr << "Writer bytes (" << writer_out.size() << "): " << writer_out << "\n";
            t.assert_true(name + " bytes match", false);
        }
    };

    t.test("user-only no tools", [&](testing & t) {
        (void) t;
        check("user_only",
              ordered_json::array({{{"role", "user"}, {"content", "Hello"}}}),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("system + user", [&](testing & t) {
        (void) t;
        check("system_user",
              ordered_json::array({
                  {{"role", "system"}, {"content", "You are X"}},
                  {{"role", "user"},   {"content", "Hi"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("multi-turn user/assistant", [&](testing & t) {
        (void) t;
        check("multi_turn",
              ordered_json::array({
                  {{"role", "user"},      {"content", "Hello"}},
                  {{"role", "assistant"}, {"content", "Hi there"}},
                  {{"role", "user"},      {"content", "Goodbye"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true);
    });

    t.test("with single-arg tool", [&](testing & t) {
        (void) t;
        ordered_json tool = {
            {"type", "function"},
            {"function", {
                {"name", "special_function"},
                {"description", "I'm special"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {{"arg1", {{"type", "integer"}, {"description", "The arg."}}}}},
                    {"required", ordered_json::array({"arg1"})},
                }},
            }},
        };
        check("with_tool",
              ordered_json::array({{{"role", "user"}, {"content", "Hi"}}}),
              ordered_json::array({tool}),
              /*add_generation_prompt=*/true);
    });

    t.test("assistant tool call", [&](testing & t) {
        (void) t;
        ordered_json tool = {
            {"type", "function"},
            {"function", {
                {"name", "special_function"},
                {"description", "I'm special"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {{"arg1", {{"type", "integer"}, {"description", "The arg."}}}}},
                    {"required", ordered_json::array({"arg1"})},
                }},
            }},
        };
        ordered_json tool_call = {
            {"type", "function"},
            {"function", {
                {"name", "special_function"},
                {"arguments", {{"arg1", 1}}},
            }},
        };
        check("assistant_tool_call",
              ordered_json::array({
                  {{"role", "user"},      {"content", "Hi"}},
                  {{"role", "assistant"}, {"content", ""}, {"tool_calls", ordered_json::array({tool_call})}},
              }),
              ordered_json::array({tool}),
              /*add_generation_prompt=*/false);
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// QwQ writer parity tests
// ──────────────────────────────────────────────────────────────────────────────

static void test_qwq_render(testing & t) {
    auto tmpls = load_template("models/templates/Qwen-QwQ-32B.jinja");
    t.assert_true("QwQ template loads", tmpls != nullptr);
    if (!tmpls) {
        return;
    }

    auto check = [&](const std::string & name,
                     const ordered_json & messages,
                     const ordered_json & tools,
                     bool add_generation_prompt,
                     bool enable_thinking) {
        const std::string jinja_out = render_via_jinja(
            tmpls.get(), messages, tools, add_generation_prompt, enable_thinking);
        auto gp = to_generation_params(messages, tools, add_generation_prompt, enable_thinking);
        const std::string writer_out = common_chat_qwq_render(gp);
        if (jinja_out == writer_out) {
            t.assert_true(name + " bytes match", true);
        } else {
            std::cerr << "\n=== QwQ mismatch: " << name << " ===\n";
            std::cerr << "Jinja  bytes (" << jinja_out.size() << "): " << jinja_out  << "\n";
            std::cerr << "Writer bytes (" << writer_out.size() << "): " << writer_out << "\n";
            t.assert_true(name + " bytes match", false);
        }
    };

    t.test("user-only thinking on", [&](testing & t) {
        (void) t;
        check("user_only_think",
              ordered_json::array({{{"role", "user"}, {"content", "Hello"}}}),
              ordered_json::array(),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/true);
    });

    t.test("user-only thinking off", [&](testing & t) {
        (void) t;
        check("user_only_no_think",
              ordered_json::array({{{"role", "user"}, {"content", "Hello"}}}),
              ordered_json::array(),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/false);
    });

    t.test("system + user", [&](testing & t) {
        (void) t;
        check("system_user",
              ordered_json::array({
                  {{"role", "system"}, {"content", "You are X"}},
                  {{"role", "user"},   {"content", "Hi"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/true);
    });

    t.test("with tools", [&](testing & t) {
        (void) t;
        ordered_json tool = {
            {"type", "function"},
            {"function", {
                {"name", "get_weather"},
                {"description", "Get the weather"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {{"city", {{"type", "string"}}}}},
                    {"required", ordered_json::array({"city"})},
                }},
            }},
        };
        check("with_tools",
              ordered_json::array({{{"role", "user"}, {"content", "Hi"}}}),
              ordered_json::array({tool}),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/true);
    });

    t.test("past-assistant thinking strip", [&](testing & t) {
        (void) t;
        // Past assistant content with `</think>` has the prefix stripped.
        check("past_thinking_strip",
              ordered_json::array({
                  {{"role", "user"},      {"content", "Q1"}},
                  {{"role", "assistant"}, {"content", "<think>past r</think>past a"}},
                  {{"role", "user"},      {"content", "Q2"}},
              }),
              ordered_json::array(),
              /*add_generation_prompt=*/true,
              /*enable_thinking=*/true);
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// main
// ──────────────────────────────────────────────────────────────────────────────

int main(int argc, char * argv[]) {
    testing t(std::cout);

    if (argc >= 2) {
        t.set_filter(argv[1]);
    }
    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }

    common_chat_grammar_init("grammars/chat");

    t.test("kimi-k2 writer parity",          test_kimi_k2_render);
    t.test("functionary-v3.2 writer parity", test_functionary_v3_2_render);
    t.test("gigachat-v3 writer parity",      test_gigachat_v3_render);
    t.test("glm-4-7-flash writer parity",    test_glm_4_7_flash_render);
    t.test("lfm2 writer parity",             test_lfm2_render);
    t.test("lfm2-5 writer parity",           test_lfm2_5_render);
    t.test("ministral-3 writer parity",      test_ministral_3_render);
    t.test("gemma4 writer parity",           test_gemma4_render);
    t.test("deepseek-v3.2 writer parity",    test_deepseek_v3_2_render);
    t.test("gpt-oss writer parity",          test_gpt_oss_render);
    t.test("hermes writer parity",           test_hermes_render);
    t.test("qwq writer parity",              test_qwq_render);

    return t.summary();
}
