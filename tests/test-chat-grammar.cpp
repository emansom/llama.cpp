// Tests that each per-format chat grammar accepts hand-authored canonical
// full-conversation strings (system / user / tool-result / assistant turns +
// optional generation prompt) via the `conversation` rule.
//
// The `conversation` rule is a peer entry to the grammar's default `start:`
// rule (which is the assistant-turn continuation consumed by the sampler).
// Tests build the PEG arena, override its root to `conversation` via
// `arena.set_root(arena.get_rule("conversation"))`, and parse a canonical
// string. No grammar text is mutated.

#include "chat.h"
#include "chat-formats/deepseek-v3-2-format.h"
#include "chat-formats/format-state-registry.h"
#include "chat-formats/functionary-v3-2-format.h"
#include "chat-formats/gemma4-format.h"
#include "chat-formats/gigachat-v3-format.h"
#include "chat-formats/glm-4-7-flash-format.h"
#include "chat-formats/gpt-oss-format.h"
#include "chat-formats/kimi-k2-format.h"
#include "chat-formats/lfm2-5-format.h"
#include "chat-formats/lfm2-format.h"
#include "chat-formats/granite-4-format.h"
#include "chat-formats/ministral-3-format.h"
#include "chat-formats/qwen3-5-format.h"
#include "common.h"
#include "lark-to-peg.h"
#include "peg-parser.h"
#include "testing.h"

#include <iostream>
#include <string>

// Substitutes the per-format grammar's runtime placeholders with the same
// generic matchers used by chat_grammar_to_peg() — see common/chat.cpp.
static std::string substitute_placeholders(const std::string & grammar) {
    std::string out = grammar;
    string_replace_all(out, "{{TOOL_SCHEMA}}",     "__JSON_OBJECT__");
    string_replace_all(out, "{{RESPONSE_SCHEMA}}", "__JSON_VALUE__");
    return out;
}

// Builds a PEG arena for `model_key` rooted at `entry_rule`. Loads the
// LLAMA_USE_LLGUIDANCE-selected grammar variant from the registry and
// transpiles via lark-to-peg (this test always exercises the Lark path
// because llguidance compatibility is the goal).
static common_peg_arena build_arena_for_rule(const std::string & model_key,
                                             const std::string & entry_rule) {
    const std::string raw = common_chat_grammar_get(model_key);
    if (raw.empty()) {
        return {};
    }
    auto arena = common_lark_to_peg(substitute_placeholders(raw));
    if (arena.has_rule(entry_rule)) {
        arena.set_root(arena.get_rule(entry_rule));
    }
    return arena;
}

static bool parse_full(const common_peg_arena & arena, const std::string & input) {
    common_peg_parse_context ctx(input);
    auto result = arena.parse(ctx);
    return result.success() && result.end == input.size();
}

// Asserts that every grammar rule declared by `rules` exists in the format's
// PEG arena. The arena is built by `build_arena_for_rule(model_key, "")`,
// which loads the grammar via the registry and transpiles it via lark-to-peg
// without overriding the root rule.
static void assert_state_rules_validate(testing & t,
                                        const std::string & model_key,
                                        const common_chat_format_state_rules & rules) {
    const auto arena = build_arena_for_rule(model_key, "");
    if (arena.empty()) {
        t.assert_true(model_key + " grammar loads", false);
        return;
    }
    const auto missing = rules.missing_in(arena);
    if (missing.empty()) {
        t.assert_true(model_key + " state rules all exist in grammar", true);
    } else {
        std::string list;
        for (const auto & r : missing) {
            list += r + " ";
        }
        t.assert_true(model_key + " state rules missing in grammar: " + list, false);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Kimi-K2
// ──────────────────────────────────────────────────────────────────────────────
//
// Wire shape (per moonshotai-Kimi-K2.jinja):
//   <|im_system|>tool_declare<|im_middle|>{tools_json}<|im_end|>
//   <|im_system|>system<|im_middle|>BODY<|im_end|>
//   <|im_user|>user<|im_middle|>BODY<|im_end|>
//   <|im_assistant|>assistant<|im_middle|>BODY<|im_end|>
//   <|im_system|>tool<|im_middle|>"## Return of "ID"\n"BODY<|im_end|>
//   <|im_assistant|>assistant<|im_middle|>      (generation prompt — optional)

static void test_grammar_kimi_k2_conversation(testing & t) {
    const auto arena = build_arena_for_rule("kimi-k2", "conversation");
    t.assert_true("kimi-k2 grammar loads", !arena.empty());
    t.assert_true("kimi-k2 has conversation rule", arena.has_rule("conversation"));

    t.test("single user turn alone", [&](testing & t) {
        // Smallest possible: one user_turn under turn+.
        const std::string input = "<|im_user|>user<|im_middle|>Hello<|im_end|>";
        t.assert_true("accepts a single user turn", parse_full(arena, input));
    });

    t.test("single user turn with empty body", [&](testing & t) {
        const std::string input = "<|im_user|>user<|im_middle|><|im_end|>";
        t.assert_true("accepts a single user turn with empty body", parse_full(arena, input));
    });

    t.test("system + user + assistant + generation prompt", [&](testing & t) {
        const std::string input =
            "<|im_system|>system<|im_middle|>You are a helpful assistant<|im_end|>"
            "<|im_user|>user<|im_middle|>Hello<|im_end|>"
            "<|im_assistant|>assistant<|im_middle|>Hi there<|im_end|>"
            "<|im_assistant|>assistant<|im_middle|>";
        t.assert_true("accepts canonical 3-turn conversation", parse_full(arena, input));
    });

    t.test("with tool_declare + tool result", [&](testing & t) {
        const std::string input =
            "<|im_system|>tool_declare<|im_middle|>[{\"name\":\"get_weather\"}]<|im_end|>"
            "<|im_user|>user<|im_middle|>What's the weather?<|im_end|>"
            "<|im_assistant|>assistant<|im_middle|>"
                "<|tool_calls_section_begin|>"
                    "<|tool_call_begin|>functions.get_weather:0"
                    "<|tool_call_argument_begin|>{\"city\":\"Paris\"}"
                    "<|tool_call_end|>"
                "<|tool_calls_section_end|>"
            "<|im_end|>"
            "<|im_system|>tool<|im_middle|>## Return of call_0\n{\"temp\":18}<|im_end|>"
            "<|im_assistant|>assistant<|im_middle|>";
        t.assert_true("accepts tool_declare + tool call + tool result + generation prompt",
                      parse_full(arena, input));
    });

    t.test("rejects missing role-marker open", [&](testing & t) {
        // Begins with the channel separator <|im_middle|> rather than the role
        // marker opener (<|im_system|>, <|im_user|>, ...). No turn rule should
        // accept this prefix.
        const std::string input = "<|im_middle|>Hello<|im_end|>";
        t.assert_true("rejects turn missing role-marker opener",
                      !parse_full(arena, input));
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// Functionary v3.2
// ──────────────────────────────────────────────────────────────────────────────

static void test_grammar_functionary_v3_2_conversation(testing & t) {
    const auto arena = build_arena_for_rule("functionary-v3.2", "conversation");
    t.assert_true("functionary-v3.2 grammar loads", !arena.empty());
    t.assert_true("functionary-v3.2 has conversation rule", arena.has_rule("conversation"));

    t.test("system + user + assistant + generation prompt", [&](testing & t) {
        const std::string input =
            "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n>>>all\nHi there<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n>>>";
        t.assert_true("accepts canonical 3-turn conversation", parse_full(arena, input));
    });

    t.test("tool result + assistant tool call", [&](testing & t) {
        const std::string input =
            "<|start_header_id|>user<|end_header_id|>\n\nWhat's the weather?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n>>>get_weather\n{\"city\":\"Paris\"}<|eot_id|>"
            "<|start_header_id|>tool<|end_header_id|>\n\n{\"temp\":18}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n>>>";
        t.assert_true("accepts tool-call + tool-result + generation prompt",
                      parse_full(arena, input));
    });

    t.test("rejects body without <|eot_id|>", [&](testing & t) {
        // Truncated turn — no <|eot_id|> on user message.
        const std::string input =
            "<|start_header_id|>user<|end_header_id|>\n\nHello"
            "<|start_header_id|>assistant<|end_header_id|>\n\n>>>";
        t.assert_true("rejects malformed conversation",
                      !parse_full(arena, input));
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// GigaChat v3
// ──────────────────────────────────────────────────────────────────────────────

static void test_grammar_gigachat_v3_conversation(testing & t) {
    const auto arena = build_arena_for_rule("gigachat-v3", "conversation");
    t.assert_true("gigachat-v3 grammar loads", !arena.empty());
    t.assert_true("gigachat-v3 has conversation rule", arena.has_rule("conversation"));

    t.test("system + user + assistant + generation prompt", [&](testing & t) {
        const std::string input =
            "developer system<|role_sep|>\nDevsystem text<|message_sep|>\n\n"
            "system<|role_sep|>\nYou are helpful.<|message_sep|>\n\n"
            "user<|role_sep|>\nHello<|message_sep|>\n\n"
            "assistant<|role_sep|>\nHi there<|message_sep|>\n\n"
            "assistant<|role_sep|>\n";
        t.assert_true("accepts canonical conversation", parse_full(arena, input));
    });

    t.test("function call + function result + generation prompt", [&](testing & t) {
        const std::string input =
            "user<|role_sep|>\nWhat's the weather?<|message_sep|>\n\n"
            "assistant<|role_sep|>\n<|message_sep|>\n\n"
            "function call<|role_sep|>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}<|message_sep|>\n\n"
            "function result<|role_sep|>\n{\"temp\":18}<|message_sep|>\n\n"
            "assistant<|role_sep|>\n";
        t.assert_true("accepts function call + function result + generation prompt",
                      parse_full(arena, input));
    });

    t.test("rejects unknown role marker", [&](testing & t) {
        const std::string input = "robot<|role_sep|>\nHello<|message_sep|>\n\n";
        t.assert_true("rejects 'robot' role", !parse_full(arena, input));
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// GLM-4.7-Flash
// ──────────────────────────────────────────────────────────────────────────────

static void test_grammar_glm_4_7_flash_conversation(testing & t) {
    const auto arena = build_arena_for_rule("glm-4-7-flash", "conversation");
    t.assert_true("glm-4-7-flash grammar loads", !arena.empty());
    t.assert_true("glm-4-7-flash has conversation rule", arena.has_rule("conversation"));

    t.test("BOS prefix + system + user + assistant + generation prompt", [&](testing & t) {
        const std::string input =
            "[gMASK]<sop>"
            "<|system|>You are helpful."
            "<|user|>Hello"
            "<|assistant|></think>Hi there"
            "<|assistant|><think>";
        t.assert_true("accepts canonical conversation",
                      parse_full(arena, input));
    });

    t.test("observation (tool result) turn", [&](testing & t) {
        const std::string input =
            "<|user|>What's the weather?"
            "<|assistant|></think>Let me check<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
            "<|observation|><tool_response>{\"temp\":18}</tool_response>"
            "<|assistant|><think>";
        t.assert_true("accepts observation turn after tool call",
                      parse_full(arena, input));
    });

    t.test("rejects bare body without role marker", [&](testing & t) {
        const std::string input = "Just some text without any role marker";
        t.assert_true("rejects free body without role marker",
                      !parse_full(arena, input));
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// LFM2
// ──────────────────────────────────────────────────────────────────────────────

static void test_grammar_lfm2_conversation(testing & t) {
    const auto arena = build_arena_for_rule("lfm2", "conversation");
    t.assert_true("lfm2 grammar loads", !arena.empty());
    t.assert_true("lfm2 has conversation rule", arena.has_rule("conversation"));

    t.test("system + user + assistant + generation prompt", [&](testing & t) {
        const std::string input =
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\nHi there<|im_end|>\n"
            "<|im_start|>assistant\n";
        t.assert_true("accepts canonical conversation",
                      parse_full(arena, input));
    });

    t.test("tool turn with response wrapper", [&](testing & t) {
        const std::string input =
            "<|im_start|>user\nWhat's the weather?<|im_end|>\n"
            "<|im_start|>tool\n<|tool_response_start|>{\"temp\":18}<|tool_response_end|><|im_end|>\n"
            "<|im_start|>assistant\n";
        t.assert_true("accepts tool turn with <|tool_response_start|>/<|tool_response_end|>",
                      parse_full(arena, input));
    });

    t.test("rejects tool turn missing <|tool_response_end|>", [&](testing & t) {
        const std::string input =
            "<|im_start|>tool\n<|tool_response_start|>{\"temp\":18}<|im_end|>\n";
        t.assert_true("rejects tool turn without <|tool_response_end|>",
                      !parse_full(arena, input));
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// LFM2.5
// ──────────────────────────────────────────────────────────────────────────────

static void test_grammar_lfm2_5_conversation(testing & t) {
    const auto arena = build_arena_for_rule("lfm2-5", "conversation");
    t.assert_true("lfm2-5 grammar loads", !arena.empty());
    t.assert_true("lfm2-5 has conversation rule", arena.has_rule("conversation"));

    t.test("system + user + assistant + tool + generation prompt", [&](testing & t) {
        const std::string input =
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\nHi there<|im_end|>\n"
            "<|im_start|>tool\n{\"temp\":18}<|im_end|>\n"
            "<|im_start|>assistant\n";
        t.assert_true("accepts canonical conversation",
                      parse_full(arena, input));
    });

    t.test("rejects body without <|im_end|>", [&](testing & t) {
        const std::string input = "<|im_start|>user\nHello\n<|im_start|>assistant\n";
        t.assert_true("rejects unclosed user turn",
                      !parse_full(arena, input));
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// Ministral-3
// ──────────────────────────────────────────────────────────────────────────────

static void test_grammar_ministral_3_conversation(testing & t) {
    const auto arena = build_arena_for_rule("ministral-3", "conversation");
    t.assert_true("ministral-3 grammar loads", !arena.empty());
    t.assert_true("ministral-3 has conversation rule", arena.has_rule("conversation"));

    t.test("system + user + assistant + user (next turn)", [&](testing & t) {
        const std::string input =
            "[SYSTEM_PROMPT]You are helpful.[/SYSTEM_PROMPT]"
            "[INST]Hello[/INST]"
            "Hi there"
            "[INST]Goodbye[/INST]";
        t.assert_true("accepts system + alternating user/assistant turns",
                      parse_full(arena, input));
    });

    t.test("assistant tool call + tool result + assistant", [&](testing & t) {
        const std::string input =
            "[AVAILABLE_TOOLS][{\"name\":\"get_weather\"}][/AVAILABLE_TOOLS]"
            "[INST]What's the weather?[/INST]"
            "[TOOL_CALLS]get_weather[ARGS]{\"city\":\"Paris\"}"
            "[TOOL_RESULTS]{\"temp\":18}[/TOOL_RESULTS]"
            "It's 18 degrees in Paris.";
        t.assert_true("accepts tool-call + tool-results + assistant follow-up",
                      parse_full(arena, input));
    });

    // No structural rejection test — Ministral's no-marker assistant turn
    // forces `assistant_inline_text` to fall back to a permissive rest()-style
    // matcher that accepts any prefix. Strict rejection enforcement is the
    // bidirectional walker's job (Package C+).
}

static void test_grammar_ministral_3_no_reasoning_conversation(testing & t) {
    const auto arena = build_arena_for_rule("ministral-3-no-reasoning", "conversation");
    t.assert_true("ministral-3-no-reasoning grammar loads", !arena.empty());
    t.assert_true("ministral-3-no-reasoning has conversation rule", arena.has_rule("conversation"));

    t.test("conversation rule matches the reasoning variant", [&](testing & t) {
        // The no-reasoning variant intentionally has the SAME `conversation`
        // rule as the reasoning variant — it only differs in how the
        // assistant-turn [THINK] block is surfaced.
        const std::string input =
            "[INST]Hello[/INST]"
            "Hi there"
            "[INST]Goodbye[/INST]";
        t.assert_true("accepts canonical conversation",
                      parse_full(arena, input));
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// Gemma-4
// ──────────────────────────────────────────────────────────────────────────────

static void test_grammar_gemma4_conversation(testing & t) {
    const auto arena = build_arena_for_rule("gemma4", "conversation");
    t.assert_true("gemma4 grammar loads", !arena.empty());
    t.assert_true("gemma4 has conversation rule", arena.has_rule("conversation"));

    t.test("system + user + model + generation prompt", [&](testing & t) {
        const std::string input =
            "<|turn>system\nYou are helpful.<turn|>\n"
            "<|turn>user\nHello<turn|>\n"
            "<|turn>model\nHi there<turn|>\n"
            "<|turn>model\n";
        t.assert_true("accepts canonical 3-turn conversation + generation prompt",
                      parse_full(arena, input));
    });

    t.test("generation prompt with thought_open marker (thinking disabled)", [&](testing & t) {
        const std::string input =
            "<|turn>user\nHello<turn|>\n"
            "<|turn>model\n<|channel>thought\n<channel|>";
        t.assert_true("accepts generation prompt with empty thought block",
                      parse_full(arena, input));
    });

    t.test("rejects turn missing <turn|>", [&](testing & t) {
        const std::string input = "<|turn>user\nHello\n";
        t.assert_true("rejects unclosed user turn",
                      !parse_full(arena, input));
    });
}

static void test_grammar_gemma4_no_reasoning_conversation(testing & t) {
    const auto arena = build_arena_for_rule("gemma4-no-reasoning", "conversation");
    t.assert_true("gemma4-no-reasoning grammar loads", !arena.empty());
    t.assert_true("gemma4-no-reasoning has conversation rule", arena.has_rule("conversation"));

    t.test("conversation rule matches the reasoning variant", [&](testing & t) {
        const std::string input =
            "<|turn>user\nHello<turn|>\n"
            "<|turn>model\nHi there<turn|>\n"
            "<|turn>model\n";
        t.assert_true("accepts canonical conversation",
                      parse_full(arena, input));
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// DeepSeek-V3.2
// ──────────────────────────────────────────────────────────────────────────────

static void test_grammar_deepseek_v3_2_conversation(testing & t) {
    const auto arena = build_arena_for_rule("deepseek-v3.2", "conversation");
    t.assert_true("deepseek-v3.2 grammar loads", !arena.empty());
    t.assert_true("deepseek-v3.2 has conversation rule", arena.has_rule("conversation"));

    t.test("system prefix + user + assistant + generation prompt", [&](testing & t) {
        const std::string input =
            "You are helpful."
            "<｜User｜>Hello"
            "<｜Assistant｜>Hi there<｜end▁of▁sentence｜>"
            "<｜Assistant｜><think>";
        t.assert_true("accepts canonical conversation with system prefix",
                      parse_full(arena, input));
    });

    t.test("function_results block after assistant tool calls", [&](testing & t) {
        const std::string input =
            "<｜User｜>What's the weather?"
            "<｜Assistant｜><think></think>Let me check.<｜end▁of▁sentence｜>"
            "\n\n<function_results>\n<result>{\"temp\":18}</result>\n</function_results>"
            "<｜Assistant｜>";
        t.assert_true("accepts function_results block + generation prompt",
                      parse_full(arena, input));
    });

    t.test("rejects assistant turn missing end-of-sentence", [&](testing & t) {
        const std::string input =
            "<｜User｜>Hello"
            "<｜Assistant｜>Hi there";
        t.assert_true("rejects unclosed assistant turn",
                      !parse_full(arena, input));
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// GPT-OSS (Harmony format)
// ──────────────────────────────────────────────────────────────────────────────

static void test_grammar_gpt_oss_conversation(testing & t) {
    const auto arena = build_arena_for_rule("gpt-oss", "conversation");
    t.assert_true("gpt-oss grammar loads", !arena.empty());
    t.assert_true("gpt-oss has conversation rule", arena.has_rule("conversation"));

    t.test("system + user + assistant analysis + final + generation prompt", [&](testing & t) {
        const std::string input =
            "<|start|>system<|message|>You are helpful.<|end|>"
            "<|start|>user<|message|>Hello<|end|>"
            "<|start|>assistant<|channel|>analysis<|message|>Thinking...<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Hi there<|return|>"
            "<|start|>assistant";
        t.assert_true("accepts canonical multi-turn Harmony conversation",
                      parse_full(arena, input));
    });

    t.test("function tool call + functions response + generation prompt", [&](testing & t) {
        const std::string input =
            "<|start|>user<|message|>What's the weather?<|end|>"
            "<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{\"city\":\"Paris\"}<|call|>"
            "<|start|>functions.get_weather to=assistant<|channel|>commentary<|message|>{\"temp\":18}<|end|>"
            "<|start|>assistant";
        t.assert_true("accepts function tool call + functions response",
                      parse_full(arena, input));
    });

    t.test("rejects body missing terminator", [&](testing & t) {
        const std::string input =
            "<|start|>user<|message|>Hello"
            "<|start|>assistant";
        t.assert_true("rejects user turn missing <|end|>",
                      !parse_full(arena, input));
    });
}

static void test_grammar_gpt_oss_no_reasoning_conversation(testing & t) {
    const auto arena = build_arena_for_rule("gpt-oss-no-reasoning", "conversation");
    t.assert_true("gpt-oss-no-reasoning grammar loads", !arena.empty());
    t.assert_true("gpt-oss-no-reasoning has conversation rule", arena.has_rule("conversation"));

    t.test("conversation rule matches the reasoning variant", [&](testing & t) {
        const std::string input =
            "<|start|>user<|message|>Hello<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Hi<|return|>"
            "<|start|>assistant";
        t.assert_true("accepts canonical conversation",
                      parse_full(arena, input));
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// FSM-state-to-grammar-rule registry validation
// ──────────────────────────────────────────────────────────────────────────────

static void test_state_rules_kimi_k2(testing & t) {
    assert_state_rules_validate(t, "kimi-k2", kimi_k2_state_rules);
}

static void test_state_rules_functionary_v3_2(testing & t) {
    assert_state_rules_validate(t, "functionary-v3.2", functionary_v3_2_state_rules);
}

static void test_state_rules_gigachat_v3(testing & t) {
    assert_state_rules_validate(t, "gigachat-v3", gigachat_v3_state_rules);
}

static void test_state_rules_ministral_3(testing & t) {
    assert_state_rules_validate(t, "ministral-3", ministral_3_state_rules);
}

static void test_state_rules_gpt_oss(testing & t) {
    assert_state_rules_validate(t, "gpt-oss", gpt_oss_state_rules);
}

static void test_state_rules_lfm2(testing & t) {
    assert_state_rules_validate(t, "lfm2", lfm2_state_rules);
}

static void test_state_rules_lfm2_5(testing & t) {
    assert_state_rules_validate(t, "lfm2-5", lfm2_5_state_rules);
}

static void test_state_rules_gemma4(testing & t) {
    assert_state_rules_validate(t, "gemma4", gemma4_state_rules);
}

static void test_state_rules_glm_4_7_flash(testing & t) {
    assert_state_rules_validate(t, "glm-4-7-flash", glm_4_7_flash_state_rules);
}

static void test_state_rules_deepseek_v3_2(testing & t) {
    assert_state_rules_validate(t, "deepseek-v3.2", deepseek_v3_2_state_rules);
}

static void test_state_rules_qwen3_5(testing & t) {
    assert_state_rules_validate(t, "qwen3-5", qwen3_5_state_rules);
}

static void test_state_rules_granite_4(testing & t) {
    assert_state_rules_validate(t, "granite-4", granite_4_state_rules);
}

// ──────────────────────────────────────────────────────────────────────────────
// Granite 4.0
// ──────────────────────────────────────────────────────────────────────────────

static void test_grammar_granite_4_conversation(testing & t) {
    const auto arena = build_arena_for_rule("granite-4", "conversation");
    t.assert_true("granite-4 grammar loads", !arena.empty());
    t.assert_true("granite-4 has conversation rule", arena.has_rule("conversation"));

    t.test("system + user + assistant + generation prompt", [&](testing & t) {
        const std::string input =
            "<|start_of_role|>system<|end_of_role|>You are helpful.<|end_of_text|>\n"
            "<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>\n"
            "<|start_of_role|>assistant<|end_of_role|>Hi there<|end_of_text|>\n"
            "<|start_of_role|>assistant<|end_of_role|>";
        t.assert_true("accepts canonical conversation",
                      parse_full(arena, input));
    });

    t.test("rejects body without <|end_of_text|>", [&](testing & t) {
        const std::string input =
            "<|start_of_role|>user<|end_of_role|>Hello"
            "<|start_of_role|>assistant<|end_of_role|>";
        t.assert_true("rejects malformed conversation",
                      !parse_full(arena, input));
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// Qwen3.5
// ──────────────────────────────────────────────────────────────────────────────

static void test_grammar_qwen3_5_conversation(testing & t) {
    const auto arena = build_arena_for_rule("qwen3-5", "conversation");
    t.assert_true("qwen3-5 grammar loads", !arena.empty());
    t.assert_true("qwen3-5 has conversation rule", arena.has_rule("conversation"));

    t.test("system + user + assistant + generation prompt", [&](testing & t) {
        const std::string input =
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\nHi there<|im_end|>\n"
            "<|im_start|>assistant\n";
        t.assert_true("accepts canonical conversation",
                      parse_full(arena, input));
    });

    t.test("rejects body without <|im_end|>", [&](testing & t) {
        const std::string input =
            "<|im_start|>user\nHello"
            "<|im_start|>assistant\n";
        t.assert_true("rejects malformed conversation",
                      !parse_full(arena, input));
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

    t.test("kimi-k2 conversation grammar", test_grammar_kimi_k2_conversation);
    t.test("functionary-v3.2 conversation grammar", test_grammar_functionary_v3_2_conversation);
    t.test("gigachat-v3 conversation grammar", test_grammar_gigachat_v3_conversation);
    t.test("glm-4-7-flash conversation grammar", test_grammar_glm_4_7_flash_conversation);
    t.test("lfm2 conversation grammar", test_grammar_lfm2_conversation);
    t.test("lfm2-5 conversation grammar", test_grammar_lfm2_5_conversation);
    t.test("ministral-3 conversation grammar", test_grammar_ministral_3_conversation);
    t.test("ministral-3-no-reasoning conversation grammar", test_grammar_ministral_3_no_reasoning_conversation);
    t.test("gemma4 conversation grammar", test_grammar_gemma4_conversation);
    t.test("gemma4-no-reasoning conversation grammar", test_grammar_gemma4_no_reasoning_conversation);
    t.test("deepseek-v3.2 conversation grammar", test_grammar_deepseek_v3_2_conversation);
    t.test("gpt-oss conversation grammar", test_grammar_gpt_oss_conversation);
    t.test("gpt-oss-no-reasoning conversation grammar", test_grammar_gpt_oss_no_reasoning_conversation);
    t.test("qwen3-5 conversation grammar", test_grammar_qwen3_5_conversation);
    t.test("granite-4 conversation grammar", test_grammar_granite_4_conversation);

    // FSM-state-to-grammar-rule registry validation.
    t.test("kimi-k2 state rules",          test_state_rules_kimi_k2);
    t.test("functionary-v3.2 state rules", test_state_rules_functionary_v3_2);
    t.test("gigachat-v3 state rules",      test_state_rules_gigachat_v3);
    t.test("ministral-3 state rules",      test_state_rules_ministral_3);
    t.test("gpt-oss state rules",          test_state_rules_gpt_oss);
    t.test("lfm2 state rules",             test_state_rules_lfm2);
    t.test("lfm2-5 state rules",           test_state_rules_lfm2_5);
    t.test("gemma4 state rules",           test_state_rules_gemma4);
    t.test("glm-4-7-flash state rules",    test_state_rules_glm_4_7_flash);
    t.test("deepseek-v3.2 state rules",    test_state_rules_deepseek_v3_2);
    t.test("qwen3-5 state rules",          test_state_rules_qwen3_5);
    t.test("granite-4 state rules",        test_state_rules_granite_4);

    return t.summary();
}
