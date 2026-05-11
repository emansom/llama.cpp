#include "chat-formats/glm-4-6-format.h"

#include <nlohmann/json.hpp>

#include <sstream>
#include <string>

namespace {
using ordered_json = nlohmann::ordered_json;

std::string to_jinja_json(const ordered_json & v) {
    if (v.is_null())   return "null";
    if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
    if (v.is_string() || v.is_number()) return v.dump();
    if (v.is_array()) {
        std::string out = "[";
        bool first = true;
        for (const auto & item : v) {
            if (!first) out += ", ";
            first = false;
            out += to_jinja_json(item);
        }
        out += "]";
        return out;
    }
    if (v.is_object()) {
        std::string out = "{";
        bool first = true;
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (!first) out += ", ";
            first = false;
            out += ordered_json(it.key()).dump();
            out += ": ";
            out += to_jinja_json(it.value());
        }
        out += "}";
        return out;
    }
    return "";
}

std::string trim_str(const std::string & s) {
    size_t s_pos = s.find_first_not_of(" \t\n\r");
    if (s_pos == std::string::npos) return {};
    size_t e_pos = s.find_last_not_of(" \t\n\r");
    return s.substr(s_pos, e_pos - s_pos + 1);
}

std::string rstrip_newlines(const std::string & s) {
    size_t pos = s.find_last_not_of('\n');
    return pos == std::string::npos ? std::string{} : s.substr(0, pos + 1);
}

std::string lstrip_newlines(const std::string & s) {
    size_t pos = s.find_first_not_of('\n');
    return pos == std::string::npos ? std::string{} : s.substr(pos);
}

std::string visible_text(const ordered_json & content) {
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (content.is_array()) {
        std::string out;
        for (const auto & item : content) {
            if (item.is_object() && item.value("type", std::string{}) == "text" &&
                item.contains("text") && item["text"].is_string()) {
                out += item["text"].get<std::string>();
            } else if (item.is_string()) {
                out += item.get<std::string>();
            }
        }
        return out;
    }
    return {};
}

bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() &&
        s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

std::string common_chat_glm_4_6_render(const autoparser::generation_params & inputs) {
    std::ostringstream out;

    out << "[gMASK]<sop>";

    const bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    if (has_tools) {
        out << "<|system|>\n# Tools\n\n"
               "You may call one or more functions to assist with the user query.\n\n"
               "You are provided with function signatures within <tools></tools> XML tags:\n"
               "<tools>\n";
        for (const auto & tool : inputs.tools) {
            out << to_jinja_json(tool) << "\n";
        }
        out << "</tools>\n\n"
               "For each function call, output the function name and arguments within the following XML format:\n"
               "<tool_call>{function-name}\n"
               "<arg_key>{arg-key-1}</arg_key>\n"
               "<arg_value>{arg-value-1}</arg_value>\n"
               "<arg_key>{arg-key-2}</arg_key>\n"
               "<arg_value>{arg-value-2}</arg_value>\n"
               "...\n"
               "</tool_call>";
    }

    if (!inputs.messages.is_array()) {
        if (inputs.add_generation_prompt) {
            out << "<|assistant|>";
            if (!inputs.enable_thinking) out << "\n<think></think>";
        }
        return out.str();
    }

    int last_user_index = -1;
    for (size_t i = 0; i < inputs.messages.size(); ++i) {
        if (inputs.messages[i].value("role", std::string{}) == "user") {
            last_user_index = static_cast<int>(i);
        }
    }

    for (size_t i = 0; i < inputs.messages.size(); ++i) {
        const auto & m = inputs.messages[i];
        const std::string role = m.value("role", std::string{});

        if (role == "user") {
            const std::string content = visible_text(m.value("content", ordered_json{}));
            out << "<|user|>\n" << content;
            const bool needs_nothink = !inputs.enable_thinking && !ends_with(content, "/nothink");
            if (needs_nothink) {
                out << "/nothink";
            }
        } else if (role == "assistant") {
            out << "<|assistant|>";
            std::string content = visible_text(m.value("content", ordered_json{}));
            std::string reasoning_content;
            if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) {
                reasoning_content = m["reasoning_content"].get<std::string>();
            } else if (content.find("</think>") != std::string::npos) {
                const std::string close = "</think>";
                size_t close_pos = content.find(close);
                std::string before = rstrip_newlines(content.substr(0, close_pos));
                size_t open_pos = before.rfind("<think>");
                if (open_pos != std::string::npos) {
                    before = before.substr(open_pos + std::string("<think>").size());
                }
                reasoning_content = lstrip_newlines(before);
                content = lstrip_newlines(content.substr(close_pos + close.size()));
            }
            const int loop_idx = static_cast<int>(i);
            if (loop_idx > last_user_index && !reasoning_content.empty()) {
                out << "\n<think>" << trim_str(reasoning_content) << "</think>";
            } else {
                out << "\n<think></think>";
            }
            const std::string content_trim = trim_str(content);
            if (!content_trim.empty()) {
                out << "\n" << content_trim;
            }
            if (m.contains("tool_calls") && m["tool_calls"].is_array() && !m["tool_calls"].empty()) {
                for (const auto & tc_in : m["tool_calls"]) {
                    ordered_json tc = tc_in;
                    if (tc.contains("function") && tc["function"].is_object()) {
                        tc = tc["function"];
                    }
                    out << "\n<tool_call>" << tc.value("name", std::string{});
                    const auto & args = tc.contains("arguments") ? tc["arguments"] : ordered_json::object();
                    if (args.is_object()) {
                        for (auto it = args.begin(); it != args.end(); ++it) {
                            out << "\n<arg_key>" << it.key() << "</arg_key>";
                            out << "\n<arg_value>";
                            const auto & v = it.value();
                            if (v.is_string()) {
                                out << v.get<std::string>();
                            } else {
                                out << to_jinja_json(v);
                            }
                            out << "</arg_value>";
                        }
                    }
                    out << "\n</tool_call>";
                }
            }
        } else if (role == "tool") {
            const std::string prev_role = i > 0
                ? inputs.messages[i - 1].value("role", std::string{}) : std::string{};
            if (i == 0 || prev_role != "tool") {
                out << "<|observation|>";
            }
            const std::string content = m.contains("content") && m["content"].is_string()
                ? m["content"].get<std::string>() : std::string{};
            out << "\n<tool_response>\n" << content << "\n</tool_response>";
        } else if (role == "system") {
            out << "<|system|>\n" << visible_text(m.value("content", ordered_json{}));
        }
    }

    if (inputs.add_generation_prompt) {
        out << "<|assistant|>";
        if (!inputs.enable_thinking) {
            out << "\n<think></think>";
        }
    }
    return out.str();
}
