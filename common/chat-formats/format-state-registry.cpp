#include "chat-formats/format-state-registry.h"

#include <set>

std::string common_chat_format_state_rules::rule_for(common_chat_format_state state) const {
    auto it = by_state.find(state);
    return it == by_state.end() ? std::string{} : it->second;
}

std::vector<std::string> common_chat_format_state_rules::declared_rules() const {
    std::set<std::string> uniq;
    for (const auto & [_, name] : by_state) {
        if (!name.empty()) {
            uniq.insert(name);
        }
    }
    return { uniq.begin(), uniq.end() };
}

std::vector<std::string> common_chat_format_state_rules::missing_in(const common_peg_arena & arena) const {
    std::vector<std::string> missing;
    for (const auto & rule : declared_rules()) {
        if (!arena.has_rule(rule)) {
            missing.push_back(rule);
        }
    }
    return missing;
}
