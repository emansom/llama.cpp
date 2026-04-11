// FC format utility functions for Gemma 4 native tool format.
// See fc-format.h for documentation.

#include "fc-format.h"

#include <algorithm>
#include <set>
#include <sstream>

using ordered_json = nlohmann::ordered_json;

// ---------------------------------------------------------------------------
// fc_format_value — recursive value formatter matching LiteRT-LM
// ---------------------------------------------------------------------------

std::string fc_format_value(const ordered_json & value, const fc_format_config & cfg) {
    if (value.is_null()) {
        return "null";
    }
    if (value.is_boolean()) {
        return value.get<bool>() ? "true" : "false";
    }
    if (value.is_number_integer()) {
        return std::to_string(value.get<int64_t>());
    }
    if (value.is_number_float()) {
        // Use dump() to preserve the original representation (e.g. 3.14, 1.5e10)
        return value.dump();
    }
    if (value.is_string()) {
        return cfg.quote + value.get<std::string>() + cfg.quote;
    }
    if (value.is_array()) {
        std::string result = "[";
        bool first = true;
        for (const auto & item : value) {
            if (!first) result += ',';
            first = false;
            result += fc_format_value(item, cfg);
        }
        result += ']';
        return result;
    }
    if (value.is_object()) {
        std::string result = "{";
        bool first = true;
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (!first) result += ',';
            first = false;
            result += it.key() + ':' + fc_format_value(it.value(), cfg);
        }
        result += '}';
        return result;
    }
    return value.dump();
}

// ---------------------------------------------------------------------------
// fc_format_schema_property — JSON Schema property → native declaration
// ---------------------------------------------------------------------------

// Resolve nullable unions: {anyOf:[{type:"string"},{type:"null"}]} → "string"
static std::string resolve_schema_type(const ordered_json & schema) {
    if (schema.contains("type")) {
        const auto & t = schema["type"];
        if (t.is_string()) {
            return t.get<std::string>();
        }
        if (t.is_array()) {
            // e.g. ["string", "null"] → pick the non-null type
            for (const auto & item : t) {
                if (item.is_string() && item.get<std::string>() != "null") {
                    return item.get<std::string>();
                }
            }
        }
    }
    // anyOf / oneOf nullable pattern
    for (const char * key : {"anyOf", "oneOf"}) {
        if (schema.contains(key) && schema[key].is_array()) {
            for (const auto & variant : schema[key]) {
                if (variant.contains("type") && variant["type"].is_string() && variant["type"] != "null") {
                    return variant["type"].get<std::string>();
                }
            }
        }
    }
    return "object"; // default fallback
}

static std::string to_upper(const std::string & s) {
    std::string result = s;
    for (auto & c : result) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return result;
}

// Format the parameters block of a single schema property.
// Produces the inner content: description:<|"|>...<|"|>,type:<|"|>TYPE<|"|>
// plus nested properties/items/enum/nullable/required as needed.
static std::string format_property_inner(const ordered_json & schema, const fc_format_config & cfg) {
    std::string result;
    bool add_comma = false;

    // description
    if (schema.contains("description") && schema["description"].is_string()) {
        result += "description:" + cfg.quote + schema["description"].get<std::string>() + cfg.quote;
        add_comma = true;
    }

    std::string type = resolve_schema_type(schema);
    std::string type_upper = to_upper(type);

    // enum (for STRING types)
    if (type_upper == "STRING" && schema.contains("enum") && schema["enum"].is_array()) {
        if (add_comma) result += ',';
        add_comma = true;
        result += "enum:" + fc_format_value(schema["enum"], cfg);
    }

    // items (for ARRAY types)
    if (type_upper == "ARRAY" && schema.contains("items") && schema["items"].is_object()) {
        if (add_comma) result += ',';
        add_comma = true;
        const auto & items = schema["items"];
        result += "items:{";

        bool items_comma = false;
        // Render items sub-fields in sorted order to match Jinja template dictsort
        std::vector<std::string> item_keys;
        for (auto it = items.begin(); it != items.end(); ++it) {
            item_keys.push_back(it.key());
        }
        std::sort(item_keys.begin(), item_keys.end());

        for (const auto & ik : item_keys) {
            if (items[ik].is_null()) continue;
            if (items_comma) result += ',';
            items_comma = true;

            if (ik == "properties" && items[ik].is_object()) {
                result += "properties:{";
                // Get required list for items
                std::set<std::string> items_required;
                if (items.contains("required") && items["required"].is_array()) {
                    for (const auto & r : items["required"]) {
                        if (r.is_string()) items_required.insert(r.get<std::string>());
                    }
                }
                bool prop_comma = false;
                // Sort property keys to match Jinja dictsort
                std::vector<std::string> sorted_props;
                for (auto pit = items[ik].begin(); pit != items[ik].end(); ++pit) {
                    sorted_props.push_back(pit.key());
                }
                std::sort(sorted_props.begin(), sorted_props.end());
                for (const auto & pk : sorted_props) {
                    if (prop_comma) result += ',';
                    prop_comma = true;
                    result += pk + ":{" + format_property_inner(items[ik][pk], cfg) + "}";
                }
                result += "}";
            } else if (ik == "required" && items[ik].is_array()) {
                result += "required:[";
                bool req_comma = false;
                for (const auto & r : items[ik]) {
                    if (req_comma) result += ',';
                    req_comma = true;
                    result += cfg.quote + r.get<std::string>() + cfg.quote;
                }
                result += "]";
            } else if (ik == "type") {
                if (items[ik].is_string()) {
                    result += "type:" + cfg.quote + to_upper(items[ik].get<std::string>()) + cfg.quote;
                } else if (items[ik].is_array()) {
                    result += "type:[";
                    bool tc = false;
                    for (const auto & ti : items[ik]) {
                        if (tc) result += ',';
                        tc = true;
                        result += cfg.quote + to_upper(ti.get<std::string>()) + cfg.quote;
                    }
                    result += "]";
                }
            } else {
                result += ik + ":" + fc_format_value(items[ik], cfg);
            }
        }
        result += "}";
    }

    // nullable
    if (schema.contains("nullable") && schema["nullable"].is_boolean() && schema["nullable"].get<bool>()) {
        if (add_comma) result += ',';
        add_comma = true;
        result += "nullable:true";
    }

    // nested properties (for OBJECT types)
    if (type_upper == "OBJECT") {
        if (schema.contains("properties") && schema["properties"].is_object()) {
            if (add_comma) result += ',';
            add_comma = true;
            result += "properties:{";
            // Get required set
            std::set<std::string> req_set;
            if (schema.contains("required") && schema["required"].is_array()) {
                for (const auto & r : schema["required"]) {
                    if (r.is_string()) req_set.insert(r.get<std::string>());
                }
            }
            bool prop_comma = false;
            // Sort to match Jinja dictsort
            std::vector<std::string> sorted_props;
            for (auto it = schema["properties"].begin(); it != schema["properties"].end(); ++it) {
                sorted_props.push_back(it.key());
            }
            std::sort(sorted_props.begin(), sorted_props.end());
            for (const auto & pk : sorted_props) {
                if (prop_comma) result += ',';
                prop_comma = true;
                result += pk + ":{" + format_property_inner(schema["properties"][pk], cfg) + "}";
            }
            result += "}";
        }
        // required list for object
        if (schema.contains("required") && schema["required"].is_array()) {
            if (add_comma) result += ',';
            add_comma = true;
            result += "required:[";
            bool req_comma = false;
            for (const auto & r : schema["required"]) {
                if (req_comma) result += ',';
                req_comma = true;
                result += cfg.quote + r.get<std::string>() + cfg.quote;
            }
            result += "]";
        }
    }

    // type (always last per Jinja template: type:<|"|>TYPE<|"|>)
    if (add_comma) result += ',';
    result += "type:" + cfg.quote + type_upper + cfg.quote;

    return result;
}

std::string fc_format_schema_property(const std::string & name, const ordered_json & schema, const fc_format_config & cfg) {
    return name + ":{" + format_property_inner(schema, cfg) + "}";
}

// ---------------------------------------------------------------------------
// fc_format_tool_declaration — OpenAI tool → native declaration
// ---------------------------------------------------------------------------

std::string fc_format_tool_declaration(const ordered_json & tool, const fc_format_config & cfg) {
    const auto & func = tool["function"];
    std::string result = "declaration:" + func["name"].get<std::string>();
    result += "{description:" + cfg.quote + func["description"].get<std::string>() + cfg.quote;

    if (func.contains("parameters") && func["parameters"].is_object()) {
        const auto & params = func["parameters"];
        result += ",parameters:{";

        // properties
        if (params.contains("properties") && params["properties"].is_object()) {
            result += "properties:{";
            // Sort to match Jinja dictsort
            std::vector<std::string> sorted_props;
            for (auto it = params["properties"].begin(); it != params["properties"].end(); ++it) {
                sorted_props.push_back(it.key());
            }
            std::sort(sorted_props.begin(), sorted_props.end());
            bool prop_comma = false;
            for (const auto & pk : sorted_props) {
                if (prop_comma) result += ',';
                prop_comma = true;
                result += pk + ":{" + format_property_inner(params["properties"][pk], cfg) + "}";
            }
            result += "},";
        }

        // required
        if (params.contains("required") && params["required"].is_array()) {
            result += "required:[";
            bool req_comma = false;
            for (const auto & r : params["required"]) {
                if (req_comma) result += ',';
                req_comma = true;
                result += cfg.quote + r.get<std::string>() + cfg.quote;
            }
            result += "],";
        }

        // type
        if (params.contains("type") && params["type"].is_string()) {
            result += "type:" + cfg.quote + to_upper(params["type"].get<std::string>()) + cfg.quote;
        }

        result += "}";
    }

    // response (optional)
    if (func.contains("response") && func["response"].is_object()) {
        const auto & resp = func["response"];
        result += ",response:{";
        if (resp.contains("description") && resp["description"].is_string()) {
            result += "description:" + cfg.quote + resp["description"].get<std::string>() + cfg.quote + ",";
        }
        if (resp.contains("type") && resp["type"].is_string()) {
            result += "type:" + cfg.quote + to_upper(resp["type"].get<std::string>()) + cfg.quote;
        }
        result += "}";
    }

    result += "}";
    return result;
}

// ---------------------------------------------------------------------------
// fc_format_tool_call_args — JSON args → native key:value pairs
// ---------------------------------------------------------------------------

std::string fc_format_tool_call_args(const ordered_json & args, const fc_format_config & cfg) {
    std::string result;
    bool first = true;
    for (auto it = args.begin(); it != args.end(); ++it) {
        if (!first) result += ',';
        first = false;
        result += it.key() + ':' + fc_format_value(it.value(), cfg);
    }
    return result;
}

// ---------------------------------------------------------------------------
// fc_format_tool_response — tool result → native format
// ---------------------------------------------------------------------------

std::string fc_format_tool_response(const std::string & name, const ordered_json & content, const fc_format_config & cfg) {
    std::string result = "response:" + name + "{";
    if (content.is_object()) {
        bool first = true;
        for (auto it = content.begin(); it != content.end(); ++it) {
            if (!first) result += ',';
            first = false;
            result += it.key() + ':' + fc_format_value(it.value(), cfg);
        }
    } else {
        result += "value:" + fc_format_value(content, cfg);
    }
    result += '}';
    return result;
}

// ---------------------------------------------------------------------------
// Wrap helpers
// ---------------------------------------------------------------------------

std::string fc_wrap_tool_declaration(const ordered_json & tool, const fc_format_config & cfg) {
    return cfg.tool_start + fc_format_tool_declaration(tool, cfg) + cfg.tool_end;
}

std::string fc_wrap_tool_call(const std::string & name, const ordered_json & args, const fc_format_config & cfg) {
    return cfg.tool_call_start + "call:" + name + "{" + fc_format_tool_call_args(args, cfg) + "}" + cfg.tool_call_end;
}

std::string fc_wrap_tool_response(const std::string & name, const ordered_json & content, const fc_format_config & cfg) {
    return cfg.response_start + fc_format_tool_response(name, content, cfg) + cfg.response_end;
}
