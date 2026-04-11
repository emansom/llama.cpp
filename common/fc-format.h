// FC format utility functions for Gemma 4 native tool format.
//
// These functions mirror LiteRT-LM's fc_tool_format_utils and serve as a
// **validation reference** in tests — they are NOT used in the production
// pipeline. The Jinja template and PEG parser handle all format conversion
// at runtime; these functions verify their output matches LiteRT-LM.
//
// Reference: https://github.com/google-ai-edge/LiteRT-LM

#pragma once

#include <nlohmann/json.hpp>
#include <string>

// Configuration matching LiteRT-LM's Gemma4DataProcessorConfig
struct fc_format_config {
    std::string tool_start      = "<|tool>";
    std::string tool_end        = "<tool|>";
    std::string tool_call_start = "<|tool_call>";
    std::string tool_call_end   = "<tool_call|>";
    std::string response_start  = "<|tool_response>";
    std::string response_end    = "<tool_response|>";
    std::string quote           = "<|\"|>";
};

// Recursive value formatter (mirrors LiteRT-LM's FormatValueAsFc).
// Strings are wrapped in <|"|> delimiters, numbers/booleans are raw,
// objects use {key:value,...} with bare keys, arrays use [item,...].
std::string fc_format_value(const nlohmann::ordered_json & value, const fc_format_config & cfg = {});

// JSON Schema property → native format parameter declaration.
// Produces: name:{description:<|"|>...<|"|>,type:<|"|>TYPE<|"|>}
// where TYPE is UPPERCASE (STRING, INTEGER, NUMBER, BOOLEAN, OBJECT, ARRAY).
std::string fc_format_schema_property(const std::string & name, const nlohmann::ordered_json & schema, const fc_format_config & cfg = {});

// Tool definition (OpenAI format) → native declaration.
// Input: {"type":"function","function":{"name":"...","description":"...","parameters":{...}}}
// Output: declaration:name{description:<|"|>...<|"|>,parameters:{...}}
std::string fc_format_tool_declaration(const nlohmann::ordered_json & tool, const fc_format_config & cfg = {});

// Tool call arguments (JSON object) → native format key-value pairs.
// Output: key:<|"|>value<|"|>,key2:42
std::string fc_format_tool_call_args(const nlohmann::ordered_json & args, const fc_format_config & cfg = {});

// Tool response → native format.
// Output: response:name{key:value,...} or response:name{value:<|"|>text<|"|>}
std::string fc_format_tool_response(const std::string & name, const nlohmann::ordered_json & content, const fc_format_config & cfg = {});

// Wrap helpers — add surrounding tokens.
std::string fc_wrap_tool_declaration(const nlohmann::ordered_json & tool, const fc_format_config & cfg = {});
std::string fc_wrap_tool_call(const std::string & name, const nlohmann::ordered_json & args, const fc_format_config & cfg = {});
std::string fc_wrap_tool_response(const std::string & name, const nlohmann::ordered_json & content, const fc_format_config & cfg = {});
