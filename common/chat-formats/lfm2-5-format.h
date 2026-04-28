#pragma once

#include "chat-formats/lfm2-format.h"

// LFM2.5 chat format: same Python-style `[name(k=v)]` as LFM2, but the
// `<|tool_call_start|>...<|tool_call_end|>` wrapper tokens are optional.
// The grammar handles that difference; the four pipeline layers behave the
// same as LFM2.

class common_chat_lfm2_5_tracker     : public common_chat_lfm2_tracker {};
class common_chat_lfm2_5_decoder     : public common_chat_lfm2_decoder {
  public:
    using common_chat_lfm2_decoder::common_chat_lfm2_decoder;
};
class common_chat_lfm2_5_transformer : public common_chat_lfm2_transformer {
  public:
    using common_chat_lfm2_transformer::common_chat_lfm2_transformer;
};
