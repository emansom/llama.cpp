#include "sampling.h"
#include "log.h"

#ifdef LLAMA_USE_LLGUIDANCE

#    include "llguidance.h"
#    include <cmath>

struct llama_sampler_llg {
    const llama_vocab * vocab;
    std::string         grammar_kind;
    std::string         grammar_data;
    LlgTokenizer *      tokenizer;
    LlgMatcher *        grammar;
    // Set once the matcher has died and this sampler has stopped constraining
    // anything. `grammar == nullptr` alone cannot distinguish "no grammar was
    // ever requested" from "a grammar was requested and is now silently gone",
    // and those two need very different reactions.
    bool                failed_open;
    // Generation step, so the per-token trace lines up with the output.
    int                 step;
};

// Why the last grammar failed to build.
//
// The matcher owns the message and is freed on failure, so it has to be copied
// out here or it is gone. It is worth keeping: the caller supplied that grammar
// and gets a 400 for it, and "failed to parse grammar" does not tell them which
// line of their own text was wrong. llguidance's message does.
static std::string & llg_last_error_slot() {
    static thread_local std::string err;
    return err;
}

const char * llama_sampler_llg_last_error() {
    return llg_last_error_slot().c_str();
}

static LlgMatcher * llama_sampler_llg_new(LlgTokenizer * tokenizer, const char * grammar_kind,
                                          const char * grammar_data) {
    LlgConstraintInit cinit;
    llg_constraint_init_set_defaults(&cinit, tokenizer);
    const char * log_level = getenv("LLGUIDANCE_LOG_LEVEL");
    if (log_level && *log_level) {
        cinit.log_stderr_level = atoi(log_level);
    }
    llg_last_error_slot().clear();
    auto c = llg_new_matcher(&cinit, grammar_kind, grammar_data);
    if (llg_matcher_get_error(c)) {
        LOG_ERR("llg error: %s\n", llg_matcher_get_error(c));
        llg_last_error_slot() = llg_matcher_get_error(c);
        llg_free_matcher(c);
        return nullptr;
    }

    return c;
}

static const char * llama_sampler_llg_name(const llama_sampler * /*smpl*/) {
    return "llguidance";
}

static void llama_sampler_llg_accept_impl(llama_sampler * smpl, llama_token token) {
    auto * ctx = (llama_sampler_llg *) smpl->ctx;
    if (ctx->grammar) {
        llg_matcher_consume_token(ctx->grammar, token);
        ctx->step++;
    }
}

static void llama_sampler_llg_apply(llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_llg *) smpl->ctx;
    if (ctx->grammar) {
        const uint32_t * mask = llg_matcher_get_mask(ctx->grammar);
        if (mask == nullptr) {
            if (llg_matcher_compute_mask(ctx->grammar) == 0) {
                mask = llg_matcher_get_mask(ctx->grammar);
            } else {
                // FAIL-OPEN. The matcher is dropped here and every later apply()
                // is a no-op, so from this token on the model samples with NO
                // constraint whatsoever -- and the only trace of it was one
                // LOG_ERR line, easily lost in a busy log. A silently
                // unconstrained sampler is the single most consequential thing
                // that can go wrong in a grammar-driven format, so say so in
                // terms that cannot be mistaken for a transient hiccup.
                LOG_ERR("llg error: %s\n", llg_matcher_get_error(ctx->grammar));
                LOG_ERR("%s", "llguidance: constraint DROPPED at this token; the rest of this "
                              "generation is UNCONSTRAINED. Output may not conform to the "
                              "requested grammar or tool schema.\n");
                llg_free_matcher(ctx->grammar);
                ctx->grammar     = nullptr;
                ctx->failed_open = true;
                return;
            }
        }

        size_t allowed = 0;
        for (size_t i = 0; i < cur_p->size; ++i) {
            auto token = cur_p->data[i].id;
            if ((mask[token / 32] & (1 << (token % 32))) == 0) {
                cur_p->data[i].logit = -INFINITY;
            } else {
                allowed++;
            }
        }

        // The per-token trace: what the grammar actually did at this step.
        //
        // This is the only place the mask is observable from a running server.
        // Response-level `logprobs` cannot substitute for it: `post_sampling_probs`
        // reports the distribution after the whole chain has collapsed onto the
        // selected token, and the pre-sampling view shows the model's preference
        // rather than what was legal.
        //
        // TWO PATHS, and conflating them makes the trace lie. common_sampler_sample
        // runs the chain FIRST and then applies this sampler to a ONE-ELEMENT array
        // holding the already-sampled token -- grammar-based rejection sampling. So
        // a size-1 call is a yes/no check on that token, and "1/1 allowed" there
        // means "accepted", not "the grammar left one legal token". Only when that
        // check REJECTS does the caller re-sample with the full mask applied first,
        // which is the size-N call where the count is a real mask width.
        if (cur_p->size == 1) {
            LOG_DBG("llguidance: step %d | check token %6d -> %s\n", ctx->step,
                    cur_p->data[0].id, allowed == 1 ? "ok" : "REJECTED, re-sampling under mask");
        } else {
            LOG_DBG("llguidance: step %d | mask allows %zu of %zu tokens%s\n", ctx->step, allowed,
                    cur_p->size, allowed == 1 ? "  (FORCED: no choice)" : "");
        }
    } else if (ctx->failed_open) {
        LOG_DBG("%s", "llguidance: sampling UNCONSTRAINED (constraint was dropped earlier)\n");
    }
}

static void llama_sampler_llg_reset(llama_sampler * smpl) {
    auto * ctx = (llama_sampler_llg *) smpl->ctx;
    if (ctx->grammar) {
        llg_matcher_reset(ctx->grammar);
    }
    // `failed_open` is deliberately NOT cleared: a dropped matcher is gone for
    // the life of this sampler, so resetting the step counter must not make the
    // sampler look constrained again.
    ctx->step = 0;
}

static llama_sampler * llama_sampler_llg_clone(const llama_sampler * smpl) {
    const auto * ctx = (const llama_sampler_llg *) smpl->ctx;

    auto * result = llama_sampler_init_llg(ctx->vocab, nullptr, nullptr);

    // copy the state
    {
        auto * result_ctx = (llama_sampler_llg *) result->ctx;

        if (ctx->grammar) {
            result_ctx->grammar_kind = ctx->grammar_kind;
            result_ctx->grammar_data = ctx->grammar_data;
            result_ctx->grammar      = llg_clone_matcher(ctx->grammar);
            result_ctx->tokenizer    = llg_clone_tokenizer(ctx->tokenizer);
        }
        // Carried so a clone of a dropped constraint still reports itself as
        // unconstrained rather than as "no grammar requested".
        result_ctx->failed_open = ctx->failed_open;
        result_ctx->step        = ctx->step;
    }

    return result;
}

static void llama_sampler_llg_free(llama_sampler * smpl) {
    const auto * ctx = (llama_sampler_llg *) smpl->ctx;

    if (ctx->grammar) {
        llg_free_matcher(ctx->grammar);
        llg_free_tokenizer(ctx->tokenizer);
    }

    delete ctx;
}

static llama_sampler_i llama_sampler_llg_i = {
    /* .name              = */ llama_sampler_llg_name,
    /* .accept            = */ llama_sampler_llg_accept_impl,
    /* .apply             = */ llama_sampler_llg_apply,
    /* .reset             = */ llama_sampler_llg_reset,
    /* .clone             = */ llama_sampler_llg_clone,
    /* .free              = */ llama_sampler_llg_free,
    /* .backend_init      = */ NULL,
    /* .backend_accept    = */ NULL,
    /* .backend_apply     = */ NULL,
    /* .backend_set_input = */ NULL,
};

static size_t llama_sampler_llg_tokenize_fn(const void * user_data, const uint8_t * bytes, size_t bytes_len,
                                            uint32_t * output_tokens, size_t output_tokens_len) {
    const llama_vocab * vocab = (const llama_vocab *) user_data;
    int                 r     = 0;
    try {
        r = llama_tokenize(vocab, (const char *) bytes, bytes_len, (int32_t *) output_tokens, output_tokens_len, false,
                           true);
    } catch (const std::exception & e) {
        GGML_ABORT("llama_tokenize failed: %s\n", e.what());
    }
    if (r < 0) {
        return -r;
    }
    return r;
}

static LlgTokenizer * llama_sampler_llg_new_tokenizer(const llama_vocab * vocab) {
    // TODO store the tokenizer in the vocab somehow
    static const llama_vocab * vocab_cache;
    static LlgTokenizer *      tokenizer_cache;

    if (vocab_cache == vocab) {
        return llg_clone_tokenizer(tokenizer_cache);
    }

    auto tok_eos = llama_vocab_eot(vocab);
    if (tok_eos == LLAMA_TOKEN_NULL) {
        tok_eos = llama_vocab_eos(vocab);
    }

    size_t vocab_size = llama_vocab_n_tokens(vocab);

    auto token_lens       = new uint32_t[vocab_size];
    // we typically have ~7 bytes per token; let's go on the safe side here
    auto token_bytes_size = vocab_size * 16 + 1024 * 1024;
    auto token_bytes      = new uint8_t[token_bytes_size];

    size_t offset = 0;
    for (size_t i = 0; i < vocab_size; i++) {
        size_t max_token = 1024;
        if (token_bytes_size - offset < max_token) {
            GGML_ABORT("token_bytes buffer too small\n");
        }

        llama_token token = i;
        auto        dp    = (char *) token_bytes + offset;

        // A token is opaque to llguidance -- addressable by name in a grammar,
        // and impossible for a regex to produce -- when its bytes carry the
        // '\xff' marker below.
        //
        // Detokenizing with special=false was the only test for that, which
        // catches CONTROL tokens (they render as nothing) and misses
        // USER_DEFINED ones, because those render their literal text. The
        // distinction is not meaningful here: both are entries the tokenizer's
        // added-tokens list DECLARES, neither is producible as ordinary text,
        // and a grammar needs to name both.
        //
        // Gemma 4 is squarely in the gap. `<|turn>` is CONTROL and worked;
        // `<|channel>`, `<channel|>`, `<|tool_call>`, `<tool_call|>`,
        // `<tool_response|>` and `<|"|>` are USER_DEFINED and did not, so
        // referring to them in a grammar failed with `unknown special token:
        // "<|channel>"`. Left as plain bytes they are also forgeable, so every
        // free-text rule had to carry a hand-built exclusion to avoid eating
        // one -- the thing that made this grammar hard to write correctly.
        //
        // This pair of attributes IS the tokenizer's added-tokens list. Conversion
        // splits that list in two -- an added token becomes CONTROL when it is
        // marked special and USER_DEFINED when it is not -- and llama.cpp exposes
        // no single "was this an added token" predicate, so the union is how the
        // question gets asked.
        //
        // Reading declared attributes, never the token's spelling; nothing is
        // inferred from what the text looks like.
        const bool declared_special =
            llama_vocab_is_control(vocab, token) ||
            (llama_vocab_get_attr(vocab, token) & LLAMA_TOKEN_ATTR_USER_DEFINED) != 0;

        auto size = declared_special ? 0
                                     : llama_detokenize(vocab, &token, 1, dp, max_token, false, false);
        if (size < 0) {
            GGML_ABORT("llama_detokenize failed\n");
        }
        if (size == 0) {
            size = llama_detokenize(vocab, &token, 1, dp + 1, max_token - 1, false, true);
            if (size < 0) {
                GGML_ABORT("llama_detokenize failed\n");
            }
            if (size != 0) {
                *dp = '\xff';  // special token prefix marker
                size += 1;
            }
        }

        token_lens[i] = size;
        offset += size;
    }

    LlgTokenizerInit tinit = {
        /* .vocab_size                         = */ (uint32_t) vocab_size,
        /* .tok_eos                            = */ (uint32_t) tok_eos,
        /* .token_lens                         = */ token_lens,
        /* .token_bytes                        = */ token_bytes,
        /* .tokenizer_json                     = */ nullptr,
        /* .tokenize_assumes_string            = */ true,
        /* .tokenize_fn                        = */ llama_sampler_llg_tokenize_fn,
        /* .use_approximate_greedy_tokenize_fn = */ false,
        /* .tokenize_user_data                 = */ vocab,
        /* .slices                             = */ nullptr,
    };

    char           error_buffer[1024];
    LlgTokenizer * tokenizer = llg_new_tokenizer(&tinit, error_buffer, sizeof(error_buffer));

    delete[] token_bytes;
    delete[] token_lens;

    if (tokenizer == nullptr) {
        LOG_ERR("llg tokenizer error: %s\n", error_buffer);
        return tokenizer;
    }

    if (tokenizer_cache) {
        llg_free_tokenizer(tokenizer_cache);
    }
    vocab_cache     = vocab;
    tokenizer_cache = tokenizer;

    return llg_clone_tokenizer(tokenizer_cache);
}

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab, const char * grammar_kind,
                                       const char * grammar_data) {
    auto * ctx = new llama_sampler_llg;

    if (grammar_kind != nullptr && grammar_kind[0] != '\0') {
        auto tokenizer = llama_sampler_llg_new_tokenizer(vocab);
        *ctx           = {
            /* .vocab        = */ vocab,
            /* .grammar_kind = */ grammar_kind,
            /* .grammar_data = */ grammar_data,
            /* .tokenizer    = */ tokenizer,
            /* .grammar      = */ llama_sampler_llg_new(tokenizer, grammar_kind, grammar_data),
            /* .failed_open  = */ false,
            /* .step         = */ 0,
        };
        if (ctx->grammar) {
            GGML_ASSERT(((size_t) llama_vocab_n_tokens(vocab) + 31) / 32 * 4 ==
                        llg_matcher_get_mask_byte_size(ctx->grammar));
            LOG_DBG("llguidance: matcher built for a %s grammar (%zu bytes); constraint ACTIVE\n",
                    grammar_kind, ctx->grammar_data.size());
        } else {
            // The grammar was ASKED for and could not be built.
            //
            // FAIL CLOSED. Returning a sampler here returns one whose every
            // apply() is a no-op, i.e. a generation that runs with no constraint
            // at all while looking constrained from the outside -- and it looked
            // constrained to common_sampler_init too, whose
            // `if (!grmr && !grammar_str.empty()) throw` could never fire
            // because this function always handed back a non-null sampler. So
            // the loudest failure this fork has was reduced to one LOG_ERR in a
            // busy log, and the request succeeded with unconstrained output.
            //
            // A caller that asked for a grammar and cannot have one needs an
            // error, not prose. nullptr is what makes that throw reachable; the
            // server turns it into a request error naming the grammar.
            LOG_ERR("%s", "llguidance: grammar failed to build; refusing to sample "
                          "unconstrained (the request asked for a grammar).\n");
            llg_free_tokenizer(tokenizer);
            delete ctx;
            return nullptr;
        }
    } else {
        *ctx = {
            /* .vocab        = */ vocab,
            /* .grammar_kind = */ {},
            /* .grammar_data = */ {},
            /* .tokenizer    = */ nullptr,
            /* .grammar      = */ nullptr,
            // No grammar was requested. Unconstrained is correct here, so this
            // is NOT failed_open and must not warn.
            /* .failed_open  = */ false,
            /* .step         = */ 0,
        };
    }

    return llama_sampler_init(
        /* .iface = */ &llama_sampler_llg_i,
        /* .ctx   = */ ctx);
}

#else

llama_sampler * llama_sampler_init_llg(const llama_vocab *, const char *, const char *) {
    LOG_WRN("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
    return nullptr;
}

const char * llama_sampler_llg_last_error() {
    return "llguidance is not enabled in this build";
}

#endif  // LLAMA_USE_LLGUIDANCE
