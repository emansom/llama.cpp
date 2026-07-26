"""Which chat format serves a request, and where that answer came from.

Nothing is inferred in this build: the format is resolved from the request, then
per-model configuration, then declared GGUF metadata, and from no fourth place.
There is no template text to sniff.

The precedence LOGIC is asserted in-process by test-grammar-llguidance
(`test_chat_format_resolution`), which can read the resolver's own source string
and does not need a server. What only a server can show is the two ends of the
chain -- a `chat_format` on the wire, and a bad one refusing to start -- so those
are here. Both were verified by hand first, by starting a server four ways and
reading the log line, which is exactly the check that stops being run.
"""

import pytest
from utils import *

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    # Gemma 4 for the same reason test_chat_completion.py uses it: format
    # resolution reads general.architecture, and stories260K declares `llama`,
    # which this build has no plugin for.
    server = ServerPreset.gemma4()


def _say_hi(**extra):
    return server.make_request("POST", "/chat/completions", data={
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    })


def test_request_chat_format_is_accepted():
    """A request may name the format explicitly -- the top of the precedence chain."""
    server.chat_format = "gemma4"
    server.start()
    res = _say_hi(chat_format="gemma4")
    assert res.status_code == 200, res.body


def test_request_chat_format_unregistered_is_a_400():
    """...and naming one this build has no plugin for is a request error.

    Asserted on the message, not just the status: an error that does not say which
    format was asked for, or what is available instead, sends the reader to the
    source. Both halves have been present from the start and both are easy to lose
    in a refactor.
    """
    server.chat_format = "gemma4"
    server.start()
    res = _say_hi(chat_format="definitely-not-a-format")
    assert res.status_code == 400, res.body
    message = res.body["error"]["message"]
    assert "definitely-not-a-format" in message
    assert "gemma4" in message


def test_config_chat_format_unregistered_fails_at_load():
    """A bad --chat-format stops the server rather than half-working.

    This is also the proof that config OUTRANKS metadata: the model declares
    `gemma4`, so a server that consulted metadata first would come up fine. It
    must not.
    """
    server.chat_format = "definitely-not-a-format"
    with pytest.raises(RuntimeError, match="died"):
        server.start(timeout_seconds=30)


def test_no_chat_format_resolves_from_gguf_metadata():
    """With nothing configured, the model's own declaration is enough.

    This is what lets a stock OpenAI client talk to an unconfigured Gemma 4 GGUF
    with no extra parameters -- the compatibility requirement, not a shortcut.
    """
    server.chat_format = None
    server.start()
    res = _say_hi()
    assert res.status_code == 200, res.body
