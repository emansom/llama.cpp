import pytest
from openai import OpenAI
from utils import *

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()

def test_responses_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_output_tokens=8,
        temperature=0.8,
    )
    assert res.id.startswith("resp_")
    assert res.output[0].id is not None
    assert res.output[0].id.startswith("msg_")
    assert match_regex("(Suddenly)+", res.output_text)

def test_responses_stream_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    stream = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_output_tokens=8,
        temperature=0.8,
        stream=True,
    )

    gathered_text = ''
    resp_id = ''
    msg_id = ''
    for r in stream:
        if r.type == "response.created":
            assert r.response.id.startswith("resp_")
            resp_id = r.response.id
        if r.type == "response.in_progress":
            assert r.response.id == resp_id
        if r.type == "response.output_item.added":
            assert r.item.id is not None
            assert r.item.id.startswith("msg_")
            msg_id = r.item.id
        if (r.type == "response.content_part.added" or
            r.type == "response.output_text.delta" or
            r.type == "response.output_text.done" or
            r.type == "response.content_part.done"):
            assert r.item_id == msg_id
        if r.type == "response.output_item.done":
            assert r.item.id == msg_id

        if r.type == "response.output_text.delta":
            gathered_text += r.delta
        if r.type == "response.completed":
            assert r.response.id.startswith("resp_")
            assert r.response.output[0].id is not None
            assert r.response.output[0].id.startswith("msg_")
            assert gathered_text == r.response.output_text
            assert match_regex("(Suddenly)+", r.response.output_text)


def test_responses_stream_with_llama_telemetry():
    global server
    server.n_ctx = 256
    server.n_batch = 32
    server.n_slots = 1
    server.start()

    saw_progress = False
    saw_delta_timings = False
    completed = None

    res = server.make_stream_request("POST", "/responses", data={
        "input": "This is a test" * 10,
        "max_output_tokens": 8,
        "temperature": 0.8,
        "stream": True,
        "timings_per_token": True,
        "return_progress": True,
    })

    for data in res:
        if "prompt_progress" in data:
            assert data["type"] == "response.in_progress"
            assert data["prompt_progress"]["total"] > 0
            assert data["prompt_progress"]["processed"] >= data["prompt_progress"]["cache"]
            saw_progress = True
        if "timings" in data:
            assert "prompt_per_second" in data["timings"]
            assert "predicted_per_second" in data["timings"]
            if data["type"] == "response.output_text.delta":
                saw_delta_timings = True
        if data["type"] == "response.completed":
            completed = data

    assert saw_progress
    assert saw_delta_timings
    assert completed is not None
    assert "usage" in completed["response"]
    assert "timings" in completed


def test_responses_reports_truncation_as_incomplete():
    """A token-cap stop must be reported as incomplete, not as a finished answer.

    status was hardcoded to "completed", so a severed generation was
    indistinguishable from a whole one -- the information Chat Completions has
    always carried as finish_reason "length" was simply dropped here. That is
    worse than an error: half a JSON object still parses as data, and half an
    answer still reads like an answer.
    """
    global server
    server = ServerPreset.gemma4()
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.responses.create(
        model=server.model_alias,
        input=[{"role": "user", "content": "Write a very long story about a book"}],
        max_output_tokens=8,
    )
    assert res.status == "incomplete"
    assert res.incomplete_details is not None
    assert res.incomplete_details.reason == "max_output_tokens"
    # The cap really was the reason, rather than the model happening to stop.
    assert res.usage.output_tokens == 8


def test_responses_reports_a_natural_stop_as_completed():
    """The other half of the pair: a generation that ends on its own must not
    be labelled incomplete, or the status is noise rather than signal."""
    global server
    server = ServerPreset.gemma4()
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.responses.create(
        model=server.model_alias,
        input=[{"role": "user", "content": "Hello"}],
        max_output_tokens=200,
    )
    assert res.status == "completed"
    assert res.incomplete_details is None
    assert res.usage.output_tokens < 200


def test_responses_stream_reports_truncation_as_incomplete():
    """The streaming form carries its own copy of the response payload, so it
    needs the same status -- a client must not have to know which form it asked
    for to learn that the output was severed."""
    global server
    server = ServerPreset.gemma4()
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    stream = client.responses.create(
        model=server.model_alias,
        input=[{"role": "user", "content": "Write a very long story about a book"}],
        max_output_tokens=8,
        stream=True,
    )
    final = None
    for event in stream:
        if event.type == "response.completed":
            final = event.response
    assert final is not None
    assert final.status == "incomplete"
    assert final.incomplete_details.reason == "max_output_tokens"


def _has_reasoning(res) -> bool:
    return any(item.type == "reasoning" and item.content for item in res.output)


def test_responses_reasoning_effort_pins_thinking_both_ways():
    """reasoning_effort is the OAI-spec spelling of the thinking toggle, and a
    request must be able to set it in BOTH directions.

    Only "none" used to be read, so a client could turn reasoning off but not
    on: asking for it inherited whatever --reasoning the server was started
    with, and there was no spec-compliant way to override that. This server is
    started with reasoning OFF (see ServerPreset.gemma4), so an effort level
    that produces a thought channel here can only have come from the request.

    The DEGREE stays unhandled on purpose -- llama.cpp has no notion of how
    much a model should think. Whether to think at all is not model-specific.
    """
    global server
    server = ServerPreset.gemma4()
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")

    def ask(**kwargs):
        return client.responses.create(
            model=server.model_alias,
            input=[{"role": "user", "content": "Is 91 prime?"}],
            max_output_tokens=200,
            **kwargs,
        )

    assert _has_reasoning(ask(reasoning={"effort": "medium"})), \
        "an effort level must turn thinking on despite --reasoning off"
    assert not _has_reasoning(ask(reasoning={"effort": "none"})), \
        "effort none must turn thinking off"
    assert not _has_reasoning(ask()), \
        "with no effort stated the server default (off) must stand"
