from anvil.agents.coding_agent import CodingAgent
from anvil.agents.planner import Planner
from anvil.config import AgentSettings
from anvil.llm.client import LLMUnavailableError
from anvil.models import (
    FinalEvent,
    PlanEvent,
    RunStatus,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from anvil.tools.filesystem import ListFiles, ReadFile, WriteFile
from anvil.tools.registry import ToolRegistry
from anvil.tools.workspace import Workspace
from tests.fakes import FakeLLM, make_ctx, reply

REGISTRY = ToolRegistry([ListFiles(), ReadFile(), WriteFile()])


def agent(llm: FakeLLM, planner: Planner | None = None, **settings: object) -> CodingAgent:
    return CodingAgent(llm, REGISTRY, AgentSettings.model_validate(settings), planner)


async def test_answers_without_tools(workspace: Workspace) -> None:
    llm = FakeLLM([reply("Hello there")])
    events = [e async for e in agent(llm).run_stream("hi", make_ctx(workspace))]
    assert [type(e) for e in events] == [TextDeltaEvent, TextDeltaEvent, FinalEvent]
    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.result.status is RunStatus.COMPLETED
    assert final.result.answer == "Hello there"
    assert final.result.usage_by_model["fake-model"].input_tokens == 10


async def test_tool_loop_feeds_wrapped_output_back(workspace: Workspace) -> None:
    llm = FakeLLM([reply(None, ("read_file", '{"path": "main.py"}')), reply("It is a calculator.")])
    result = await agent(llm).run("what is main.py?", make_ctx(workspace))
    assert result.status is RunStatus.COMPLETED
    assert result.steps == 2
    assert [c.name for c in result.tool_calls] == ["read_file"]
    tool_message = llm.requests[1][-1]
    assert tool_message.role == "tool"
    assert tool_message.tool_call_id == "call_0"
    assert (tool_message.content or "").startswith('<tool_output tool="read_file">')
    assert llm.tools_seen[0] is not None
    assert {t["function"]["name"] for t in llm.tools_seen[0]} == {
        "list_files",
        "read_file",
        "write_file",
    }


async def test_wrapping_can_be_disabled(workspace: Workspace) -> None:
    llm = FakeLLM([reply(None, ("list_files", "{}")), reply("done")])
    await agent(llm, wrap_tool_output=False).run("ls", make_ctx(workspace))
    assert not (llm.requests[1][-1].content or "").startswith("<tool_output")


async def test_stops_at_max_steps(workspace: Workspace) -> None:
    llm = FakeLLM([reply("still looking", ("list_files", "{}")) for _ in range(5)])
    result = await agent(llm, max_steps=3).run("loop forever", make_ctx(workspace))
    assert result.status is RunStatus.MAX_STEPS
    assert result.steps == 3
    assert result.answer == "still looking"
    assert len(llm.requests) == 3


async def test_llm_outage_ends_with_error_result(workspace: Workspace) -> None:
    llm = FakeLLM([reply(None, ("list_files", "{}")), LLMUnavailableError("down")])
    result = await agent(llm).run("hi", make_ctx(workspace))
    assert result.status is RunStatus.ERROR
    assert result.error == "down"
    assert result.steps == 1


async def test_model_mistakes_do_not_crash_the_loop(workspace: Workspace) -> None:
    llm = FakeLLM(
        [
            reply(None, ("no_such_tool", "{}"), ("read_file", "{not json")),
            reply("recovered"),
        ]
    )
    events = [e async for e in agent(llm).run_stream("x", make_ctx(workspace))]
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert [r.ok for r in results] == [False, False]
    assert len([e for e in events if isinstance(e, ToolCallEvent)]) == 2
    assert isinstance(events[-1], FinalEvent)
    assert events[-1].result.answer == "recovered"


async def test_denied_tool_is_withdrawn_for_the_rest_of_the_run(workspace: Workspace) -> None:
    write = ("write_file", '{"path": "main.py", "content": "x"}')
    llm = FakeLLM([reply(None, write), reply(None, write), reply("I could not.")])
    asked: list[str] = []

    class CountingDeny:
        async def approve(self, tool: str, arguments: dict[str, object]) -> bool:
            asked.append(tool)
            return False

    result = await agent(llm).run("edit", make_ctx(workspace, CountingDeny()))
    assert [c.error_type for c in result.tool_calls] == ["denied", "denied"]
    assert asked == ["write_file"]  # the user is not asked again
    assert "disabled for the rest of this task" in (llm.requests[1][-1].content or "")
    offered = [{t["function"]["name"] for t in tools or []} for tools in llm.tools_seen]
    assert "write_file" in offered[0]
    assert "write_file" not in offered[1]
    assert "Calculator" in (workspace.root / "main.py").read_text()


async def test_history_is_sent_before_the_task(workspace: Workspace) -> None:
    from anvil.models import Message

    llm = FakeLLM([reply("ok")])
    history = [Message(role="user", content="earlier"), Message(role="assistant", content="sure")]
    await agent(llm).run("now", make_ctx(workspace), history)
    roles = [(m.role, m.content) for m in llm.requests[0][1:]]
    assert roles == [("user", "earlier"), ("assistant", "sure"), ("user", "now")]


async def test_planner_adds_plan(workspace: Workspace) -> None:
    fast = FakeLLM([reply('{"steps": ["list files", "read main.py"]}')], model="fast-model")
    main = FakeLLM([reply("done")])
    planner = Planner(fast, REGISTRY, "v1")
    events = [e async for e in agent(main, planner).run_stream("task", make_ctx(workspace))]
    assert isinstance(events[0], PlanEvent)
    assert events[0].steps == ["list files", "read main.py"]
    assert "- read main.py" in (main.requests[0][-1].content or "")
    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert set(final.result.usage_by_model) == {"fast-model", "fake-model"}


async def test_planner_failure_fails_open(workspace: Workspace) -> None:
    fast = FakeLLM([reply("this is not json")])
    main = FakeLLM([reply("done")])
    result = await agent(main, Planner(fast, REGISTRY, "v1")).run("task", make_ctx(workspace))
    assert result.status is RunStatus.COMPLETED
    assert result.plan is None
    assert main.requests[0][-1].content == "task"
