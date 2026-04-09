"""AgentLoop — multi-turn tool-call conversation orchestrator.

This implements the core agentic loop that makes local models useful:

    User prompt → Model → tool_call → Execute → Result → Model → tool_call → ...

The loop continues until the model produces a final text response (no more tool
calls) or max_turns is reached. This is the same pattern used by Claude Code,
ChatGPT plugins, and other tool-using LLM systems.

KEY SECURITY CONSTRAINT: AgentLoop calls SecurityGateway.execute_tool() for ALL
tool execution. It NEVER calls MCPClientManager directly. This is enforced by
architecture — AgentLoop doesn't even have a reference to MCPClientManager.
SecurityGateway.execute_tool() is the atomic operation that validates parameters,
checks gates, calls MCP, sanitizes results, and writes the audit log.

ERROR PHILOSOPHY: The loop never crashes from tool execution failures. If a tool
is blocked, denied, rate-limited, or errors out, the error is converted to a
tool result message and fed back to the model. The model then decides what to do
(try a different tool, ask the user, give up gracefully). Only infrastructure
failures (Ollama unreachable) propagate as exceptions.

STREAMING: Uses Ollama's native /api/chat endpoint (NOT /v1/chat/completions).
The /v1 endpoint silently drops tool_calls when streaming — a confirmed Ollama bug.
The native endpoint handles streaming + tool calls correctly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncIterator, Callable, Awaitable

from .errors import (
    BridgeError,
    ConfirmationDeniedError,
    LoopError,
    MaxTurnsError,
    MCPToolError,
    ParameterRejectedError,
    RateLimitError,
    StuckModelError,
    ToolBlockedError,
)
from .ollama_client import OllamaClient
from .security import SecurityGateway
from .translator import ToolTranslator
from .types import (
    ApprovedTool,
    BridgeResult,
    OllamaToolCall,
    StreamEvent,
    StreamEventType,
    ToolCallRecord,
    ToolSignalCode,
)

logger = logging.getLogger(__name__)

# Maximum correction retries for malformed tool calls
MAX_CORRECTION_RETRIES = 2
# Maximum nudges for stuck model
MAX_NUDGES = 1


class AgentLoop:
    """Multi-turn tool-call loop with security enforcement.

    All tool execution goes through SecurityGateway.execute_tool() — atomic
    validate → gate → rate-check → call → sanitize → audit pipeline.
    """

    def __init__(
        self,
        ollama: OllamaClient,
        security: SecurityGateway,
        translator: ToolTranslator,
        max_turns: int = 10,
        max_turns_hard_cap: int = 50,
    ):
        self._ollama = ollama
        self._security = security
        self._translator = translator
        self._max_turns = min(max_turns, max_turns_hard_cap)

    @staticmethod
    def _get_tool_schema(
        tool_call: OllamaToolCall, approved: list[ApprovedTool]
    ) -> dict[str, Any]:
        """Get the input schema for a tool call, for correction hints.

        Used in ADR[8] retry-with-correction: when parameter validation fails,
        we include the expected schema in the error message so the model can
        self-correct on the next turn.
        """
        for tool in approved:
            if tool.namespaced_name == tool_call.function_name or tool.name == tool_call.tool_name:
                return tool.input_schema
        return {}

    async def execute(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
        on_event: Callable[[StreamEvent], Awaitable[None]] | None = None,
        trace_id: str = "",
    ) -> BridgeResult:
        """Execute a multi-turn tool-call loop.

        Args:
            prompt: User prompt.
            model: Ollama model name.
            system_prompt: Optional system prompt.
            on_event: Optional callback for streaming events.
            trace_id: Correlation ID assigned by Bridge.run() — propagated to all
                ToolCallRecords and the final BridgeResult so callers can correlate
                all tool calls belonging to a single Bridge.run() invocation.

        Returns:
            BridgeResult with final response and tool call records.
        """
        # Build initial messages
        messages = ToolTranslator.to_ollama_messages(system_prompt, prompt)

        # Get approved tools in Ollama format
        approved = self._security.get_approved_tools()
        ollama_tools = self._translator.to_ollama_tools(approved) if approved else None

        tool_records: list[ToolCallRecord] = []
        nudge_count = 0

        for turn in range(self._max_turns):
            # Send to Ollama
            try:
                response = await self._ollama.chat(
                    model=model,
                    messages=messages,
                    tools=ollama_tools,
                )
            except BridgeError:
                raise
            except Exception as e:
                raise LoopError(f"Ollama chat failed on turn {turn}: {e}") from e

            # Check for tool calls
            tool_calls = OllamaClient.extract_tool_calls(response)

            if not tool_calls:
                # No tool calls — check if model produced content
                content = response.message.content if response.message else ""
                if content:
                    if on_event:
                        await on_event(StreamEvent(
                            type=StreamEventType.TEXT,
                            content=content,
                        ))
                    return BridgeResult(
                        content=content,
                        tool_calls=tool_records,
                        model=model,
                        turns=turn + 1,
                        trace_id=trace_id,
                    )

                # Empty response, no tool calls — model is stuck
                nudge_count += 1
                if nudge_count > MAX_NUDGES:
                    raise StuckModelError(
                        f"Model returned empty response with no tool calls "
                        f"after {nudge_count} nudges"
                    )
                messages.append({
                    "role": "user",
                    "content": (
                        "You didn't provide a response or call any tools. "
                        "Please either answer the question or use one of the available tools."
                    ),
                })
                continue

            # CQ-R1 FIX: Append assistant message BEFORE tool results.
            # Ollama expects message ordering: assistant (with tool_calls) → tool results.
            # The assistant message records what the model said/requested; the tool
            # result messages are the bridge's responses to those requests.
            if response.message:
                messages.append({
                    "role": "assistant",
                    "content": response.message.content or "",
                    "tool_calls": [
                        {"function": {"name": tc.function_name, "arguments": tc.arguments}}
                        for tc in tool_calls
                    ] if tool_calls else None,
                })

            # Process each tool call from the model's response.
            # Small models (4B-8B) frequently hallucinate tool names that don't exist,
            # especially when many tools are available. Each error is fed back to the
            # model as a tool result so it can self-correct.
            for tc in tool_calls:
                # Parse the tool call: resolve namespace, match against approved tools.
                # Returns None if the model hallucinated a tool name.
                parsed = self._translator.parse_tool_call(tc, approved)

                if parsed is None:
                    # Unknown tool — model hallucinated a tool name.
                    available = [t.namespaced_name for t in approved]
                    error_msg = (
                        f"Unknown tool '{tc.function_name}'. "
                        f"Available tools: {', '.join(available)}"
                    )
                    messages.append(
                        ToolTranslator.to_tool_result_message(tc.function_name, error_msg)
                    )
                    tool_records.append(ToolCallRecord(
                        server="",
                        tool_name=tc.function_name,
                        arguments=tc.arguments,
                        result_summary=error_msg[:200],
                        blocked=True,
                        block_reason="unknown_tool",
                        signal=ToolSignalCode.FAILURE,
                        trace_id=trace_id,
                    ))
                    if on_event:
                        await on_event(StreamEvent(
                            type=StreamEventType.ERROR,
                            error=error_msg,
                            tool=tc.function_name,
                        ))
                    continue

                server, tool_name, args = parsed

                if on_event:
                    await on_event(StreamEvent(
                        type=StreamEventType.TOOL_CALL,
                        tool=tool_name,
                        server=server,
                        content=json.dumps(args)[:200],
                    ))

                # Execute through SecurityGateway's atomic pipeline.
                # This single call handles: permission check, parameter validation,
                # human confirmation (if destructive), rate limiting, MCP execution,
                # result sanitization, and audit logging. We never call MCP directly.
                try:
                    result = await self._security.execute_tool(
                        tc, model_id=model, turn=turn
                    )
                    messages.append(
                        ToolTranslator.to_tool_result_message(
                            tc.function_name, result.content
                        )
                    )
                    tool_records.append(ToolCallRecord(
                        server=result.server,
                        tool_name=result.tool_name,
                        arguments=args,
                        result_summary=result.content[:200],
                        duration_ms=result.duration_ms,
                        signal=ToolSignalCode.SUCCESS,
                        trace_id=trace_id,
                    ))
                    if on_event:
                        await on_event(StreamEvent(
                            type=StreamEventType.TOOL_RESULT,
                            tool=tool_name,
                            server=server,
                            content=result.content[:200],
                        ))

                except ToolBlockedError as e:
                    msg = f"BLOCKED: {e.reason}"
                    messages.append(
                        ToolTranslator.to_tool_result_message(tc.function_name, msg)
                    )
                    tool_records.append(ToolCallRecord(
                        server=server,
                        tool_name=tool_name,
                        arguments=args,
                        blocked=True,
                        block_reason=e.reason,
                        signal=ToolSignalCode.FAILURE,
                        trace_id=trace_id,
                    ))

                except ConfirmationDeniedError:
                    msg = "DENIED by user"
                    messages.append(
                        ToolTranslator.to_tool_result_message(tc.function_name, msg)
                    )
                    tool_records.append(ToolCallRecord(
                        server=server,
                        tool_name=tool_name,
                        arguments=args,
                        blocked=True,
                        block_reason="user_denied",
                        signal=ToolSignalCode.FAILURE,
                        trace_id=trace_id,
                    ))

                except ParameterRejectedError as e:
                    # ADR[8]: Retry-with-correction — give the model the schema
                    # so it can fix its parameters on the next turn. This is more
                    # helpful than a bare error message because small models often
                    # malform parameters and can self-correct when shown the schema.
                    schema_hint = json.dumps(
                        self._get_tool_schema(tc, approved), indent=2
                    )[:500]
                    msg = (
                        f"Parameter error: {'; '.join(e.validation_errors)}. "
                        f"Expected format: {schema_hint}"
                    )
                    messages.append(
                        ToolTranslator.to_tool_result_message(tc.function_name, msg)
                    )
                    tool_records.append(ToolCallRecord(
                        server=server,
                        tool_name=tool_name,
                        arguments=args,
                        blocked=True,
                        block_reason="parameter_rejected",
                        signal=ToolSignalCode.INVALID_STATE,
                        trace_id=trace_id,
                    ))

                except RateLimitError as e:
                    # ADR[8]: Auto-backoff — sleep for the retry period before
                    # continuing the loop. This gives the rate limiter's sliding
                    # window time to clear. Without this, the model would just
                    # immediately hit the limit again on the next turn.
                    backoff = min(e.retry_after_seconds, 5.0)  # cap at 5s
                    if backoff > 0:
                        await asyncio.sleep(backoff)
                    msg = (
                        f"Rate limit exceeded, waited {backoff:.0f}s. "
                        f"Please reduce call frequency."
                    )
                    messages.append(
                        ToolTranslator.to_tool_result_message(tc.function_name, msg)
                    )
                    tool_records.append(ToolCallRecord(
                        server=server,
                        tool_name=tool_name,
                        arguments=args,
                        blocked=True,
                        block_reason="rate_limited",
                        signal=ToolSignalCode.TIMEOUT,
                        trace_id=trace_id,
                    ))

                except MCPToolError as e:
                    msg = f"ERROR: {e.safe_message}"
                    messages.append(
                        ToolTranslator.to_tool_result_message(tc.function_name, msg)
                    )
                    tool_records.append(ToolCallRecord(
                        server=server,
                        tool_name=tool_name,
                        arguments=args,
                        result_summary=msg[:200],
                        signal=ToolSignalCode.FAILURE,
                        trace_id=trace_id,
                    ))

        # Max turns reached — signal RECOVERY_REQUIRED on the last record if present
        if tool_records:
            last = tool_records[-1]
            tool_records[-1] = last.model_copy(
                update={"signal": ToolSignalCode.RECOVERY_REQUIRED}
            )
        return BridgeResult(
            content="Maximum turns reached. Partial results may be available in tool_calls.",
            tool_calls=tool_records,
            model=model,
            turns=self._max_turns,
            truncated=True,
            trace_id=trace_id,
        )

    async def execute_stream(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Execute with live streaming events.

        Yields StreamEvent objects as they happen during the loop, not after.
        A background task runs execute() and pushes events into an asyncio.Queue.
        The caller receives events incrementally as tools are called and results
        arrive. A DONE event is always emitted last (on success or failure).
        """
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        worker_error: list[BaseException] = []

        async def push_event(event: StreamEvent) -> None:
            await queue.put(event)

        async def run_worker() -> None:
            try:
                result = await self.execute(
                    prompt=prompt,
                    model=model,
                    system_prompt=system_prompt,
                    on_event=push_event,
                )
                await queue.put(StreamEvent(
                    type=StreamEventType.DONE,
                    content=result.content,
                ))
            except BaseException as exc:
                worker_error.append(exc)
                await queue.put(StreamEvent(
                    type=StreamEventType.ERROR,
                    error=str(exc),
                ))
            finally:
                await queue.put(None)  # sentinel

        task = asyncio.create_task(run_worker())

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if worker_error:
            raise worker_error[0]
