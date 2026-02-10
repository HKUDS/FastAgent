from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastagent.agents.base import BaseAgent
from fastagent.grounding.core.types import BackendType, ToolResult
from fastagent.platform.screenshot import ScreenshotClient
from fastagent.prompts import GroundingAgentPrompts
from fastagent.utils.logging import Logger

if TYPE_CHECKING:
    from fastagent.llm import LLMClient
    from fastagent.agents.coordinator import AgentCoordinator
    from fastagent.grounding.core.grounding_client import GroundingClient
    from fastagent.recording import RecordingManager

logger = Logger.get_logger(__name__)


class GroundingAgent(BaseAgent):
    def __init__(
        self,
        name: str = "GroundingAgent",
        backend_scope: Optional[List[str]] = None,
        llm_client: Optional[LLMClient] = None,
        coordinator: Optional[AgentCoordinator] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 15,
        visual_analysis_timeout: float = 30.0,
        tool_retrieval_llm: Optional[LLMClient] = None,
        visual_analysis_model: Optional[str] = None,
    ) -> None:
        """
        Initialize the Grounding Agent.

        Args:
            name: Agent name
            backend_scope: List of backends this agent can access (None = all available)
            llm_client: LLM client for reasoning
            coordinator: AgentCoordinator for resource access
            system_prompt: Custom system prompt
            max_iterations: Maximum LLM reasoning iterations for self-correction
            visual_analysis_timeout: Timeout for visual analysis LLM calls in seconds
            tool_retrieval_llm: LLM client for tool retrieval filter (None = use llm_client)
            visual_analysis_model: Model name for visual analysis (None = use llm_client.model)
        """
        super().__init__(
            name=name,
            backend_scope=backend_scope or ["gui", "shell", "mcp", "web", "system"],
            llm_client=llm_client,
            coordinator=coordinator,
        )

        self._system_prompt = system_prompt or self._default_system_prompt()
        self._max_iterations = max_iterations
        self._visual_analysis_timeout = visual_analysis_timeout
        self._tool_retrieval_llm = tool_retrieval_llm
        self._visual_analysis_model = visual_analysis_model

        logger.info(f"Grounding Agent initialized: {name}")
        logger.info(f"Backend scope: {self._backend_scope}")
        logger.info(f"Max iterations: {self._max_iterations}")
        logger.info(f"Visual analysis timeout: {self._visual_analysis_timeout}s")
        if tool_retrieval_llm:
            logger.info(f"Tool retrieval model: {tool_retrieval_llm.model}")
        if visual_analysis_model:
            logger.info(f"Visual analysis model: {visual_analysis_model}")

    @property
    def grounding_client(self) -> Optional[GroundingClient]:
        """Get the grounding client from coordinator."""
        return self.get_grounding_client()

    @property
    def recording_manager(self) -> Optional[RecordingManager]:
        """Get the recording manager from coordinator."""
        if self._coordinator:
            return self._coordinator.recording_manager
        return None

    def _default_system_prompt(self) -> str:
        """Default system prompt for the grounding agent."""
        return GroundingAgentPrompts.SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Message truncation (prevent context overflow in long iterations)
    # ------------------------------------------------------------------

    def _truncate_messages(
        self,
        messages: List[Dict[str, Any]],
        keep_recent: int = 8,
        max_tokens_estimate: int = 120000,
    ) -> List[Dict[str, Any]]:
        """Truncate message history to prevent context length issues.

        Keeps system messages, the first user instruction, and the most recent
        *keep_recent* rounds of conversation.
        """
        if len(messages) <= keep_recent + 2:  # +2 for system and initial user
            return messages

        total_text = json.dumps(messages, ensure_ascii=False)
        estimated_tokens = len(total_text) // 4

        if estimated_tokens < max_tokens_estimate:
            return messages

        logger.info(
            f"Truncating message history: {len(messages)} messages, "
            f"~{estimated_tokens:,} tokens -> keeping recent {keep_recent} rounds"
        )

        system_messages: List[Dict] = []
        user_instruction: Optional[Dict] = None
        conversation_messages: List[Dict] = []

        for msg in messages:
            role = msg.get("role")
            if role == "system":
                system_messages.append(msg)
            elif role == "user" and user_instruction is None:
                user_instruction = msg
            else:
                conversation_messages.append(msg)

        recent_messages = (
            conversation_messages[-(keep_recent * 2):]
            if conversation_messages
            else []
        )

        truncated = system_messages.copy()
        if user_instruction:
            truncated.append(user_instruction)
        truncated.extend(recent_messages)

        logger.info(
            f"After truncation: {len(truncated)} messages, "
            f"~{len(json.dumps(truncated, ensure_ascii=False)) // 4:,} tokens (estimated)"
        )

        return truncated

    # ------------------------------------------------------------------
    # Multi-round process
    # ------------------------------------------------------------------

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task execution request with multi-round iteration control.

        The agent iterates up to *max_iterations* times.  Each iteration the
        LLM may call tools or emit the ``<COMPLETE>`` token to signal the task
        is done.
        """
        instruction = context.get("instruction", "")
        if not instruction:
            logger.error("Grounding Agent: No instruction provided")
            return {"error": "No instruction provided", "status": "error"}

        # Store current instruction for visual analysis context
        self._current_instruction = instruction

        logger.info(f"Grounding Agent: Processing instruction at step {self.step}")

        # Check existing workspace files
        workspace_info = await self._check_workspace_artifacts(context)
        if workspace_info["has_files"]:
            context["workspace_artifacts"] = workspace_info
            logger.info(
                f"Workspace has {len(workspace_info['files'])} existing files: "
                f"{workspace_info['files']}"
            )

        # Get available tools (auto-search with cap)
        tools = await self._get_available_tools(instruction)

        # Build retrieved tools list for return value
        retrieved_tools_list = []
        for tool in tools:
            tool_info = {
                "name": getattr(tool, "name", str(tool)),
                "description": getattr(tool, "description", ""),
            }
            if hasattr(tool, "backend_type"):
                tool_info["backend"] = (
                    tool.backend_type.value
                    if hasattr(tool.backend_type, "value")
                    else str(tool.backend_type)
                )
            if hasattr(tool, "_runtime_info") and tool._runtime_info:
                tool_info["server_name"] = tool._runtime_info.server_name
            retrieved_tools_list.append(tool_info)

        # Initialize iteration state
        max_iterations = context.get("max_iterations", self._max_iterations)
        current_iteration = 0
        all_tool_results: List[Dict] = []
        iteration_contexts: List[Dict] = []
        consecutive_empty_responses = 0
        MAX_CONSECUTIVE_EMPTY = 5

        # Build initial messages
        messages = self.construct_messages(context)

        try:
            while current_iteration < max_iterations:
                current_iteration += 1
                logger.info(
                    f"Grounding Agent: Iteration {current_iteration}/{max_iterations}"
                )

                # Truncate message history after 5 iterations
                if current_iteration >= 5:
                    messages = self._truncate_messages(
                        messages,
                        keep_recent=8,
                        max_tokens_estimate=120000,
                    )

                messages_input_snapshot = copy.deepcopy(messages)

                # Call LLMClient for single round with visual analysis callback
                llm_response = await self._llm_client.complete(
                    messages=messages,
                    tools=tools if context.get("auto_execute", True) else None,
                    execute_tools=context.get("auto_execute", True),
                    summary_prompt=None,
                    tool_result_callback=self._visual_analysis_callback,
                )

                # Update messages with LLM response
                messages = llm_response["messages"]

                # Collect tool results
                tool_results_this_iteration = llm_response.get("tool_results", [])
                if tool_results_this_iteration:
                    all_tool_results.extend(tool_results_this_iteration)

                assistant_message = llm_response.get("message", {})
                assistant_content = assistant_message.get("content", "")

                has_tool_calls = llm_response.get("has_tool_calls", False)
                logger.info(
                    f"Iteration {current_iteration} - Has tool calls: {has_tool_calls}, "
                    f"Tool results: {len(tool_results_this_iteration)}, "
                    f"Content length: {len(assistant_content)} chars"
                )

                if len(assistant_content) > 0:
                    logger.info(
                        f"Iteration {current_iteration} - Assistant content preview: "
                        f"{repr(assistant_content[:300])}"
                    )
                    consecutive_empty_responses = 0
                else:
                    if not has_tool_calls:
                        consecutive_empty_responses += 1
                        logger.warning(
                            f"Iteration {current_iteration} - NO tool calls and NO content "
                            f"(empty response {consecutive_empty_responses}/{MAX_CONSECUTIVE_EMPTY})"
                        )
                        if consecutive_empty_responses >= MAX_CONSECUTIVE_EMPTY:
                            logger.error(
                                f"Exiting due to {MAX_CONSECUTIVE_EMPTY} consecutive "
                                "empty LLM responses. This may indicate API issues, "
                                "rate limiting, or context too long."
                            )
                            break
                    else:
                        consecutive_empty_responses = 0

                messages_output_snapshot = copy.deepcopy(messages)

                # Record iteration context
                iteration_context = {
                    "iteration": current_iteration,
                    "messages_input": messages_input_snapshot,
                    "messages_output": messages_output_snapshot,
                    "llm_response_summary": {
                        "assistant_content": assistant_content,
                        "has_tool_calls": has_tool_calls,
                        "tool_calls_count": len(tool_results_this_iteration),
                    },
                }
                iteration_contexts.append(iteration_context)

                # Check for completion token in assistant content
                is_complete = GroundingAgentPrompts.TASK_COMPLETE in assistant_content

                if is_complete:
                    logger.info(
                        f"Task completed at iteration {current_iteration} "
                        f"(found {GroundingAgentPrompts.TASK_COMPLETE})"
                    )
                    break
                else:
                    if tool_results_this_iteration:
                        logger.debug(
                            f"Task in progress, LLM called "
                            f"{len(tool_results_this_iteration)} tools"
                        )
                    else:
                        logger.debug(
                            "Task in progress, LLM did not generate <COMPLETE>"
                        )

                    # Remove previous iteration guidance to avoid accumulation
                    messages = [
                        msg
                        for msg in messages
                        if not (
                            msg.get("role") == "system"
                            and "Iteration" in msg.get("content", "")
                            and "complete" in msg.get("content", "")
                        )
                    ]

                    guidance_msg = {
                        "role": "system",
                        "content": (
                            f"Iteration {current_iteration} complete. "
                            f"Check if task is finished - if yes, output "
                            f"{GroundingAgentPrompts.TASK_COMPLETE}. "
                            f"If not, continue with next action."
                        ),
                    }
                    messages.append(guidance_msg)
                    continue

            # Build final result
            result = await self._build_final_result(
                instruction=instruction,
                messages=messages,
                all_tool_results=all_tool_results,
                iterations=current_iteration,
                max_iterations=max_iterations,
                iteration_contexts=iteration_contexts,
                retrieved_tools_list=retrieved_tools_list,
            )

            # Record agent action to recording manager
            if self._coordinator and self._coordinator.recording_manager:
                await self._record_agent_execution(result, instruction)

            # Increment step
            self.increment_step()

            logger.info(
                f"Grounding Agent: Execution completed with status: "
                f"{result.get('status')}"
            )
            return result

        except Exception as e:
            logger.error(f"Grounding Agent: Execution failed: {e}")
            result = {
                "error": str(e),
                "status": "error",
                "instruction": instruction,
                "iteration": current_iteration,
            }
            self.increment_step()
            return result

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def construct_messages(
        self,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build the initial prompt messages from context."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt}
        ]

        instruction = context.get("instruction", "")
        if not instruction:
            raise ValueError("context must contain 'instruction' field")

        # Add workspace directory
        workspace_dir = context.get("workspace_dir")
        if workspace_dir:
            messages.append({
                "role": "system",
                "content": GroundingAgentPrompts.workspace_directory(workspace_dir),
            })

        # Add workspace artifacts information
        workspace_artifacts = context.get("workspace_artifacts")
        if workspace_artifacts and workspace_artifacts.get("has_files"):
            files = workspace_artifacts.get("files", [])
            matching_files = workspace_artifacts.get("matching_files", [])
            recent_files = workspace_artifacts.get("recent_files", [])

            if matching_files:
                artifact_msg = GroundingAgentPrompts.workspace_matching_files(
                    matching_files
                )
            elif len(recent_files) >= 2:
                artifact_msg = GroundingAgentPrompts.workspace_recent_files(
                    total_files=len(files),
                    recent_files=recent_files,
                )
            else:
                artifact_msg = GroundingAgentPrompts.workspace_file_list(files)

            messages.append({"role": "system", "content": artifact_msg})

        # User instruction
        messages.append({"role": "user", "content": instruction})

        return messages

    # ------------------------------------------------------------------
    # Tool retrieval (smart search + fallback)
    # ------------------------------------------------------------------

    async def _get_available_tools(
        self, task_description: Optional[str]
    ) -> List:
        """
        Retrieve tools with auto-search + cap to control prompt bloat.
        Falls back to returning all tools if search fails.
        """
        grounding_client = self.grounding_client
        if not grounding_client:
            return []

        backends = [BackendType(name) for name in self._backend_scope]

        try:
            retrieval_llm = self._tool_retrieval_llm or self._llm_client
            tools = await grounding_client.get_tools_with_auto_search(
                task_description=task_description,
                backend=backends,
                use_cache=True,
                llm_callable=retrieval_llm,
            )
            logger.info(
                f"GroundingAgent selected {len(tools)} tools (auto-search) "
                f"from {len(backends)} backends"
            )
            return tools
        except Exception as e:
            logger.warning(
                f"Auto-search tools failed, falling back to full list: {e}"
            )

        # Fallback: fetch all tools
        all_tools: List = []
        for backend_name in self._backend_scope:
            try:
                backend_type = BackendType(backend_name)
                tools = await grounding_client.list_tools(backend=backend_type)
                all_tools.extend(tools)
                logger.debug(
                    f"Retrieved {len(tools)} tools from backend: {backend_name}"
                )
            except Exception as e:
                logger.debug(f"Could not get tools from {backend_name}: {e}")

        logger.info(
            f"GroundingAgent fallback retrieved {len(all_tools)} tools "
            f"from {len(self._backend_scope)} backends"
        )
        return all_tools

    # ------------------------------------------------------------------
    # Visual analysis callback
    # ------------------------------------------------------------------

    async def _visual_analysis_callback(
        self,
        result: ToolResult,
        tool_name: str,
        tool_call: Dict,
        backend: str,
    ) -> ToolResult:
        """
        Callback for LLMClient to handle visual analysis after tool execution.
        Only applies to GUI backend results that contain screenshots.
        """
        # 1. Check if LLM requested to skip visual analysis
        skip_visual_analysis = False
        try:
            arguments = tool_call.function.arguments
            if isinstance(arguments, str):
                args = json.loads(arguments.strip() or "{}")
            else:
                args = arguments

            if isinstance(args, dict) and args.get("skip_visual_analysis"):
                skip_visual_analysis = True
                logger.info(
                    f"Visual analysis skipped for {tool_name} "
                    "(meta-parameter set by LLM)"
                )
        except Exception as e:
            logger.debug(f"Could not parse tool arguments: {e}")

        if skip_visual_analysis:
            return result

        # 2. Only apply to GUI backend
        if backend != "gui":
            return result

        # 3. Check if tool has visual data
        metadata = getattr(result, "metadata", None)
        has_screenshots = metadata and (
            metadata.get("screenshot") or metadata.get("screenshots")
        )

        # 4. If no visual data, try to capture a screenshot
        if not has_screenshots:
            try:
                logger.info(
                    f"No visual data from {tool_name}, capturing screenshot..."
                )
                screenshot_client = ScreenshotClient()
                screenshot_bytes = await screenshot_client.capture()

                if screenshot_bytes:
                    if metadata is None:
                        result.metadata = {}
                        metadata = result.metadata
                    metadata["screenshot"] = screenshot_bytes
                    has_screenshots = True
                    logger.info("Screenshot captured for visual analysis")
                else:
                    logger.warning("Failed to capture screenshot")
            except Exception as e:
                logger.warning(f"Error capturing screenshot: {e}")

        # 5. If still no screenshots, return original result
        if not has_screenshots:
            logger.debug(f"No visual data available for {tool_name}")
            return result

        # 6. Perform visual analysis
        return await self._enhance_result_with_visual_context(result, tool_name)

    async def _enhance_result_with_visual_context(
        self,
        result: ToolResult,
        tool_name: str,
    ) -> ToolResult:
        """Enhance tool result with visual analysis for grounding agent workflows."""
        import asyncio
        import base64
        import litellm

        try:
            metadata = getattr(result, "metadata", None)
            if not metadata:
                return result

            # Collect all screenshots
            screenshots_bytes: List[bytes] = []

            if metadata.get("screenshots"):
                screenshots_list = metadata["screenshots"]
                if isinstance(screenshots_list, list):
                    screenshots_bytes = [s for s in screenshots_list if s]
            elif metadata.get("screenshot"):
                screenshots_bytes = [metadata["screenshot"]]

            if not screenshots_bytes:
                return result

            # Select key screenshots if there are too many
            selected_screenshots = self._select_key_screenshots(
                screenshots_bytes, max_count=3
            )

            # Convert to base64
            visual_b64_list: List[str] = []
            for visual_data in selected_screenshots:
                if isinstance(visual_data, bytes):
                    visual_b64_list.append(
                        base64.b64encode(visual_data).decode("utf-8")
                    )
                else:
                    visual_b64_list.append(visual_data)

            num_screenshots = len(visual_b64_list)

            prompt = GroundingAgentPrompts.visual_analysis(
                tool_name=tool_name,
                num_screenshots=num_screenshots,
                task_description=getattr(self, "_current_instruction", ""),
            )

            content: List[Dict] = [{"type": "text", "text": prompt}]
            for visual_b64 in visual_b64_list:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{visual_b64}"},
                })

            visual_model = self._visual_analysis_model or (
                self._llm_client.model
                if self._llm_client
                else "openrouter/anthropic/claude-sonnet-4.5"
            )
            response = await asyncio.wait_for(
                litellm.acompletion(
                    model=visual_model,
                    messages=[{"role": "user", "content": content}],
                    timeout=self._visual_analysis_timeout,
                ),
                timeout=self._visual_analysis_timeout + 5,
            )

            analysis = response.choices[0].message.content.strip()

            original_content = result.content or "(no text output)"
            enhanced_content = (
                f"{original_content}\n\n**Visual content**: {analysis}"
            )

            enhanced_result = ToolResult(
                status=result.status,
                content=enhanced_content,
                error=result.error,
                metadata={
                    **metadata,
                    "visual_analyzed": True,
                    "visual_analysis": analysis,
                },
                execution_time=result.execution_time,
            )

            logger.info(
                f"Enhanced {tool_name} result with visual analysis "
                f"({num_screenshots} screenshot(s))"
            )
            return enhanced_result

        except asyncio.TimeoutError:
            logger.warning(
                f"Visual analysis timed out for {tool_name}, returning original result"
            )
            return result
        except Exception as e:
            logger.warning(
                f"Failed to analyze visual content for {tool_name}: {e}"
            )
            return result

    # ------------------------------------------------------------------
    # Screenshot selection helper
    # ------------------------------------------------------------------

    def _select_key_screenshots(
        self,
        screenshots: List[bytes],
        max_count: int = 3,
    ) -> List[bytes]:
        """Select key screenshots if there are too many."""
        if len(screenshots) <= max_count:
            return screenshots

        selected_indices: set = set()

        # Always include last (final state)
        selected_indices.add(len(screenshots) - 1)

        # If room, include first (initial state)
        if max_count >= 2:
            selected_indices.add(0)

        # Fill remaining slots with evenly spaced middle screenshots
        remaining_slots = max_count - len(selected_indices)
        if remaining_slots > 0:
            available_indices = [
                i
                for i in range(1, len(screenshots) - 1)
                if i not in selected_indices
            ]

            if available_indices:
                step = max(1, len(available_indices) // (remaining_slots + 1))
                for i in range(remaining_slots):
                    idx = min((i + 1) * step, len(available_indices) - 1)
                    if idx < len(available_indices):
                        selected_indices.add(available_indices[idx])

        selected = [screenshots[i] for i in sorted(selected_indices)]
        logger.debug(
            f"Selected {len(selected)} screenshots at indices "
            f"{sorted(selected_indices)} from total of {len(screenshots)}"
        )
        return selected

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------

    def _get_workspace_path(self, context: Dict[str, Any]) -> Optional[str]:
        """Get workspace directory path from context or recording manager."""
        workspace_path = context.get("workspace_dir")
        if workspace_path:
            return workspace_path

        # Try to get from recording manager
        if self._coordinator and self._coordinator.recording_manager:
            rm = self._coordinator.recording_manager
            if hasattr(rm, "workspace_dir"):
                return rm.workspace_dir

        return None

    def _scan_workspace_files(
        self,
        workspace_path: Optional[str],
        recent_threshold: int = 600,
    ) -> Dict[str, Any]:
        """Scan workspace directory and collect file information."""
        import os
        import time

        result: Dict[str, Any] = {
            "files": [],
            "file_details": {},
            "recent_files": [],
        }

        if not workspace_path or not os.path.exists(workspace_path):
            return result

        # Recording system files to exclude from workspace scanning
        excluded_files = {"metadata.json", "traj.jsonl"}

        try:
            current_time = time.time()

            for filename in os.listdir(workspace_path):
                filepath = os.path.join(workspace_path, filename)
                if os.path.isfile(filepath) and filename not in excluded_files:
                    result["files"].append(filename)
                    stat = os.stat(filepath)
                    file_info = {
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "age_seconds": current_time - stat.st_mtime,
                    }
                    result["file_details"][filename] = file_info
                    if file_info["age_seconds"] < recent_threshold:
                        result["recent_files"].append(filename)

            result["files"] = sorted(result["files"])

        except Exception as e:
            logger.debug(f"Error scanning workspace files: {e}")

        return result

    async def _check_workspace_artifacts(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check workspace directory for existing artifacts."""
        import re

        workspace_info: Dict[str, Any] = {
            "has_files": False,
            "files": [],
            "file_details": {},
            "recent_files": [],
        }

        try:
            workspace_path = self._get_workspace_path(context)
            scan_result = self._scan_workspace_files(
                workspace_path, recent_threshold=600
            )

            if scan_result["files"]:
                workspace_info["has_files"] = True
                workspace_info["files"] = scan_result["files"]
                workspace_info["file_details"] = scan_result["file_details"]
                workspace_info["recent_files"] = scan_result["recent_files"]

                logger.info(
                    f"Grounding Agent: Found {len(scan_result['files'])} "
                    f"existing files in workspace "
                    f"({len(scan_result['recent_files'])} recent)"
                )

                # Check if instruction mentions specific filenames
                instruction = context.get("instruction", "")
                if instruction:
                    potential_outputs = []
                    file_patterns = re.findall(
                        r'["\']?([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+)["\']?',
                        instruction,
                    )
                    for pattern in file_patterns:
                        if pattern in scan_result["files"]:
                            potential_outputs.append(pattern)

                    if potential_outputs:
                        workspace_info["matching_files"] = potential_outputs
                        logger.info(
                            f"Grounding Agent: Found {len(potential_outputs)} "
                            f"files matching task: {potential_outputs}"
                        )

        except Exception as e:
            logger.debug(f"Could not check workspace artifacts: {e}")

        return workspace_info

    # ------------------------------------------------------------------
    # Backend descriptions (used by HostAgent for planning context)
    # ------------------------------------------------------------------

    async def get_backend_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all available backends, their servers, and tools.

        For MCP backend, groups tools by server to help HostAgent understand
        which server provides which capabilities.
        """
        grounding_client = self.grounding_client
        if not grounding_client:
            logger.warning("Grounding Agent: No grounding client available")
            return {}

        descriptions: Dict[str, str] = {}

        try:
            for backend_name in self._backend_scope:
                try:
                    if backend_name == "system":
                        descriptions[backend_name] = (
                            "System tools for querying backend capabilities"
                        )
                        continue

                    backend_type = BackendType(backend_name)
                    tools = await grounding_client.list_tools(
                        backend=backend_type
                    )

                    if not tools:
                        descriptions[backend_name] = (
                            f"Backend '{backend_name}' available but no tools found"
                        )
                        continue

                    # Group tools by server
                    server_tools: Dict[Optional[str], List] = {}
                    for tool in tools:
                        server_name = (
                            getattr(tool.runtime_info, "server_name", None)
                            if hasattr(tool, "runtime_info")
                            else None
                        )
                        server_tools.setdefault(server_name, []).append(tool)

                    desc_lines: List[str] = []

                    good_for_map = {
                        "shell": "File operations, command execution, system tasks",
                        "gui": "GUI automation, browser control, visual interactions",
                        "mcp": "Application control via MCP servers",
                        "web": "Web search, information gathering",
                        "system": "Querying system capabilities",
                    }

                    if backend_name.lower() == "mcp":
                        desc_lines.append(
                            f"MCP Backend ({len(tools)} total tools across "
                            f"{len(server_tools)} servers)"
                        )
                        desc_lines.append(
                            f"Good for: {good_for_map.get(backend_name.lower(), 'Various tasks')}"
                        )

                        for srv_name, srv_tools in server_tools.items():
                            srv_display = srv_name or "<default>"
                            desc_lines.append(
                                f"\n  Server: {srv_display} ({len(srv_tools)} tools)"
                            )
                            tool_names = [t.schema.name for t in srv_tools]
                            desc_lines.append(
                                f"    Tools: {', '.join(tool_names[:8])}"
                            )
                            if len(tool_names) > 8:
                                desc_lines.append(
                                    f"           {', '.join(tool_names[8:16])}"
                                )
                                if len(tool_names) > 16:
                                    desc_lines.append(
                                        f"           (and {len(tool_names) - 16} more)"
                                    )
                            examples = srv_tools[:3]
                            if examples:
                                desc_lines.append("    Example capabilities:")
                                for t in examples:
                                    td = t.schema.description or "No description"
                                    if len(td) > 80:
                                        td = td[:77] + "..."
                                    desc_lines.append(
                                        f"      - {t.schema.name}: {td}"
                                    )
                    else:
                        desc_lines.append(
                            f"{backend_name.upper()} Backend ({len(tools)} tools)"
                        )
                        desc_lines.append(
                            f"Good for: {good_for_map.get(backend_name.lower(), 'Various tasks')}"
                        )
                        tool_names = [t.schema.name for t in tools]
                        desc_lines.append(
                            f"  Tools: {', '.join(tool_names[:10])}"
                        )
                        if len(tool_names) > 10:
                            desc_lines.append(
                                f"         {', '.join(tool_names[10:20])}"
                            )
                            if len(tool_names) > 20:
                                desc_lines.append(
                                    f"         (and {len(tool_names) - 20} more)"
                                )
                        examples = tools[:3]
                        if examples:
                            desc_lines.append("  Example capabilities:")
                            for t in examples:
                                td = t.schema.description or "No description"
                                if len(td) > 80:
                                    td = td[:77] + "..."
                                desc_lines.append(f"    - {t.schema.name}: {td}")

                    descriptions[backend_name] = "\n".join(desc_lines)
                    logger.debug(
                        f"Backend {backend_name}: {len(tools)} tools "
                        f"across {len(server_tools)} servers"
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to get description for backend {backend_name}: {e}"
                    )
                    descriptions[backend_name] = (
                        f"Backend available but description unavailable: {e}"
                    )

            logger.info(
                f"Grounding Agent: Retrieved descriptions for "
                f"{len(descriptions)} backends"
            )
            return descriptions

        except Exception as e:
            logger.error(
                f"Grounding Agent: Failed to get backend descriptions: {e}"
            )
            return {"error": f"Failed to retrieve backend descriptions: {e}"}

    # ------------------------------------------------------------------
    # Result building & helpers
    # ------------------------------------------------------------------

    async def _build_final_result(
        self,
        instruction: str,
        messages: List[Dict],
        all_tool_results: List[Dict],
        iterations: int,
        max_iterations: int,
        iteration_contexts: Optional[List[Dict]] = None,
        retrieved_tools_list: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Build final execution result after all iterations."""
        is_complete = self._check_task_completion(messages)
        tool_executions = self._format_tool_executions(all_tool_results)

        result: Dict[str, Any] = {
            "instruction": instruction,
            "step": self.step,
            "iterations": iterations,
            "tool_executions": tool_executions,
            "messages": messages,
            "iteration_contexts": iteration_contexts or [],
            "retrieved_tools_list": retrieved_tools_list or [],
            "keep_session": True,
        }

        if is_complete:
            logger.info("Task completed with <COMPLETE> marker")
            last_response = self._extract_last_assistant_message(messages)
            result["response"] = last_response.replace(
                GroundingAgentPrompts.TASK_COMPLETE, ""
            ).strip()
            result["status"] = "success"
        else:
            result["response"] = self._extract_last_assistant_message(messages)
            result["status"] = "incomplete"
            result["warning"] = (
                f"Task reached max iterations ({max_iterations}) without completion. "
                "This may indicate the task needs more steps or clarification."
            )

        return result

    def _format_tool_executions(
        self, all_tool_results: List[Dict]
    ) -> List[Dict]:
        """Format raw tool results into a structured list."""
        executions: List[Dict] = []
        for tr in all_tool_results:
            tool_result_obj = tr.get("result")
            tool_call = tr.get("tool_call")

            status = "unknown"
            if hasattr(tool_result_obj, "status"):
                status_obj = tool_result_obj.status
                status = getattr(status_obj, "value", status_obj)

            # Extract tool_name and arguments from tool_call object
            tool_name = "unknown"
            arguments: Dict = {}
            if tool_call is not None:
                if hasattr(tool_call, "function"):
                    tool_name = getattr(tool_call.function, "name", "unknown")
                    args_raw = getattr(tool_call.function, "arguments", "{}")
                    if isinstance(args_raw, str):
                        try:
                            arguments = (
                                json.loads(args_raw) if args_raw.strip() else {}
                            )
                        except json.JSONDecodeError:
                            arguments = {}
                    else:
                        arguments = (
                            args_raw if isinstance(args_raw, dict) else {}
                        )
                elif isinstance(tool_call, dict):
                    func = tool_call.get("function", {})
                    tool_name = func.get("name", "unknown")
                    args_raw = func.get("arguments", "{}")
                    if isinstance(args_raw, str):
                        try:
                            arguments = (
                                json.loads(args_raw) if args_raw.strip() else {}
                            )
                        except json.JSONDecodeError:
                            arguments = {}
                    else:
                        arguments = (
                            args_raw if isinstance(args_raw, dict) else {}
                        )

            executions.append({
                "tool_name": tool_name,
                "arguments": arguments,
                "backend": tr.get("backend"),
                "server_name": tr.get("server_name"),
                "status": status,
                "content": (
                    tool_result_obj.content
                    if hasattr(tool_result_obj, "content")
                    else None
                ),
                "error": (
                    tool_result_obj.error
                    if hasattr(tool_result_obj, "error")
                    else None
                ),
                "execution_time": (
                    tool_result_obj.execution_time
                    if hasattr(tool_result_obj, "execution_time")
                    else None
                ),
                "metadata": (
                    tool_result_obj.metadata
                    if hasattr(tool_result_obj, "metadata")
                    else {}
                ),
            })
        return executions

    def _check_task_completion(self, messages: List[Dict]) -> bool:
        """Check whether the last assistant message contains the completion token."""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                return GroundingAgentPrompts.TASK_COMPLETE in content
        return False

    def _extract_last_assistant_message(self, messages: List[Dict]) -> str:
        """Extract the content of the last assistant message."""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""

    # ------------------------------------------------------------------
    # Iteration feedback helpers (kept for future re-enabling)
    # ------------------------------------------------------------------

    def _build_iteration_feedback(
        self,
        iteration: int,
        llm_summary: Optional[str] = None,
        add_guidance: bool = True,
    ) -> Optional[Dict[str, str]]:
        """Build feedback message to add to next iteration."""
        if not llm_summary:
            return None

        feedback_content = GroundingAgentPrompts.iteration_feedback(
            iteration=iteration,
            llm_summary=llm_summary,
            add_guidance=add_guidance,
        )
        return {"role": "system", "content": feedback_content}

    def _remove_previous_guidance(
        self, messages: List[Dict[str, Any]]
    ) -> None:
        """Remove guidance section from previous iteration feedback messages."""
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if (
                    "## Iteration" in content
                    and "Summary" in content
                    and "---" in content
                ):
                    summary_only = content.split("---")[0].strip()
                    msg["content"] = summary_only

    async def _generate_final_summary(
        self,
        instruction: str,
        messages: List[Dict],
        iterations: int,
    ) -> tuple:
        """
        Generate final summary across all iterations (disabled by default).

        Returns:
            tuple[str, bool, List[Dict]]: (summary_text, success_flag, context_used)
        """
        final_summary_prompt = {
            "role": "user",
            "content": GroundingAgentPrompts.final_summary(
                instruction=instruction,
                iterations=iterations,
            ),
        }

        clean_messages: List[Dict] = []
        for msg in messages:
            if msg.get("role") == "tool":
                continue
            clean_msg = msg.copy()
            if "tool_calls" in clean_msg:
                del clean_msg["tool_calls"]
            clean_messages.append(clean_msg)

        clean_messages.append(final_summary_prompt)
        context_for_return = copy.deepcopy(clean_messages)

        try:
            summary_response = await self._llm_client.complete(
                messages=clean_messages,
                tools=None,
                execute_tools=False,
            )
            final_summary = summary_response.get("message", {}).get(
                "content", ""
            )

            if final_summary:
                logger.info(
                    f"Generated final summary: {final_summary[:200]}..."
                )
                return final_summary, True, context_for_return
            else:
                logger.warning("LLM returned empty final summary")
                return (
                    f"Task completed after {iterations} iteration(s). "
                    "Check execution history for details.",
                    True,
                    context_for_return,
                )

        except Exception as e:
            logger.error(f"Error generating final summary: {e}")
            return (
                f"Task completed after {iterations} iteration(s), "
                f"but failed to generate summary: {str(e)}",
                False,
                context_for_return,
            )

    # ------------------------------------------------------------------
    # Recording helper
    # ------------------------------------------------------------------

    async def _record_agent_execution(
        self,
        result: Dict[str, Any],
        instruction: str,
    ) -> None:
        """Record agent execution to recording manager."""
        recording_manager = self.recording_manager
        if not recording_manager:
            return

        tool_summary = []
        if result.get("tool_executions"):
            for exec_info in result["tool_executions"]:
                tool_summary.append({
                    "tool": exec_info.get("tool_name", "unknown"),
                    "backend": exec_info.get("backend", "unknown"),
                    "status": exec_info.get("status", "unknown"),
                })

        await recording_manager.record_agent_action(
            agent_name=self.name,
            action_type="execute",
            input_data={"instruction": instruction},
            reasoning={
                "response": result.get("response", ""),
                "tools_selected": tool_summary,
            },
            output_data={
                "status": result.get("status", "unknown"),
                "iterations": result.get("iterations", 0),
                "num_tool_executions": len(
                    result.get("tool_executions", [])
                ),
            },
            metadata={
                "step": self.step,
                "instruction": instruction,
            },
        )
