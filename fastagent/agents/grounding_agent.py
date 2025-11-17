from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastagent.agents.base import BaseAgent
from fastagent.llm import LLMClient
from fastagent.grounding.core.grounding_client import GroundingClient
from fastagent.grounding.core.types import BackendType
from fastagent.utils.logging import Logger

if TYPE_CHECKING:
    from fastagent.agents.coordinator import AgentCoordinator
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
        max_iterations: int = 20,
    ) -> None:
        """
        Initialize the Grounding Agent.
        
        Args:
            name: Agent name
            backend_scope: List of backends this agent can access (None = all available)
            llm_client: LLM client for reasoning
            coordinator: AgentCoordinator for resource access
            system_prompt: Custom system prompt
            max_iterations: Maximum LLM reasoning iterations for self-correction (default: 10)
        """
        super().__init__(
            name=name,
            backend_scope=backend_scope or ["gui", "shell", "mcp", "web", "system"],
            llm_client=llm_client,
            coordinator=coordinator
        )
        
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._max_iterations = max_iterations
        
        logger.info(f"Grounding Agent initialized: {name}")
        logger.info(f"Backend scope: {self._backend_scope}")
        logger.info(f"Max iterations (self-correction): {self._max_iterations}")

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
        # Dynamically generate available backend types from backend_scope
        available_backends = ", ".join([backend.upper() for backend in self._backend_scope])
        
        return f"""You are a Grounding Agent responsible for executing concrete tasks using available tools.

## Role in the System

You receive specific task instructions from the Host Agent, which has already decomposed the user's request into clear subtasks. Your job is to:
1. Execute the given task using appropriate tools
2. Save any outputs/results immediately
3. Report completion status

Available backends: {available_backends}

## Core Principle: Task Completion

Your PRIMARY goal is completing the assigned task. The task description tells you WHAT to achieve, not HOW to do it. You decide which tools to use based on the task requirements.

Success = Task objective achieved + Required outputs saved

## Execution Guidelines

### Tool Selection
- Read the task description carefully to understand requirements
- Choose tools based on what the task needs to accomplish
- You have access to ALL tools - pick the right ones for the job

### Save Your Work
Critical rule: If you create, retrieve, or generate ANY data or content, you MUST save it immediately.

- Break complex tasks into smaller tool calls when appropriate
- Choose the most suitable tool for each specific task
- Explain your reasoning when selecting tools
- If a tool fails, try alternative approaches or report the issue clearly
- Summarize final results after completing all necessary tool calls

Use the workspace_dir provided in context. Do NOT create subdirectories unless explicitly required.

### When to Stop
Stop immediately when:
- Task objective is achieved
- Required outputs are saved
- No further action adds value

Do NOT:
- Over-verify (checking multiple times)
- Add extra features not requested
- Continue working after task is complete

### Error Handling
If a tool fails:
- Try alternative approaches
- Use different tools if available
- Report if task cannot be completed

## Execution Flow

1. Understand: Read the task instruction
2. Execute: Call appropriate tools to complete the task
3. Save: Immediately save any outputs
4. Report: Briefly describe what was done

## Communication Format

Keep responses concise and factual. After completing your work, include a summary in this format:

[SUMMARY]
What I did: <brief description>
Has artifacts: YES/NO
Artifact type: <type if YES>
Artifact source: <tool name if YES>
Artifact description: <content description if YES>

## Important Notes

- The Host Agent has already planned the overall workflow - you execute individual tasks
- Focus on task completion, not process documentation
- Be decisive - if the task is done, stop
- Save work immediately after creation/retrieval
- Report failures clearly so the system can adapt

## Example

Task: "Search web for latest AI news, save top 5 results to news.json"

<I execute web_search tool and save results>

[SUMMARY]
What I did: Searched for AI news, saved top 5 articles to news.json
Has artifacts: YES
Artifact type: query_result
Artifact source: web_search
Artifact description: JSON file with 5 articles (titles, URLs, summaries, dates)

Task complete."""

    async def get_backend_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all available backends, their servers, and tools.
        
        For MCP backend, groups tools by server to help HostAgent understand
        which server provides which capabilities.
        
        Returns:
            Dictionary mapping backend names to their detailed descriptions
        """
        grounding_client = self.grounding_client
        if not grounding_client:
            logger.warning("Grounding Agent: No grounding client available")
            return {}
        
        descriptions = {}
        
        try:
            # Get descriptions for each backend in scope
            for backend_name in self._backend_scope:
                try:
                    if backend_name == "system":
                        descriptions[backend_name] = "System tools for querying backend capabilities"
                        continue
                    
                    backend_type = BackendType(backend_name)
                    tools = await grounding_client.list_tools(
                        backend=backend_type
                    )
                    
                    if not tools:
                        descriptions[backend_name] = f"Backend '{backend_name}' available but no tools found"
                        continue
                    
                    # Group tools by server (important for MCP which has multiple servers)
                    server_tools = {}
                    for tool in tools:
                        # Get server name from tool's runtime info
                        server_name = getattr(tool.runtime_info, 'server_name', None) if hasattr(tool, 'runtime_info') else None
                        if server_name not in server_tools:
                            server_tools[server_name] = []
                        server_tools[server_name].append(tool)
                    
                    # Build description
                    desc_lines = []
                    
                    good_for_map = {
                        "shell": "File operations, command execution, system tasks",
                        "gui": "GUI automation, browser control, visual interactions",
                        "mcp": "Application control via MCP servers",
                        "web": "Web search, information gathering",
                        "system": "Querying system capabilities"
                    }
                    
                    if backend_name.lower() == "mcp":
                        # For MCP, show each server with its tools
                        desc_lines.append(f"MCP Backend ({len(tools)} total tools across {len(server_tools)} servers)")
                        desc_lines.append(f"Good for: {good_for_map.get(backend_name.lower(), 'Various tasks')}")
                        
                        for server_name, server_tool_list in server_tools.items():
                            server_display = server_name or "<default>"
                            desc_lines.append(f"\n  Server: {server_display} ({len(server_tool_list)} tools)")
                            
                            # Show tool names
                            tool_names = [t.schema.name for t in server_tool_list]
                            desc_lines.append(f"    Tools: {', '.join(tool_names[:8])}")
                            if len(tool_names) > 8:
                                desc_lines.append(f"           {', '.join(tool_names[8:16])}")
                                if len(tool_names) > 16:
                                    desc_lines.append(f"           (and {len(tool_names) - 16} more)")
                            
                            # Show example tool descriptions
                            examples = server_tool_list[:3]
                            if examples:
                                desc_lines.append(f"    Example capabilities:")
                                for tool in examples:
                                    tool_desc = tool.schema.description or "No description"
                                    if len(tool_desc) > 80:
                                        tool_desc = tool_desc[:77] + "..."
                                    desc_lines.append(f"      - {tool.schema.name}: {tool_desc}")
                    else:
                        # For non-MCP backends, show all tools together
                        desc_lines.append(f"{backend_name.upper()} Backend ({len(tools)} tools)")
                        desc_lines.append(f"Good for: {good_for_map.get(backend_name.lower(), 'Various tasks')}")
                        
                        tool_names = [t.schema.name for t in tools]
                        desc_lines.append(f"  Tools: {', '.join(tool_names[:10])}")
                        if len(tool_names) > 10:
                            desc_lines.append(f"         {', '.join(tool_names[10:20])}")
                            if len(tool_names) > 20:
                                desc_lines.append(f"         (and {len(tool_names) - 20} more)")
                        
                        # Show example tool descriptions
                        examples = tools[:3]
                        if examples:
                            desc_lines.append(f"  Example capabilities:")
                            for tool in examples:
                                tool_desc = tool.schema.description or "No description"
                                if len(tool_desc) > 80:
                                    tool_desc = tool_desc[:77] + "..."
                                desc_lines.append(f"    - {tool.schema.name}: {tool_desc}")
                    
                    descriptions[backend_name] = "\n".join(desc_lines)
                    logger.debug(f"Backend {backend_name}: {len(tools)} tools across {len(server_tools)} servers")
                    
                except Exception as e:
                    logger.warning(f"Failed to get description for backend {backend_name}: {e}")
                    descriptions[backend_name] = f"Backend available but description unavailable: {e}"
            
            logger.info(f"Grounding Agent: Retrieved descriptions for {len(descriptions)} backends")
            return descriptions
            
        except Exception as e:
            logger.error(f"Grounding Agent: Failed to get backend descriptions: {e}")
            return {"error": f"Failed to retrieve backend descriptions: {e}"}

    def construct_messages(
        self,
        instruction: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Construct messages for LLM reasoning.
        
        Args:
            instruction: Instruction from Host Agent (card.description or card.title)
            context: Additional context (workflow state, execution_card, hints, etc.)
            
        Returns:
            List of messages for LLM
        """
        messages = [{"role": "system", "content": self._system_prompt}]
        
        # 1. Add workflow context
        if context and "accumulated_context" in context:
            acc_ctx = context["accumulated_context"]
            ctx_lines = []
            
            # Overall goal
            original_task = acc_ctx.get('original_task', 'N/A')
            if original_task and original_task != 'N/A':
                ctx_lines.append(f"## Overall Goal\n\n{original_task}")
            
            # Progress
            completed = acc_ctx.get('completed_steps', 0)
            remaining = acc_ctx.get("remaining_executions", [])
            if remaining:
                total = completed + len(remaining)
                ctx_lines.append(f"\n**Progress**: Step {completed + 1}/{total}")
                
                # Show next step after current
                if len(remaining) > 1:
                    next_step = remaining[1]  # remaining[0] is current, remaining[1] is next
                    ctx_lines.append(f"**Next After This**: {next_step.get('title', 'unknown')}")
            
            # Previous results with data
            previous_results = acc_ctx.get("previous_results", [])
            if previous_results:
                relevant_results = []
                for result in previous_results[-3:]:  # Last 3 max
                    if result.get('full_content') or result.get('data'):
                        relevant_results.append(result)
                
                if relevant_results:
                    ctx_lines.append("\n## Previous Results\n")
                    for result in relevant_results:
                        ctx_lines.append(f"- **{result.get('title', 'unknown')}**")
                        
                        if result.get('full_content'):
                            data = result.get('data')
                            if data:
                                if isinstance(data, str):
                                    data_preview = data[:500] if len(data) > 500 else data
                                    ctx_lines.append(f"    {data_preview}")
                                    if len(data) > 500:
                                        ctx_lines.append(f"    ... ({len(data)} chars total)")
                                else:
                                    data_str = json.dumps(data, ensure_ascii=False)[:500]
                                    ctx_lines.append(f"    {data_str}")
                        elif result.get('summary'):
                            ctx_lines.append(f"    {result['summary'][:200]}")
            
            if ctx_lines:
                messages.append({
                    "role": "system",
                    "content": "\n".join(ctx_lines)
                })
        
        # 2. Add task metadata and completion criteria
        metadata_lines = []
        
        if context:
            # Get metadata from execution_card
            if "execution_card" in context:
                exec_card = context["execution_card"]
                metadata = exec_card.get("metadata", {})
                
                # Step order
                step_order = metadata.get("step_order")
                if step_order is not None:
                    metadata_lines.append(f"**Current Step**: #{step_order}")
                
                # Task category hints
                task_category = metadata.get("task_category")
                if task_category:
                    metadata_lines.append(f"**Task Category**: {task_category}")
                
                # Backend hints (if Host Agent explicitly provided)
                backend_hint = metadata.get("preferred_backend") or context.get("preferred_backend")
                if backend_hint:
                    metadata_lines.append(f"**Backend Suggestion**: {backend_hint} (Host Agent hint)")
                
                # Constraints (if any)
                constraints = metadata.get("constraints")
                if constraints:
                    metadata_lines.append(f"**Constraints**: {constraints}")
        
        if metadata_lines:
            messages.append({
                "role": "system",
                "content": "\n".join(metadata_lines)
            })
        
        # 3. Add workspace directory information (if provided)
        workspace_dir = context.get("workspace_dir") if context else None
        if workspace_dir:
            # Construct workspace message
            workspace_msg = (
                f"**Workspace Directory**: `{workspace_dir}` (save files directly here, "
                f"do NOT add /workspace/ subdirectory)"
            )
            messages.append({
                "role": "system",
                "content": workspace_msg
            })
        
        # Add workspace artifacts information
        workspace_artifacts = context.get("workspace_artifacts") if context else None
        if workspace_artifacts and workspace_artifacts.get("has_files"):
            files_list = ", ".join(workspace_artifacts["files"])
            
            # Check if there are matching files that suggest task completion
            matching_files = workspace_artifacts.get("matching_files", [])
            recent_files = workspace_artifacts.get("recent_files", [])
            
            if matching_files:
                # Task likely already completed - prompt verification
                artifact_msg = f"""Workspace Status: Found existing file(s) that may satisfy task requirements: {', '.join(matching_files)}

Action: First verify if existing files meet the requirements. If yes, report completion. If no, proceed with task execution."""
            elif len(recent_files) >= 2:
                # Multiple recent files - suggest checking first
                artifact_msg = f"""Workspace Status: Found {len(workspace_artifacts['files'])} existing files ({len(recent_files)} recently modified): {files_list}

Action: Check if existing files already satisfy requirements before creating new ones to avoid redundant work."""
            else:
                # Standard workspace notification
                artifact_msg = f"""Workspace Status: Found {len(workspace_artifacts['files'])} existing files: {files_list}

Note: Check existing files before creating new ones if relevant to your task."""
            
            messages.append({
                "role": "system",
                "content": artifact_msg
            })
        
        #Add related execution history
        execution_history = context.get("execution_history") if context else None
        if execution_history:
            history_lines = ["**Related Task Executions** (previous steps that may have produced useful outputs):"]
            for i, exec_info in enumerate(execution_history, 1):
                status_emoji = "✓" if exec_info["status"] == "done" else "✗"
                history_lines.append(f"{i}. {status_emoji} {exec_info['title']} - {exec_info.get('summary', 'No summary')[:200]}")
            
            history_lines.append("\nConsider leveraging outputs from completed tasks above before starting from scratch.")
            messages.append({
                "role": "system",
                "content": "\n".join(history_lines)
            })
        
        # 4. User instruction (most important - comes last)
        # Note: instruction is card.description or card.title (set by engine)
        messages.append({"role": "user", "content": instruction})
        
        return messages

    async def _check_workspace_artifacts(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check workspace directory for existing artifacts that might be relevant to the task.
        Enhanced to detect if task might already be completed.
        """
        import os
        import time
        workspace_info = {"has_files": False, "files": [], "file_details": {}, "recent_files": []}
        
        try:
            # Get workspace path from recording manager or context
            workspace_path = None
            if self._coordinator and self._coordinator.recording_manager:
                recording_manager = self._coordinator.recording_manager
                if hasattr(recording_manager, 'workspace_dir'):
                    workspace_path = recording_manager.workspace_dir
            
            # Fallback to context
            if not workspace_path:
                workspace_path = context.get("workspace_path")
            
            if workspace_path and os.path.exists(workspace_path):
                files = []
                current_time = time.time()
                recent_threshold = 300  # Files modified in last 5 minutes
                
                for filename in os.listdir(workspace_path):
                    filepath = os.path.join(workspace_path, filename)
                    if os.path.isfile(filepath):
                        files.append(filename)
                        # Get file stats
                        stat = os.stat(filepath)
                        file_info = {
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                            "age_seconds": current_time - stat.st_mtime
                        }
                        workspace_info["file_details"][filename] = file_info
                        
                        # Track recently created/modified files
                        if file_info["age_seconds"] < recent_threshold:
                            workspace_info["recent_files"].append(filename)
                
                if files:
                    workspace_info["has_files"] = True
                    workspace_info["files"] = sorted(files)
                    logger.info(f"Grounding Agent: Found {len(files)} existing files in workspace "
                               f"({len(workspace_info['recent_files'])} recent)")
                    
                    # Enhanced: Check if instruction mentions specific filenames
                    instruction = context.get("instruction", "")
                    if instruction:
                        # Look for potential file references in instruction
                        potential_outputs = []
                        import re
                        # Match common file patterns: filename.ext, "filename", 'filename'
                        file_patterns = re.findall(r'["\']?([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+)["\']?', instruction)
                        for pattern in file_patterns:
                            if pattern in files:
                                potential_outputs.append(pattern)
                        
                        if potential_outputs:
                            workspace_info["matching_files"] = potential_outputs
                            logger.info(f"Grounding Agent: Found {len(potential_outputs)} files matching task: {potential_outputs}")
        
        except Exception as e:
            logger.debug(f"Could not check workspace artifacts: {e}")
        
        return workspace_info
    
    async def _get_related_execution_history(self, context: Dict[str, Any]) -> List[Dict]:
        """
        Get history of related executions (sibling executions under same response card).
        
        Args:
            context: Execution context
            
        Returns:
            List of execution results from related tasks
        """
        related_history = []
        
        try:
            # Get related executions from context (set by WorkflowEngine)
            related_executions = context.get("related_executions", [])
            
            for exec_info in related_executions:
                status = exec_info.get("status")
                # Always include finished or blocked tasks; include errors with concise message for troubleshooting
                if status in ["done", "blocked", "error"]:
                    # Prefer 'response' field for normal runs; fall back to 'error' field for failures
                    result_payload = exec_info.get("result", {}) or {}

                    if status == "error":
                        # Extract error string (may be nested) and truncate
                        err_msg = result_payload.get("error") or result_payload.get("response", "")
                        summary = (str(err_msg) or "Unknown error")[:300]
                    else:
                        summary = result_payload.get("response", "")[:300]

                    related_history.append({
                        "title": exec_info.get("title"),
                        "status": status,
                        "summary": summary
                    })
            
            if related_history:
                logger.info(f"Grounding Agent: Found {len(related_history)} related execution(s)")
        
        except Exception as e:
            logger.debug(f"Could not get related execution history: {e}")
        
        return related_history

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an execution request from Host Agent.
        
        Args:
            context: Context dictionary with:
                - instruction: What to execute
                - tools: Optional list of tools to use
                - auto_execute: Whether to auto-execute tools (default True)
                
        Returns:
            Execution result dictionary
        """
        instruction = context.get("instruction", "")
        if not instruction:
            logger.error("Grounding Agent: No instruction provided")
            return {"error": "No instruction provided", "status": "error"}
        
        logger.info(f"Grounding Agent: Processing instruction at step {self.step}")
        
        # Check for existing workspace artifacts
        workspace_info = await self._check_workspace_artifacts(context)
        if workspace_info["has_files"]:
            context["workspace_artifacts"] = workspace_info
            logger.info(f"Workspace has {len(workspace_info['files'])} existing files: {workspace_info['files']}")
        
        # Get related execution history
        related_history = await self._get_related_execution_history(context)
        if related_history:
            context["execution_history"] = related_history
            logger.info(f"Found {len(related_history)} related execution(s)")
        
        # Get available tools for this task
        tools = await self._get_tools_for_context(context)
        
        # Construct messages for LLM
        messages = self.construct_messages(instruction, context)
        
        # Get LLM response with tools
        try:
            response = await self.get_llm_response(
                messages=messages,
                tools=tools if context.get("auto_execute", True) else None,
                execute_tools=context.get("auto_execute", True),
                max_iterations=context.get("max_iterations", self._max_iterations)
            )
            
            # Process the response
            result = await self._process_execution_response(response, instruction)
            
            # Record agent action to recording manager
            if self._coordinator and self._coordinator.recording_manager:
                # Extract tool execution summary for reasoning
                tool_summary = []
                if result.get("tool_executions"):
                    for exec_info in result["tool_executions"]:
                        tool_summary.append({
                            "tool": exec_info.get("tool_name", "unknown"),
                            "backend": exec_info.get("backend", "unknown"),
                            "status": exec_info.get("status", "unknown"),
                        })
                
                await self._coordinator.recording_manager.record_agent_action(
                    agent_name=self.name,
                    action_type="execute",
                    input_data={"instruction": instruction},
                    reasoning={
                        "response": result.get("response", "")[:500],  # Truncate long responses
                        "tools_selected": tool_summary,
                    },
                    output_data={
                        "status": result.get("status", "unknown"),
                        "iterations": result.get("iterations", 0),
                        "num_tool_executions": len(result.get("tool_executions", [])),
                    },
                    metadata={
                        "step": self.step,
                        "instruction": instruction,
                    }
                )
            
            # Increment step
            self.increment_step()
            
            logger.info(f"Grounding Agent: Execution completed with status: {result.get('status')}")
            return result
            
        except Exception as e:
            logger.error(f"Grounding Agent: Execution failed: {e}")
            result = {
                "error": str(e),
                "status": "error",
                "instruction": instruction
            }
            self.increment_step()
            return result

    async def _get_tools_for_context(self, context: Dict[str, Any]) -> List:
        """
        Get all available tools from backend_scope.
        
        HostAgent may provide hints (task_category, preferred_backend) which are
        passed to LLM as context, but GroundingAgent retrieves ALL available tools
        and lets LLM make the final selection based on actual tool capabilities.
        
        Args:
            context: Execution context with optional hints from HostAgent
            
        Returns:
            List of all tools from backend_scope
        """
        grounding_client = self.grounding_client
        if not grounding_client:
            return []
        
        # Retrieve tools from all backends in scope
        all_tools = []
        for backend_name in self._backend_scope:
            try:
                backend_type = BackendType(backend_name)
                tools = await grounding_client.list_tools(backend=backend_type)
                all_tools.extend(tools)
                logger.debug(f"Retrieved {len(tools)} tools from backend: {backend_name}")
            except Exception as e:
                logger.debug(f"Could not get tools from {backend_name}: {e}")
        
        logger.info(
            f"GroundingAgent retrieved {len(all_tools)} tools from {len(self._backend_scope)} backends"
        )
        return all_tools
    
    async def _process_execution_response(
        self,
        response: Dict[str, Any],
        instruction: str
    ) -> Dict[str, Any]:
        """
        Process LLM execution response.
        
        Args:
            response: LLM response from LLMClient.complete()
                Format: {
                    "message": assistant_message,
                    "tool_results": [{"tool_call": ..., "result": ToolResult, ...}],
                    "messages": conversation_history,
                    "iterations": int
                }
            instruction: Original instruction
            
        Returns:
            Processed result dictionary
        """
        # Extract response content
        result = {
            "instruction": instruction,
            "step": self.step,
            "status": "completed",
            "iterations": response.get("iterations", 0)
        }
        
        # Check if tools were executed
        if "tool_results" in response and response["tool_results"]:
            tool_results = response["tool_results"]
            result["tool_executions"] = []
            
            # Process each tool execution result
            # tool_results format: [{"tool_call": ..., "result": ToolResult, "backend": ..., "server_name": ...}]
            for tr in tool_results:
                tool_result_obj = tr.get("result")  # This is a ToolResult object
                
                # Extract tool execution info
                execution_info = {
                    "tool_name": tr.get("tool_call", {}).get("function", {}).get("name", "unknown"),
                    "backend": tr.get("backend"),
                    "server_name": tr.get("server_name"),
                    "status": tool_result_obj.status.value if hasattr(tool_result_obj, 'status') else "unknown",
                    "content": tool_result_obj.content if hasattr(tool_result_obj, 'content') else None,
                    "error": tool_result_obj.error if hasattr(tool_result_obj, 'error') else None,
                    "execution_time": tool_result_obj.execution_time if hasattr(tool_result_obj, 'execution_time') else None,
                    "metadata": tool_result_obj.metadata if hasattr(tool_result_obj, 'metadata') else {},
                }
                result["tool_executions"].append(execution_info)
            
            # Determine overall status based on tool execution results
            # Simplified logic:
            # 1. If last tool call failed -> overall failed
            # 2. If last tool call succeeded:
            #    - If no previous failures -> overall succeeded
            #    - If previous failures exist -> needs evaluation (let EvalAgent decide)
            
            # Safety check: ensure tool_results is not empty before accessing last element
            if tool_results:
                last_result = tool_results[-1].get("result")
            else:
                # Should not happen due to outer check, but be defensive
                result["status"] = "completed"
                return result
            
            # Check if reached max iterations
            iterations = response.get("iterations", 0)
            max_iterations = self._max_iterations
            reached_max_iterations = (iterations >= max_iterations)
            
            # Case 1: Last tool failed -> directly mark as error
            if hasattr(last_result, 'is_error') and last_result.is_error:
                result["status"] = "error"
            else:
                # Case 2: Last tool succeeded -> check if there were previous failures
                has_previous_error = any(
                    hasattr(tr.get("result"), 'is_error') and tr.get("result").is_error
                    for tr in tool_results[:-1]  # Check all except the last one
                )
                
                # Check if reached max iterations
                if reached_max_iterations:
                    # Reached iteration limit - task likely incomplete or stuck in loop
                    result["status"] = "needs_eval"
                    result["reason"] = "reached_max_iterations"
                    result["warning"] = (
                        f"Task reached max iterations ({max_iterations}) without natural completion. "
                        f"This usually indicates the task is stuck in a loop or unclear about completion criteria."
                    )
                    logger.warning(
                        f"Grounding Agent: Task reached max iterations ({max_iterations}), "
                        f"marking as needs_eval for verification"
                    )
                elif not has_previous_error:
                    # No previous failures -> success
                    result["status"] = "success"
                else:
                    # Has previous failures but last succeeded -> needs evaluation
                    # Let EvalAgent use LLM to intelligently judge if task was completed
                    result["status"] = "needs_eval"
                    result["mixed_results"] = True
                    result["failed_count"] = len([
                        tr for tr in tool_results[:-1]
                        if hasattr(tr.get("result"), 'is_error') and tr.get("result").is_error
                    ])
            
            logger.debug(
                f"Grounding Agent: Processed {len(tool_results)} tool executions, "
                f"status: {result['status']}"
            )
        else:
            # No tools executed, just LLM response
            result["status"] = "completed"
        
        # Extract final LLM response message
        if "message" in response:
            message = response["message"]
            result["response"] = message.get("content", "")
        elif "content" in response:
            result["response"] = response["content"]
        
        # Include full conversation history for debugging
        if "messages" in response:
            result["conversation_history"] = response["messages"]

        if result.get("tool_executions") and result.get("status") in ["success", "needs_eval"]:
            try:
                screenshot_info = await self._extract_screenshots_info(result, instruction)
                if screenshot_info:
                    result["screenshot_analysis"] = screenshot_info
                    logger.debug(f"Screenshot analysis: {screenshot_info[:100]}...")
            except Exception as e:
                logger.warning(f"Failed to extract screenshot info: {e}")
        
        if result.get("tool_executions") and result.get("status") in ["success", "needs_eval"]:
            try:
                llm_response = result.get("response", "")
                extracted_knowledge = self._parse_llm_summary(llm_response, result)
                
                if extracted_knowledge:
                    result["extracted_knowledge"] = extracted_knowledge
                    logger.debug(
                        f"Knowledge extracted from LLM response: what_was_done='{extracted_knowledge.get('what_was_done', 'N/A')[:60]}...', "
                        f"has_artifact={extracted_knowledge.get('has_artifact', False)}"
                    )
                
            except Exception as e:
                logger.warning(f"Failed to parse knowledge from LLM response: {e}")
        
        # Soft finish: indicate that backend session remains active for potential reuse
        result["keep_session"] = True
        return result
    
    def _parse_llm_summary(self, llm_response: str, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        import re
        
        summary_match = re.search(r'\[SUMMARY\](.*?)(?:\[|$)', llm_response, re.DOTALL | re.IGNORECASE)
        if not summary_match:
            logger.debug("No [SUMMARY] section found in LLM response")
            return None
        
        summary_text = summary_match.group(1).strip()
        
        what_was_done = ""
        has_artifact = False
        artifact_type = ""
        artifact_source = ""
        artifact_description = ""
        
        what_match = re.search(r'What I did:\s*(.+?)(?:\n|$)', summary_text, re.IGNORECASE)
        if what_match:
            what_was_done = what_match.group(1).strip()
        
        has_match = re.search(r'Has artifacts:\s*(YES|NO)', summary_text, re.IGNORECASE)
        if has_match:
            has_artifact = has_match.group(1).upper() == "YES"
        
        if has_artifact:
            type_match = re.search(r'Artifact type:\s*(.+?)(?:\n|$)', summary_text, re.IGNORECASE)
            if type_match:
                artifact_type = type_match.group(1).strip()
            
            source_match = re.search(r'Artifact source:\s*(.+?)(?:\n|$)', summary_text, re.IGNORECASE)
            if source_match:
                artifact_source = source_match.group(1).strip()
            
            desc_match = re.search(r'Artifact description:\s*(.+?)(?:\n|$)', summary_text, re.IGNORECASE)
            if desc_match:
                artifact_description = desc_match.group(1).strip()
        
        extracted = {
            "what_was_done": what_was_done or "Executed task",
            "has_artifact": has_artifact
        }
        
        if has_artifact:
            artifact_content = None
            artifact_metadata_extra = {}
            tool_executions = result.get("tool_executions", [])
            
            matched_exec = None
            if artifact_source:
                for exec_info in tool_executions:
                    tool_name = exec_info.get("tool_name", "")
                    if artifact_source.lower() in tool_name.lower() or tool_name.lower() in artifact_source.lower():
                        matched_exec = exec_info
                        artifact_content = exec_info.get("content")
                        break
            
            if matched_exec is None and tool_executions:
                matched_exec = tool_executions[-1]
                artifact_content = matched_exec.get("content")
            
            # Enhance content with screenshot analysis if available
            screenshot_analysis = result.get("screenshot_analysis")
            if screenshot_analysis:
                if artifact_content:
                    artifact_content = f"{artifact_content}\n\nScreenshot Analysis:\n{screenshot_analysis}"
                else:
                    artifact_content = f"Screenshot Analysis:\n{screenshot_analysis}"
            
            if matched_exec:
                exec_metadata = matched_exec.get("metadata", {})
                backend = matched_exec.get("backend", "unknown")
                
                if "screenshot" in exec_metadata:
                    artifact_metadata_extra["screenshot"] = exec_metadata["screenshot"]
                    artifact_metadata_extra["needs_vlm_analysis"] = True
                    logger.debug("GUI artifact: screenshot saved to metadata")
                
                if "action_history" in exec_metadata:
                    artifact_metadata_extra["action_history"] = exec_metadata["action_history"]
                
                if "code_history" in exec_metadata:
                    artifact_metadata_extra["code_history"] = exec_metadata["code_history"]
                    artifact_metadata_extra["needs_code_history_analysis"] = True
                    logger.debug(f"Shell artifact: code_history with {len(exec_metadata['code_history'])} steps saved")
                
                if "raw" in exec_metadata:
                    artifact_metadata_extra["raw_data"] = exec_metadata["raw"]
                    artifact_metadata_extra["needs_structure_preservation"] = True
                    logger.debug("MCP artifact: raw structured data saved")
            
            backends_used = [exec_info.get("backend", "unknown") for exec_info in tool_executions]
            primary_backend = backends_used[-1] if backends_used else "unknown"
            
            extracted["artifact"] = {
                "type": artifact_type or "execution_output",
                "content": artifact_content,
                "description": artifact_description or "Execution artifact",
                "metadata": {
                    "backend": primary_backend,
                    "source": artifact_source,
                    "tools_used": [e.get("tool_name") for e in tool_executions],
                    **artifact_metadata_extra 
                }
            }
        
        logger.debug(f"Parsed LLM summary: what_was_done='{what_was_done[:60]}', has_artifact={has_artifact}")
        return extracted
    
    async def _extract_screenshots_info(
        self, 
        result: Dict[str, Any], 
        instruction: str,
        max_screenshots: int = 3
    ) -> Optional[str]:
        """
        Extract key information from screenshots.

        Args:
            result: Execution result containing tool_executions
            instruction: The task instruction
            max_screenshots: Maximum number of screenshots to analyze (default: 3)
        """
        import base64

        screenshots = []
        for i, exec_info in enumerate(result.get("tool_executions", [])):
            backend = exec_info.get("backend", "unknown")
            metadata = exec_info.get("metadata", {})

            if "screenshot" in metadata and metadata["screenshot"]:
                screenshots.append({
                    "index": i,
                    "tool_name": exec_info.get("tool_name", "unknown"),
                    "screenshot": metadata["screenshot"],
                    "content": exec_info.get("content", ""),
                    "backend": backend
                })
                logger.debug(f"Found screenshot from {backend} backend: {exec_info.get('tool_name')}")
        
        if not screenshots:
            return None
        
        logger.info(f"Found {len(screenshots)} screenshots to analyze")
        
        # If too many screenshots, intelligently select key ones
        selected_screenshots = self._select_key_screenshots(screenshots, max_screenshots)
        
        logger.info(f"Selected {len(selected_screenshots)} screenshots for VLM analysis")
        
        # Prepare content for VLM
        content_parts = []
        
        # Add instruction context
        prompt = f"""Task: {instruction}

Analyze the following screenshot(s) and extract KEY CONTENT INFORMATION:

**What to EXTRACT**:
1. **Specific data/numbers**: counts, statistics, measurements, quantities
2. **Important text**: titles, names, messages, labels, descriptions
3. **Content items**: what lists/tables/cards contain (their actual content)
4. **Status/state**: indicators that convey information (badges, alerts)

**What to IGNORE**:
- UI elements: buttons, menus, icons, layouts
- Design: colors, fonts, spacing (unless they convey meaning)
- Navigation: headers, footers, sidebars
- Decorative elements

**Format**: Be SPECIFIC and FACTUAL. Report exact numbers and text you see.

{"Screenshots to analyze:" if len(selected_screenshots) > 1 else "Screenshot to analyze:"}
"""
        
        content_parts.append({"type": "text", "text": prompt})
        
        # Add each selected screenshot
        for i, ss in enumerate(selected_screenshots, 1):
            screenshot_b64 = base64.b64encode(ss["screenshot"]).decode('utf-8')
            
            # Add context for this screenshot
            context = f"\n--- Screenshot {i}"
            if len(selected_screenshots) > 1:
                context += f" (from {ss['tool_name']})"
            context += " ---"
            content_parts.append({"type": "text", "text": context})
            
            # Add the screenshot
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{screenshot_b64}"
                }
            })
        
        # Add final instruction
        final_instruction = "\n\nProvide a concise summary (3-5 sentences) of the KEY CONTENT INFORMATION visible across all screenshots:"
        content_parts.append({"type": "text", "text": final_instruction})
        
        try:
            # Call VLM to analyze
            response = await self.get_llm_response(
                messages=[
                    {
                        "role": "user",
                        "content": content_parts
                    }
                ],
                tools=None,  # No tools for this extraction call
                max_tokens=300
            )
            
            extracted_info = response["message"]["content"].strip()
            logger.info(f"Successfully extracted screenshot info: {extracted_info[:100]}...")
            
            return extracted_info
            
        except Exception as e:
            logger.error(f"Failed to extract screenshot info: {e}", exc_info=True)
            return None
    
    def _select_key_screenshots(
        self, 
        screenshots: List[Dict], 
        max_count: int
    ) -> List[Dict]:
        """
        Intelligently select key screenshots if there are too many.
        """
        if len(screenshots) <= max_count:
            return screenshots
        
        selected_indices = set()
        
        # Always include last (final state)
        selected_indices.add(len(screenshots) - 1)
        
        # If room, include first (initial state)
        if max_count >= 2:
            selected_indices.add(0)
        
        # Fill remaining slots with evenly spaced middle screenshots
        remaining_slots = max_count - len(selected_indices)
        if remaining_slots > 0:
            # Calculate spacing
            available_indices = [
                i for i in range(1, len(screenshots) - 1)
                if i not in selected_indices
            ]
            
            if available_indices:
                step = max(1, len(available_indices) // (remaining_slots + 1))
                for i in range(remaining_slots):
                    idx = min((i + 1) * step, len(available_indices) - 1)
                    if idx < len(available_indices):
                        selected_indices.add(available_indices[idx])
        
        # Return screenshots in original order
        selected = [screenshots[i] for i in sorted(selected_indices)]
        
        logger.debug(
            f"Selected screenshots at indices {sorted(selected_indices)} "
            f"from total of {len(screenshots)}"
        )
        
        return selected