from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import base64
from io import BytesIO

from fastagent.agents.base import BaseAgent
from fastagent.llm import LLMClient
from fastagent.grounding.core.grounding_client import GroundingClient
from fastagent.grounding.core.types import BackendType
from fastagent.memory import MemoryItem
from fastagent.utils.logging import Logger

if TYPE_CHECKING:
    from fastagent.agents.coordinator import AgentCoordinator
    from fastagent.recording import RecordingManager

logger = Logger.get_logger(__name__)


class PromptComponents:  
    SAVE_OPERATION_CHECK = """## Persistent Output Check

If the task requires save/store/write/export/download/persist, verify save operation was performed.  
If the task only requires display/show/view/open/search/browse, saving is NOT required.

**Common issue**: Grounding Agent forgets to save when task requires it, you must catch this."""
    
    VERIFICATION_GUIDANCE = """## Verification Steps Guidance

You can request **EXECUTABLE** verification steps using shell commands or Python code.  
Each verification step should be a clear natural language task description.

**Good examples**:
- Check if file `/tmp/output.txt` exists and show its size
- Count the number of lines in `/var/log/app.log`
- Verify that process `nginx` is running

**Bad examples** (too vague):
- Check if it worked
- Verify the output"""
    
    JSON_OUTPUT_FORMAT = """## Output Format

Respond in JSON format:

```json
{
    "evaluation": "Your assessment of the execution",
    "success": true/false,  // Did the task complete its goal?
    "confidence": "high/medium/low",  // How certain are you about the success?
    "issues": ["Issue 1", "Issue 2", ...],  // Optional: List any problems found
    "suggestions_for_host": ["Suggestion 1", "Suggestion 2", ...]  // ONLY if success=false (task failed)
}
```"""
    
    JSON_OUTPUT_FORMAT_WITH_VERIFICATION = """## Output Format

Respond in JSON format:

```json
{
    "evaluation": "Your assessment of the execution",
    "success": true/false,  // Did the task complete its goal?
    "confidence": "high/medium/low",  // How certain are you about the success?
    "issues": ["Issue 1", "Issue 2", ...],  // Optional: List any problems found
    "suggestions_for_host": ["Suggestion 1", "Suggestion 2", ...],  // ONLY if success=false (task failed)
    "verification_needed": true/false,  // Do you need to run checks before making final judgment?
    "verification_steps": ["Executable task description 1", ...]  // Only if verification_needed=true
}
```"""
    
    FIELD_USAGE_GUIDELINES_BASE = """### Field Usage Guidelines

**`success` field** (the ONLY field that determines re-planning):
- `false` = task failed, will trigger re-planning
- `true` = task succeeded, no re-planning
- **NEVER** return `None`, always make a decision (`true` or `false`)

**`confidence` field**:
- `high` = very confident about your success assessment
- `medium` = reasonably confident, some uncertainty remains
- `low` = uncertain, less confident about the assessment

**`suggestions_for_host` field**:
- Provide ONLY when `success=false` (task failed)
- If `success=true`, no suggestions needed"""
    
    FIELD_USAGE_GUIDELINES_WITH_VERIFICATION = """
**`verification_needed` field**:
- Set to `true` if confidence is low and you need concrete evidence
- The system will execute your verification steps and call you again
- You can verify up to **{max_verification_attempts}** times to avoid infinite loops
- After {max_verification_attempts} attempts, you **MUST** make a final decision (cannot return `verification_needed=true` again)
"""


class EvalAgent(BaseAgent):
    def __init__(
        self,
        name: str = "EvalAgent",
        backend_scope: Optional[List[str]] = None,
        llm_client: Optional[LLMClient] = None,
        coordinator: Optional[AgentCoordinator] = None,
        system_prompt: Optional[str] = None,
        max_verification_attempts: int = 3,
    ) -> None:
        """
        Initialize the Eval Agent.
        
        Args:
            name: Agent name
            backend_scope: Backends available for verification
            llm_client: LLM client for evaluation
            coordinator: Agent coordinator for shared resources
            system_prompt: Custom system prompt (if provided, will override default)
            max_verification_attempts: Maximum number of verification attempts (default: 3)
        """
        super().__init__(
            name=name,
            backend_scope=backend_scope or ["shell"],
            llm_client=llm_client,
            coordinator=coordinator
        )
        
        self._custom_system_prompt = system_prompt  # Store custom prompt separately
        self._max_verification_attempts = max_verification_attempts
        
        # Warning if coordinator is not provided
        if coordinator is None:
            logger.warning(
                "EvalAgent initialized without coordinator. "
                "This will prevent status determination evaluations from updating EXECUTION cards. "
                "Make sure to set coordinator before running status determination tasks."
            )
        
        logger.info(
            f"Eval Agent initialized with backend scope: {self._backend_scope}, "
            f"max_verification_attempts: {max_verification_attempts}"
        )

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
    
    def _get_system_prompt(self, is_final_evaluation: bool = False) -> str:
        if self._custom_system_prompt:
            return self._custom_system_prompt
        return self._default_system_prompt(is_final_evaluation=is_final_evaluation)

    def _status_determination_prompt(self) -> str:
        """System prompt for status determination (mixed success/failure steps in GroundingAgent)."""
        return """You are an Evaluation Agent tasked with determining if a task was successfully completed.

## You Will Be Provided With

1. The original task description
2. A detailed execution history showing both successful and failed tool calls
3. Screenshots from recent execution steps

## Your Responsibility

- Analyze whether the **OVERALL TASK GOAL** was achieved, despite some intermediate failures
- Consider if failures were recovered through retries or alternative approaches
- Use visual evidence (screenshots) to verify the final state
- Make a clear determination: task succeeded or failed
- **Check if persistent output is required**: If the task asks to "save", "store", "write to file", or "export" something, verify the save operation was performed. If the task only asks to "display", "show", "view", or "search" something, saving is not required.

**Be objective and thorough**. A task can still be successful even if some intermediate steps failed, as long as the final goal was achieved. However, if critical steps failed and were not recovered, mark the task as failed.

## Output Format

Respond in JSON format:

```json
{
    "success": true/false,
    "confidence": "high/medium/low",
    "reasoning": "Your detailed explanation",
    "evidence": ["Key evidence point 1", "Key evidence point 2", ...],
    "suggestions_for_host": ["Suggestion 1 if failed", "Suggestion 2 if failed", ...]
}
```"""

    def _default_system_prompt(self, is_final_evaluation: bool = False) -> str:
        """
        Default system prompt for the eval agent.
        
        Args:
            is_final_evaluation: If True, focuses on overall task completion (final step).
                                 If False, focuses on individual step success (intermediate step).
        """
        # Check if shell backend is available for verification
        has_shell = "shell" in self._backend_scope
        
        if is_final_evaluation:
            role_and_focus = """You are an Evaluation Agent.

## Focus: TASK-LEVEL EVALUATION

You are evaluating the **ENTIRE TASK**, not just the final step."""
        else:
            role_and_focus = """You are an Evaluation Agent.

## Focus: STEP-LEVEL EVALUATION

You are evaluating **THIS SPECIFIC STEP**, not the overall task."""
        
        if is_final_evaluation:
            evaluation_scope = """
## Evaluation Criteria - Task-Level

### Key Questions

1. Does the final result satisfy the **ORIGINAL USER REQUEST**?
2. Are **ALL** required deliverables present and correct?
3. Is the overall quality acceptable to the user?
4. Were all intermediate steps successful in contributing to the final goal?

### Evaluation Strategy

- **Think holistically**: Look at the complete workflow from start to finish
- **Verify completeness**: Check if all requested actions were performed
- **Assess quality**: Ensure outputs meet reasonable standards
- **Consider user satisfaction**: Would this result make the user happy?

### Common Pitfalls to Avoid

- Don't only focus on the last step - evaluate the ENTIRE TASK
- Don't mark as successful if earlier steps failed critically
- Don't ignore missing deliverables just because execution didn't error"""
        else:
            evaluation_scope = """
## Evaluation Criteria - Step-Level

### Key Questions

1. Did this step achieve its stated sub-goal?
2. Are the outputs correct and complete for THIS step?
3. Can the workflow proceed to the next step with these results?
4. Were there any errors or issues in THIS step's execution?

### Evaluation Strategy

- **Think locally**: Focus on this step's specific responsibility
- **Check step outputs**: Verify this step produced what it should
- **Consider dependencies**: If this step depends on previous results, did it use them correctly?
- **Assess continuation**: Can the next step work with these results?

### Common Pitfalls to Avoid

- Don't evaluate the overall task completion (that's for final step)
- Don't fail this step just because earlier steps had issues (unless they directly affected this step)
- Don't mark as failed if this step completed its own goal successfully"""
        
        specific_checks = f"""
## Important Checks

{PromptComponents.SAVE_OPERATION_CHECK}"""
        
        if has_shell:
            verification_capability = f"""
## Verification Capability

You **CAN** request verification steps using shell commands or Python code.

{PromptComponents.VERIFICATION_GUIDANCE}

**When to verify**:
- When confidence is low and you need concrete evidence
- When visual evidence is unclear or insufficient
- When critical file operations or system state need confirmation"""
        else:
            verification_capability = """
## Verification Capability

Shell backend is **NOT available** - you **CANNOT** request verification steps.

- Evaluate based on execution history, tool outputs, and screenshots only
- Be more conservative in your confidence assessment without verification ability
- If evidence is insufficient, use `confidence="low"` to indicate uncertainty"""
        
        field_guidelines = PromptComponents.FIELD_USAGE_GUIDELINES.format(
            max_verification_attempts=self._max_verification_attempts
        )
        
        if has_shell:
            json_format = PromptComponents.JSON_OUTPUT_FORMAT_WITH_VERIFICATION
            field_guidelines = (
                PromptComponents.FIELD_USAGE_GUIDELINES_BASE +
                "\n" +
                PromptComponents.FIELD_USAGE_GUIDELINES_WITH_VERIFICATION.format(
                    max_verification_attempts=self._max_verification_attempts
                )
            )
        else:
            json_format = PromptComponents.JSON_OUTPUT_FORMAT
            field_guidelines = PromptComponents.FIELD_USAGE_GUIDELINES_BASE
        
        output_format = f"""
{json_format}

{field_guidelines}"""
        
        return f"""{role_and_focus}

{evaluation_scope}

{specific_checks}

{verification_capability}

{output_format}"""

    def construct_messages(
        self,
        execution_result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Construct messages for evaluation."""
        is_status_determination = (
            context and 
            context.get("metadata", {}).get("evaluation_type") == "status_determination"
        )
        
        if is_status_determination:
            # Special prompt for status determination
            messages = [{"role": "system", "content": self._status_determination_prompt()}]
            
            # Add task description (both overall task and current execution's sub-task)
            metadata_dict = context.get("metadata", {})
            original_task = metadata_dict.get("original_task", "Unknown task")
            execution_task = metadata_dict.get("execution_task", "")
            
            task_info = [f"## Overall User Request\n\n{original_task}"]
            if execution_task and execution_task != original_task:
                task_info.append(f"\n## Current Execution Step\n\n{execution_task}")
            
            messages.append({
                "role": "system",
                "content": "\n".join(task_info)
            })
            
            # Add execution history with details
            tool_executions = execution_result.get("tool_executions", [])
            history_lines = ["## Execution History"]
            failed_count = 0
            success_count = 0
            
            for i, exec_info in enumerate(tool_executions, 1):
                tool_name = exec_info.get("tool_name", "unknown")
                status = exec_info.get("status", "unknown")
                content = exec_info.get("content", "")[:200]
                error = exec_info.get("error", "")
                
                if status == "error":
                    failed_count += 1
                else:
                    success_count += 1
                
                history_lines.append(f"\n**{i}. Tool**: `{tool_name}` | **Status**: {status}")
                if content:
                    history_lines.append(f"   Output: {content}")
                if error:
                    history_lines.append(f"   Error: {error}")
            
            history_lines.append(f"\n**Summary**: {success_count} succeeded, {failed_count} failed")
            messages.append({
                "role": "system",
                "content": "\n".join(history_lines)
            })
            
            # Add screenshots if available (recent steps, the latest screenshot appears first)
            screenshots_data = self._get_recent_screenshots(context, limit=3, reverse=True)
            if screenshots_data:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Screenshots from the execution (newest to oldest):"},
                    ] + screenshots_data
                })
            
            # Add the evaluation question
            messages.append({
                "role": "user",
                "content": (
                    "Based on the above information, please determine if the task was SUCCESSFULLY COMPLETED. "
                    "Respond in JSON format as specified in the system prompt."
                )
            })
        else:
            # Regular evaluation prompt
            metadata = context.get("metadata", {}) if context else {}
            is_last_execution = context.get("is_last_execution", False) if context else False
            evaluation_type = metadata.get("evaluation_type", "")
            
            # Determine if this is a final evaluation
            is_final = (evaluation_type == "final") or is_last_execution
            
            # Get appropriate system prompt based on evaluation type
            system_prompt = self._get_system_prompt(is_final_evaluation=is_final)
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add task/step goal context
            if is_final:
                original_task = context.get("original_task") if context else None
                if original_task:
                    messages.append({
                        "role": "system",
                        "content": f"## OVERALL TASK GOAL\n\n{original_task}"
                    })
                    logger.info(f"Final step evaluation: Using overall task goal: {original_task[:100]}...")
            else:
                step_goal = execution_result.get("task_description") or execution_result.get("description", "")
                if step_goal:
                    messages.append({
                        "role": "system",
                        "content": f"## THIS STEP'S GOAL\n\n{step_goal}"
                    })
                    logger.debug(f"Intermediate step evaluation: Using step goal: {step_goal[:100]}...")
            
            if context and "accumulated_context" in context:
                acc_ctx = context["accumulated_context"]
                
                if is_final:
                    acc_lines = ["## Task-Level Context"]
                    acc_lines.append(f"\n**Total Steps Completed**: {acc_ctx.get('completed_steps', 0)}")
                    
                    # Sanity check: final evaluation should not have remaining executions
                    remaining_execs = acc_ctx.get("remaining_executions", [])
                    if remaining_execs:
                        logger.warning(
                            f"Final evaluation has {len(remaining_execs)} remaining executions! "
                            f"This should not happen. Treating as intermediate evaluation."
                        )
                        # This is actually NOT a final evaluation, fall through to else branch
                        is_final = False
                    
                    if is_final:  # Re-check after sanity check
                        acc_lines.append("")
                        
                        # Show completed steps
                        previous_results = acc_ctx.get("previous_results", [])
                        if previous_results:
                            acc_lines.append(f"\n### Execution Summary ({len(previous_results)} steps)\n")
                            for i, prev_result in enumerate(previous_results, 1):
                                title = prev_result.get('title', 'unknown')
                                status = prev_result.get('status', 'unknown')
                                summary = prev_result.get('summary', '')
                                
                                status_label = "[SUCCESS]" if status == "success" else "[FAILED]"
                                acc_lines.append(f"- {status_label} **Step {i}**: {title} (`{status}`)")
                                if summary:
                                    acc_lines.append(f"  {summary[:150]}")
                        
                        logger.debug(
                            f"Injected TASK-LEVEL context: {len(previous_results)} completed (final evaluation)"
                        )
                
                if not is_final:
                    # STEP-LEVEL EVALUATION: Focused view on current step and dependencies
                    acc_lines = ["## Step-Level Context"]
                    acc_lines.append(f"\n**Task Progress**: Step {acc_ctx.get('completed_steps', 0)} of overall task")
                    acc_lines.append(f"**Original Task**: {acc_ctx.get('original_task', 'N/A')}")
                    acc_lines.append("")
                    
                    # Show next execution card (to check if current step can connect to next)
                    remaining_execs = acc_ctx.get("remaining_executions", [])
                    if remaining_execs:
                        next_exec = remaining_execs[0]
                        acc_lines.append("### Next Planned Step\n")
                        acc_lines.append(f"- **Title**: {next_exec.get('title', 'unknown')}")
                        if next_exec.get('description'):
                            acc_lines.append(f"- **Goal**: {next_exec['description']}")
                        acc_lines.append("")
                        acc_lines.append("**IMPORTANT**: Verify that current step's output can support the next step's requirements.")
                        acc_lines.append("")
                    
                    previous_results = acc_ctx.get("previous_results", [])
                    if previous_results:
                        # INTELLIGENT DEPENDENCY FILTERING:
                        # 1. Get dependency card IDs from execution_card metadata
                        # 2. Filter previous_results to only show dependent steps
                        # 3. Fallback to recent steps if no explicit dependencies
                        
                        execution_card = context.get("execution_card", {})
                        depends_on = execution_card.get("metadata", {}).get("depends_on", [])
                        
                        # Normalize depends_on to list
                        if not isinstance(depends_on, list):
                            depends_on = [depends_on] if depends_on else []
                        
                        if depends_on:
                            # Filter to show only dependent steps
                            dependent_results = []
                            for prev_result in previous_results:
                                prev_card_id = prev_result.get("card_id", "")
                                if prev_card_id in depends_on:
                                    dependent_results.append(prev_result)
                            
                            if dependent_results:
                                acc_lines.append(f"### Dependency Context ({len(dependent_results)} dependent step(s) of {len(previous_results)} total)\n")
                                
                                for i, dep_result in enumerate(dependent_results, 1):
                                    # Find the actual step number
                                    try:
                                        actual_step = previous_results.index(dep_result) + 1
                                    except ValueError:
                                        actual_step = i
                                    
                                    acc_lines.append(f"#### Dependent Step {actual_step}\n")
                                    acc_lines.append(f"- **Title**: {dep_result.get('title', 'unknown')}")
                                    acc_lines.append(f"- **Status**: {dep_result.get('status', 'unknown')}")
                                    
                                    summary = dep_result.get('summary', '')
                                    if summary:
                                        acc_lines.append(f"  Summary: {summary}")
                                    
                                    # Include full data if available (for DATA_RETRIEVAL type)
                                    if dep_result.get('full_content'):
                                        data = dep_result.get('data')
                                        if data:
                                            if isinstance(data, str):
                                                data_preview = data[:800] if len(data) > 800 else data
                                                acc_lines.append(f"  Data: {data_preview}")
                                                if len(data) > 800:
                                                    acc_lines.append(f"       ... (truncated, {len(data)} total chars)")
                                            else:
                                                data_str = json.dumps(data, ensure_ascii=False)[:800]
                                                acc_lines.append(f"  Data: {data_str}")
                                
                                logger.debug(
                                    f"Injected STEP-LEVEL context: {len(dependent_results)} dependent steps "
                                    f"(filtered from {len(previous_results)} total based on depends_on)"
                                )
                            else:
                                logger.warning(
                                    f"Step has depends_on={depends_on} but no matching results found. "
                                    f"Falling back to recent steps."
                                )
                                depends_on = []  # Trigger fallback
                        
                        if not depends_on:
                            # FALLBACK: No explicit dependencies, show recent steps
                            recent_count = min(3, len(previous_results))
                            if len(previous_results) > recent_count:
                                acc_lines.append(f"### Recent Steps (showing last {recent_count} of {len(previous_results)})\n")
                            else:
                                acc_lines.append(f"### Previous Steps ({len(previous_results)} total)\n")
                            
                            recent_results = previous_results[-recent_count:]
                            for i, prev_result in enumerate(recent_results):
                                actual_step = len(previous_results) - recent_count + i + 1
                                acc_lines.append(f"#### Step {actual_step}\n")
                                acc_lines.append(f"- **Title**: {prev_result.get('title', 'unknown')}")
                                acc_lines.append(f"- **Status**: {prev_result.get('status', 'unknown')}")
                                
                                summary = prev_result.get('summary', '')
                                if summary:
                                    acc_lines.append(f"  Summary: {summary}")
                                
                                # Include full data if available (for DATA_RETRIEVAL type)
                                if prev_result.get('full_content'):
                                    data = prev_result.get('data')
                                    if data:
                                        if isinstance(data, str):
                                            data_preview = data[:600] if len(data) > 600 else data
                                            acc_lines.append(f"  Data: {data_preview}")
                                            if len(data) > 600:
                                                acc_lines.append(f"       ... (truncated, {len(data)} total chars)")
                                        else:
                                            data_str = json.dumps(data, ensure_ascii=False)[:600]
                                            acc_lines.append(f"  Data: {data_str}")
                            
                            logger.debug(
                                f"Injected STEP-LEVEL context: {len(previous_results)} total, "
                                f"showing last {recent_count} (fallback - no explicit dependencies)"
                            )
                
                messages.append({
                    "role": "system",
                    "content": "\n".join(acc_lines)
                })
            
            # Add plan context if available (legacy support)
            elif context and "current_plan" in context:
                messages.append({
                    "role": "system",
                    "content": f"Current Plan:\n{context['current_plan']}"
                })
            
            # Add workspace verification information
            workspace_verification = context.get("workspace_verification") if context else None
            if workspace_verification and workspace_verification.get("files_found"):
                files_list = workspace_verification["files_found"]
                file_details = []
                for filename in files_list:
                    info = workspace_verification["file_info"].get(filename, {})
                    size_kb = info.get("size_bytes", 0) / 1024
                    file_details.append(f"  - {filename} ({size_kb:.1f} KB)")
                
                workspace_msg = f"""## IMPORTANT: Workspace File Evidence

The workspace directory contains {len(files_list)} file(s):
{chr(10).join(file_details)}

**CRITICAL**: When evaluating task success, prioritize file system evidence over visual screenshots.
If the task required creating/saving files and these files exist with reasonable content, 
this is strong evidence of success, even if screenshots show unexpected UI state."""
                
                messages.append({
                    "role": "system",
                    "content": workspace_msg
                })
            
            # Add execution result
            result_str = json.dumps(execution_result, indent=2)
            messages.append({
                "role": "user",
                "content": f"Please evaluate the following execution result:\n\n{result_str}"
            })
            
            # Add screenshot if available - support both direct screenshot and from recording manager
            screenshots_added = False
            if context and "screenshot" in context:
                # Add screenshot for visual evaluation
                screenshot = context["screenshot"]
                if isinstance(screenshot, str):  # base64 encoded
                    try:
                        # Decode, compress if needed, and re-encode
                        raw_img_data = base64.b64decode(screenshot)
                        compressed_img_data, media_type = self._compress_image_if_needed(raw_img_data, max_size_mb=3.5)
                        screenshot = base64.b64encode(compressed_img_data).decode('utf-8')
                        
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Current screen state:"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{media_type};base64,{screenshot}"}
                                }
                            ]
                        })
                        screenshots_added = True
                    except Exception as e:
                        logger.warning(f"Failed to process screenshot from context: {e}")
                        screenshots_added = False
            
            # If no screenshot in context, try to get recent screenshots from recording manager
            if not screenshots_added:
                screenshots_data = self._get_recent_screenshots(context, limit=2, reverse=True)
                if screenshots_data:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Recent screenshots from execution (newest to oldest):"},
                        ] + screenshots_data
                    })
            
            if not self.memory.is_empty():
                memory_content = self._format_memory_for_prompt()
                messages.append({
                    "role": "system",
                    "content": f"Previous Evaluations (for reference):\n{memory_content}"
                })
        
        return messages

    def _format_memory_for_prompt(self) -> str:
        """Format memory items for prompt context."""
        lines = []
        for item in self.memory.content[-2:]:  # Last 2 evaluations
            item_dict = item.to_dict()
            step = item_dict.get("step", "?")
            success = item_dict.get("success", "?")
            evaluation = item_dict.get("evaluation", "")
            lines.append(f"[Step {step}] Success: {success} - {evaluation[:80]}")  # Reduced from 100
        return "\n".join(lines) if lines else "No previous evaluations"
    
    def _detect_image_format(self, image_data: bytes) -> str:
        """
        Detect image format from magic bytes.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Media type string like 'image/png' or 'image/jpeg'
        """
        # PNG magic bytes: \x89PNG\r\n\x1a\n
        if len(image_data) >= 8 and image_data[:8] == b'\x89PNG\r\n\x1a\n':
            return 'image/png'
        
        # JPEG magic bytes: \xFF\xD8\xFF
        if len(image_data) >= 3 and image_data[:3] == b'\xff\xd8\xff':
            return 'image/jpeg'
        
        # GIF magic bytes: GIF87a or GIF89a
        if len(image_data) >= 6 and image_data[:6] in (b'GIF87a', b'GIF89a'):
            return 'image/gif'
        
        # WebP magic bytes: RIFF....WEBP
        if len(image_data) >= 12 and image_data[:4] == b'RIFF' and image_data[8:12] == b'WEBP':
            return 'image/webp'
        
        # Default to PNG (most common for screenshots)
        logger.warning(f"Could not detect image format from magic bytes, defaulting to image/png")
        return 'image/png'
    
    def _compress_image_if_needed(
        self,
        image_data: bytes,
        max_size_mb: float = 3.5,
        quality_start: int = 85
    ) -> tuple[bytes, str]:
        """
        Compress image if it exceeds size limit.
        
        Args:
            image_data: Raw image bytes
            max_size_mb: Maximum size in MB (accounting for base64 encoding overhead)
            quality_start: Starting quality for compression (0-100)
            
        Returns:
            Tuple of (image_bytes, media_type) where media_type is like 'image/png' or 'image/jpeg'
        """
        try:
            from PIL import Image
        except ImportError:
            logger.warning("PIL not available, cannot process images")
            # Return original data with detected format
            format_type = self._detect_image_format(image_data)
            return image_data, format_type
        
        max_bytes = int(max_size_mb * 1024 * 1024)
        original_size = len(image_data)
        
        logger.debug(f"Processing image of size {original_size / 1024 / 1024:.2f}MB")
        
        # Detect original format
        original_format = self._detect_image_format(image_data)
        
        # If image is already small enough, return as-is with detected format
        if original_size <= max_bytes:
            logger.debug(f"Image size OK, keeping original format: {original_format}")
            return image_data, original_format
        
        # Image is too large, need to compress to JPEG
        logger.info(f"Image size {original_size / 1024 / 1024:.2f}MB exceeds limit {max_size_mb}MB, compressing to JPEG...")
        
        try:
            # Load image
            img = Image.open(BytesIO(image_data))
            
            # Convert RGBA to RGB if necessary (for JPEG)
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Try progressively lower quality until we meet the size requirement
            quality = quality_start
            compressed_data = image_data
            
            while quality >= 20:  # Don't go below 20% quality
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=quality, optimize=True)
                compressed_data = buffer.getvalue()
                compressed_size = len(compressed_data)
                
                logger.debug(f"Compressed at quality {quality}: {compressed_size / 1024 / 1024:.2f}MB")
                
                if compressed_size <= max_bytes:
                    logger.info(
                        f"Successfully compressed image from {original_size / 1024 / 1024:.2f}MB "
                        f"to {compressed_size / 1024 / 1024:.2f}MB (quality={quality})"
                    )
                    return compressed_data, 'image/jpeg'
                
                quality -= 15
            
            # If still too large, try reducing dimensions
            logger.warning(f"Quality reduction insufficient, trying to reduce dimensions...")
            scale = 0.8
            while scale >= 0.3:
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                buffer = BytesIO()
                resized_img.save(buffer, format='JPEG', quality=60, optimize=True)
                compressed_data = buffer.getvalue()
                compressed_size = len(compressed_data)
                
                logger.debug(f"Resized to {new_width}x{new_height}: {compressed_size / 1024 / 1024:.2f}MB")
                
                if compressed_size <= max_bytes:
                    logger.info(
                        f"Successfully compressed image from {original_size / 1024 / 1024:.2f}MB "
                        f"to {compressed_size / 1024 / 1024:.2f}MB (scale={scale:.2f}, quality=60)"
                    )
                    return compressed_data, 'image/jpeg'
                
                scale -= 0.1
            
            # If all else fails, return the best we got (as JPEG)
            logger.warning(
                f"Could not compress image below {max_size_mb}MB limit. "
                f"Final size: {len(compressed_data) / 1024 / 1024:.2f}MB"
            )
            return compressed_data, 'image/jpeg'
            
        except Exception as e:
            logger.error(f"Failed to compress image: {e}")
            # Return original with detected format
            format_type = self._detect_image_format(image_data)
            return image_data, format_type
    
    def _get_recent_screenshots(
        self, 
        context: Dict[str, Any], 
        limit: int = 3, 
        reverse: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get recent screenshots from recording manager.
        
        Args:
            context: Execution context
            limit: Maximum number of screenshots to retrieve
            reverse: If True, return screenshots in reverse order (newest first)
        """
        screenshots_data = []
        
        try:
            # Get screenshots from RecordingManager's trajectory recorder
            # Screenshot info is stored in recorder.steps, not in tool_executions
            if not self.recording_manager or not hasattr(self.recording_manager, '_recorder'):
                logger.debug("No recording manager or recorder available")
                return []
            
            # Check if screenshots are enabled
            if not self.recording_manager.enable_screenshot:
                logger.debug("Screenshot recording is disabled (enable_screenshot=False)")
                return []
            
            recorder = self.recording_manager._recorder
            if not recorder or not hasattr(recorder, 'steps') or not hasattr(recorder, 'trajectory_dir'):
                logger.debug("Recorder not properly initialized")
                return []
            
            # Get recorded steps (these contain screenshot paths)
            recorded_steps = recorder.steps
            if not recorded_steps:
                logger.debug("No recorded steps available")
                return []
            
            # Get the last N steps with screenshots
            steps_with_screenshots = [
                step for step in recorded_steps 
                if step.get("screenshot")
            ]
            
            if not steps_with_screenshots:
                logger.debug("No steps with screenshots found")
                return []
            
            # Get recent steps
            recent_steps = steps_with_screenshots[-limit:] if limit else steps_with_screenshots
            
            # Reverse if requested (newest first)
            if reverse:
                recent_steps = list(reversed(recent_steps))
            
            # Load and format screenshots
            for idx, step_info in enumerate(recent_steps):
                screenshot_path = step_info.get("screenshot")
                if not screenshot_path:
                    continue
                
                full_path = Path(recorder.trajectory_dir) / screenshot_path
                if not full_path.exists():
                    logger.debug(f"Screenshot file not found: {full_path}")
                    continue
                
                try:
                    # Read image data
                    with open(full_path, 'rb') as f:
                        raw_img_data = f.read()
                    
                    # Compress if needed (Claude API has 5MB limit, base64 adds ~33% overhead)
                    # So we target 3.5MB for raw image to be safe
                    compressed_img_data, media_type = self._compress_image_if_needed(raw_img_data, max_size_mb=3.5)
                    
                    # Encode to base64
                    img_data = base64.b64encode(compressed_img_data).decode('utf-8')
                    
                    # Check final base64 size (just for logging)
                    base64_size_mb = len(img_data) * 3 / 4 / 1024 / 1024  # Approximate decoded size
                    if base64_size_mb > 4.5:
                        logger.warning(
                            f"Base64 image size {base64_size_mb:.2f}MB may be close to Claude's 5MB limit"
                        )
                    
                    step_num = step_info.get("step", "?")
                    tool_name = step_info.get("tool", "unknown")
                    backend = step_info.get("backend", "unknown")
                    
                    # Determine status from result
                    result = step_info.get("result", {})
                    status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"
                    
                    # Add label to indicate which step and order
                    if reverse:
                        label = f"Step {step_num}" + (" (Latest)" if idx == 0 else "")
                    else:
                        label = f"Step {step_num}" + (" (Latest)" if idx == len(recent_steps) - 1 else "")
                    
                    # Add text label before the image
                    screenshots_data.append({
                        "type": "text",
                        "text": f"\n{label} - Backend: {backend} | Tool: {tool_name} | Status: {status}"
                    })
                    screenshots_data.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{img_data}"}
                    })
                    
                except Exception as read_error:
                    logger.debug(f"Failed to read screenshot {full_path}: {read_error}")
                    continue
                    
        except Exception as e:
            logger.debug(f"Failed to load screenshots: {e}")
        
        return screenshots_data

    async def _verify_workspace_outputs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify workspace directory for expected output files.
        """
        import os
        verification = {
            "has_workspace": False,
            "files_found": [],
            "file_info": {}
        }
        
        try:
            # Get workspace path
            workspace_path = None
            if self._coordinator and self._coordinator.recording_manager:
                recording_manager = self._coordinator.recording_manager
                if hasattr(recording_manager, 'workspace_dir'):
                    workspace_path = recording_manager.workspace_dir
            
            if not workspace_path:
                workspace_path = context.get("workspace_dir")
            
            if workspace_path and os.path.exists(workspace_path):
                verification["has_workspace"] = True
                files = []
                for filename in os.listdir(workspace_path):
                    filepath = os.path.join(workspace_path, filename)
                    if os.path.isfile(filepath):
                        files.append(filename)
                        stat = os.stat(filepath)
                        verification["file_info"][filename] = {
                            "size_bytes": stat.st_size,
                            "modified": stat.st_mtime
                        }
                
                verification["files_found"] = sorted(files)
                if files:
                    logger.info(f"EvalAgent: Workspace verification found {len(files)} file(s): {files}")
        
        except Exception as e:
            logger.debug(f"Could not verify workspace outputs: {e}")
        
        return verification

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an evaluation request.
        
        Args:
            context: Evaluation context containing:
                - execution_result: The execution result to evaluate
                - metadata: Metadata including evaluation_type
                - original_task: Original task description (for final step evaluation)
                - is_last_execution: Whether this is the last execution step
        """
        execution_result = context.get("execution_result")
        if not execution_result:
            logger.error("Eval Agent: No execution result provided")
            return {"error": "No execution result provided"}
        
        logger.info(f"Eval Agent: Evaluating execution at step {self.step}")
        
        # Verify workspace outputs before evaluating
        workspace_verification = await self._verify_workspace_outputs(context)
        if workspace_verification["files_found"]:
            context["workspace_verification"] = workspace_verification
            logger.info(
                f"Workspace contains {len(workspace_verification['files_found'])} file(s) - "
                f"will include in evaluation context"
            )
        
        # Check if this is a status determination evaluation
        metadata = context.get("metadata", {})
        is_status_determination = metadata.get("evaluation_type") == "status_determination"
        is_last_execution = context.get("is_last_execution", False)
        
        # If this is the last execution but evaluation_type is not set, automatically set it to "final"
        if is_last_execution and not is_status_determination:
            if not metadata.get("evaluation_type"):
                metadata["evaluation_type"] = "final"
                context["metadata"] = metadata
                logger.info("Automatically detected final step evaluation (is_last_execution=True)")
        
        # Coordinator dependency check for status determination
        execution_card_id = metadata.get("execution_card_id")
        if is_status_determination and not self._coordinator:
            logger.error("EvalAgent requires coordinator for status determination but coordinator is None")
            # This is a configuration error, not a task failure
            return {
                "status": "error",
                "error": "Coordinator not available for status determination",
                "evaluation": "Cannot determine status without coordinator - configuration error",
            }
        
        messages = self.construct_messages(execution_result, context)
        try:
            logger.debug(f"Calling LLM with {len(messages)} messages")
            response = await self.get_llm_response(messages)
            
            response_content = self._extract_response_content(response)
            logger.info(f"EvalAgent LLM response (first 300 chars): {response_content[:300]}...")
            
            eval_dict = self.response_to_dict(response_content)
            logger.info(f"Parsed eval_dict keys: {list(eval_dict.keys())}")
            logger.info(f"Parsed eval_dict (full): {eval_dict}")
            
            # Check if LLM returned expected fields
            missing_fields = []
            if "evaluation" not in eval_dict or not eval_dict.get("evaluation"):
                missing_fields.append("evaluation")
            if "success" not in eval_dict:
                missing_fields.append("success")
            if "confidence" not in eval_dict:
                missing_fields.append("confidence")
            
            if missing_fields:
                logger.warning(
                    f"EvalAgent: LLM response missing expected fields: {missing_fields}. "
                    f"Filling with default values. Full response was: {response_content[:500]}"
                )
                # Fill missing fields with reasonable defaults
                if "evaluation" not in eval_dict or not eval_dict.get("evaluation"):
                    eval_dict["evaluation"] = "Evaluation completed but no detailed assessment provided by LLM"
                if "success" not in eval_dict:
                    # Conservative default: mark as uncertain (None) rather than assuming success or failure
                    eval_dict["success"] = None
                if "confidence" not in eval_dict:
                    eval_dict["confidence"] = "unknown"
                if "issues" not in eval_dict:
                    eval_dict["issues"] = []
                if "suggestions_for_host" not in eval_dict:
                    eval_dict["suggestions_for_host"] = []
                if "verification_needed" not in eval_dict:
                    eval_dict["verification_needed"] = False
                if "verification_steps" not in eval_dict:
                    eval_dict["verification_steps"] = []
            
            logger.info(
                f"Final eval_dict: success={eval_dict.get('success')}, "
                f"confidence={eval_dict.get('confidence')}, "
                f"has_evaluation={bool(eval_dict.get('evaluation'))}"
            )
            
            # Handle status determination
            if is_status_determination:
                success = eval_dict.get("success", False)
                confidence = eval_dict.get("confidence", "low")
                
                logger.info(
                    f"Status determination: success={success}, confidence={confidence}"
                )
                
                return {
                    "status": "success",
                    "evaluation_type": "status_determination",
                    "success": success,
                    "confidence": confidence,
                    "reasoning": eval_dict.get("reasoning", ""),
                    "evidence": eval_dict.get("evidence", []),
                    "suggestions_for_host": eval_dict.get("suggestions_for_host", []),
                    "execution_card_id": execution_card_id,  # Include for WorkflowEngine
                }

            verification_count = 0
            all_verification_history = []  # Track all verification attempts
            
            while eval_dict.get("verification_needed") and context.get("run_verification", True):
                if verification_count >= self._max_verification_attempts:
                    # HARD LIMIT: Block evaluation if verification limit exceeded
                    logger.error(
                        f"EvalAgent: Verification limit EXCEEDED ({verification_count} attempts made in this evaluation). "
                        f"LLM still requested verification_needed=true after {self._max_verification_attempts} attempts. "
                        f"This indicates the task cannot be reliably evaluated. Blocking evaluation."
                    )
                    return {
                        "status": "error",
                        "error": f"Verification limit exceeded: {verification_count} attempts made, LLM still uncertain",
                        "evaluation": (
                            f"Evaluation blocked: After {verification_count} verification attempts within this evaluation, "
                            f"EvalAgent still cannot make a confident decision. "
                            f"This suggests the task requirements may be unclear, the execution result is ambiguous, "
                            f"or the task is too complex for automated evaluation."
                        ),
                        "verification_count": verification_count,
                        "verification_history": all_verification_history,
                        "verification_limit_exceeded": True,
                        "suggestions": [
                            "Task requirements may need to be more specific",
                            "Consider breaking down the task into smaller, more measurable steps",
                            "Manual review may be needed to determine task success",
                        ]
                    }
                
                # Increment count at the start of each verification attempt
                verification_count += 1
                logger.info(
                    f"EvalAgent: Running verification attempt {verification_count}/{self._max_verification_attempts} "
                    f"with {len(eval_dict.get('verification_steps', []))} steps"
                )
                
                # Execute verification steps
                verification_results = await self._run_verification_steps(
                    eval_dict.get("verification_steps", [])
                )
                
                # Record this verification attempt
                all_verification_history.append({
                    "attempt": verification_count,
                    "steps": eval_dict.get("verification_steps", []),
                    "results": verification_results,
                    "initial_confidence": eval_dict.get("confidence"),
                })
                
                # Check if verification was skipped (no shell backend available)
                verification_skipped = any(r.get("status") == "skipped" for r in verification_results)
                if verification_skipped:
                    logger.warning(
                        f"EvalAgent: Verification skipped (shell backend not available). "
                        f"Forcing evaluation based on current information only."
                    )
                    # Force verification_needed to False to break the loop
                    eval_dict["verification_needed"] = False
                    eval_dict["verification_skipped"] = True
                    eval_dict["verification_note"] = (
                        "Verification backend not available. "
                        "Evaluation is based on execution history and screenshots only."
                    )
                    # Don't call _update_evaluation_with_verification since there's no new info
                    # Just break the loop and return current evaluation
                    logger.info(
                        f"EvalAgent: Breaking verification loop - "
                        f"success={eval_dict.get('success')}, confidence={eval_dict.get('confidence')}"
                    )
                    break
                
                # Check if verification operations themselves failed
                verification_errors = [r for r in verification_results if r.get("status") == "error"]
                if verification_errors:
                    logger.error(
                        f"EvalAgent: {len(verification_errors)}/{len(verification_results)} "
                        f"verification steps failed in attempt {verification_count}"
                    )
                    # If verification operations failed, this is an evaluation failure
                    return {
                        "status": "error",
                        "error": f"Verification operations failed: {len(verification_errors)} errors in attempt {verification_count}",
                        "evaluation": "Evaluation failed due to verification operation errors",
                        "verification_count": verification_count,
                        "verification_history": all_verification_history,
                        "verification_errors": [r.get("error", "unknown") for r in verification_errors],
                    }
                
                logger.info(
                    f"EvalAgent: Verification attempt {verification_count} completed successfully, "
                    f"updating evaluation based on results"
                )
                
                # Re-evaluate based on verification results (LLM call with verification results)
                eval_dict = await self._update_evaluation_with_verification(
                    initial_evaluation=eval_dict,
                    verification_results=verification_results,
                    verification_attempt=verification_count,
                    max_attempts=self._max_verification_attempts,
                )
                
                logger.info(
                    f"EvalAgent: After verification attempt {verification_count} - "
                    f"success={eval_dict.get('success')}, "
                    f"confidence={eval_dict.get('confidence')}, "
                    f"verification_needed={eval_dict.get('verification_needed', False)}"
                )
                
                # Loop continues if verification_needed is still true
                # Otherwise, breaks and returns the final evaluation
            
            # Add verification summary to final result
            if verification_count > 0:
                eval_dict["total_verification_attempts"] = verification_count
                eval_dict["verification_history"] = all_verification_history
                logger.info(
                    f"EvalAgent: Completed evaluation after {verification_count} verification attempt(s)"
                )
            
            # Add to memory
            self._add_evaluation_to_memory(execution_result, eval_dict)
            
            # Record agent action to recording manager
            if self._coordinator and self._coordinator.recording_manager:
                await self._coordinator.recording_manager.record_agent_action(
                    agent_name=self.name,
                    action_type="evaluate",
                    input_data={
                        "execution_step": execution_result.get("step", "unknown"),
                        "execution_status": execution_result.get("status", "unknown"),
                    },
                    reasoning={
                        "evaluation": eval_dict.get("evaluation", "")[:500],  # Truncate long evaluations
                        "issues": eval_dict.get("issues", []),
                        "verification_needed": eval_dict.get("verification_needed", False),
                    },
                    output_data={
                        "success": eval_dict.get("success"),  # Keep None if missing (don't assume False)
                        "confidence": eval_dict.get("confidence", "unknown"),
                        "suggestions_for_host": eval_dict.get("suggestions_for_host", []),
                        "num_verification_steps": len(eval_dict.get("verification_steps", [])),
                    },
                    metadata={
                        "step": self.step,
                        "evaluated_step": execution_result.get("step", "unknown"),
                    }
                )
            
            self.increment_step()
            
            # Log final evaluation result
            success_status = eval_dict.get('success')
            if success_status is None:
                success_str = "None (LLM did not provide)"
            else:
                success_str = str(success_status)
            
            logger.info(
                f"Evaluation complete - success={success_str}, "
                f"confidence={eval_dict.get('confidence')}, "
                f"has_issues={len(eval_dict.get('issues', []))}"
            )
            
            return eval_dict
            
        except asyncio.TimeoutError as e:
            logger.error(f"LLM call timed out: {e}")
            return {
                "status": "error",
                "error": f"LLM timeout: {str(e)}",
                "evaluation": "Evaluation failed due to timeout - cannot determine task success",
                "execution_card_id": execution_card_id if is_status_determination else None,
            }
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "evaluation": "Evaluation failed due to error - cannot determine task success",
                "execution_card_id": execution_card_id if is_status_determination else None,
            }
    
    def _extract_response_content(self, response: Dict[str, Any]) -> str:
        """Extract text content from LLM response."""
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if "message" in choice:
                return choice["message"].get("content", "")
        elif "message" in response:
            return response["message"].get("content", "")
        elif "content" in response:
            return response["content"]
        return json.dumps(response)

    async def _update_evaluation_with_verification(
        self,
        initial_evaluation: Dict[str, Any],
        verification_results: List[Dict[str, Any]],
        verification_attempt: int = 1,
        max_attempts: int = 3,
    ) -> Dict[str, Any]:
        """
        Update evaluation based on verification results using a second LLM call.
        
        Args:
            initial_evaluation: The initial evaluation result from first LLM call
            verification_results: Results from verification steps execution
            verification_attempt: Current verification attempt number (1-indexed)
            max_attempts: Maximum verification attempts allowed
        """
        # Construct follow-up messages for LLM
        followup_messages = [
            {"role": "system", "content": self._get_system_prompt(is_final_evaluation=False)},
            {"role": "user", "content": self._construct_verification_followup_prompt(
                initial_evaluation=initial_evaluation,
                verification_results=verification_results,
                verification_attempt=verification_attempt,
                max_attempts=max_attempts,
            )}
        ]
        
        try:
            logger.debug("EvalAgent: Calling LLM for verification-based update")
            response = await self.get_llm_response(followup_messages)
            response_content = self._extract_response_content(response)
            
            logger.info(f"EvalAgent: Verification follow-up response (first 300 chars): {response_content[:300]}...")
            
            updated_eval = self.response_to_dict(response_content)
            
            final_eval = {
                "evaluation": updated_eval.get("evaluation", initial_evaluation.get("evaluation", "")),
                "success": updated_eval.get("success", initial_evaluation.get("success")),
                "confidence": updated_eval.get("confidence", initial_evaluation.get("confidence", "unknown")),
                "issues": updated_eval.get("issues", initial_evaluation.get("issues", [])),
                "suggestions_for_host": updated_eval.get("suggestions_for_host", initial_evaluation.get("suggestions_for_host", [])),
                "verification_needed": True,  # Mark that verification was performed
                "verification_steps": initial_evaluation.get("verification_steps", []),
                "verification_results": verification_results,
                "initial_evaluation": {  # Keep initial evaluation for reference
                    "success": initial_evaluation.get("success"),
                    "confidence": initial_evaluation.get("confidence"),
                    "evaluation": initial_evaluation.get("evaluation", "")[:200],  # Truncate
                }
            }
            
            logger.info(
                f"EvalAgent: Verification update - "
                f"initial (success={initial_evaluation.get('success')}, confidence={initial_evaluation.get('confidence')}) → "
                f"final (success={final_eval.get('success')}, confidence={final_eval.get('confidence')})"
            )
            
            return final_eval
            
        except Exception as e:
            logger.error(f"EvalAgent: Failed to update evaluation with verification: {e}")
            # Fall back to initial evaluation but mark that verification was attempted
            initial_evaluation["verification_results"] = verification_results
            initial_evaluation["verification_update_failed"] = True
            initial_evaluation["verification_update_error"] = str(e)
            return initial_evaluation
    
    def _construct_verification_followup_prompt(
        self,
        initial_evaluation: Dict[str, Any],
        verification_results: List[Dict[str, Any]],
        verification_attempt: int = 1,
        max_attempts: int = 3,
    ) -> str:
        """
        Construct prompt for follow-up LLM call after verification.
        
        Args:
            initial_evaluation: Initial evaluation from first LLM call
            verification_results: Verification execution results
            verification_attempt: Current verification attempt number
            max_attempts: Maximum verification attempts allowed
        """
        remaining_attempts = max_attempts - verification_attempt
        
        prompt_parts = [f"## Verification Attempt {verification_attempt}/{max_attempts}, Remaining: {remaining_attempts}\n"]
        if remaining_attempts == 0:
            prompt_parts.append("**WARNING**: This is your LAST verification attempt. You MUST make a final decision.\n")
        
        # Show initial evaluation
        prompt_parts.extend([
            "",
            "### Your Initial Evaluation\n",
            f"- **Success**: {initial_evaluation.get('success')}",
            f"- **Confidence**: {initial_evaluation.get('confidence')}",
            f"- **Evaluation**: {initial_evaluation.get('evaluation', 'N/A')}"
        ])
        
        # Show verification steps requested
        prompt_parts.append("\n### Verification Steps You Requested\n")
        for i, step in enumerate(initial_evaluation.get("verification_steps", []), 1):
            prompt_parts.append(f"{i}. {step}")
        
        # Show verification results
        prompt_parts.append("\n### Verification Results\n")
        for i, result in enumerate(verification_results, 1):
            step_desc = result.get("step", "unknown")
            status = result.get("status", "unknown")
            output = result.get("output", "")
            
            prompt_parts.append(f"**{i}. {step_desc}**")
            prompt_parts.append(f"   - Status: `{status}`")
            if output:
                output_preview = output[:500] if len(output) > 500 else output
                prompt_parts.append(f"   - Output: {output_preview}")
                if len(output) > 500:
                    prompt_parts.append(f"   - (truncated, total {len(output)} chars)")
        
        # Task instructions
        prompt_parts.extend([
            "",
            "### Your Task\n",
            "Now that you have verification results, update your evaluation:",
            "1. Compare verification results with your initial assessment",
            "2. Decide: Did the task succeed or fail based on concrete evidence?",
            "3. Update your confidence level based on the verification evidence"
        ])
        
        # Remaining attempts guidance
        if remaining_attempts > 0:
            prompt_parts.append(
                f"\n**Decision**: You can request {remaining_attempts} more verification(s) if evidence is still insufficient, "
                "or provide your final evaluation if you have enough information."
            )
        else:
            prompt_parts.append(
                "\n**FINAL DECISION REQUIRED**: No more verifications allowed. "
                "You must now determine success or failure based on all available evidence."
            )
        
        prompt_parts.append("\nRespond in the JSON format specified in the system prompt.")
        return "\n".join(prompt_parts)
    
    def _get_preferred_verification_backend(self) -> Optional[BackendType]:
        """
        Get preferred backend for verification.
        """
        if "shell" in self._backend_scope:
            return BackendType.SHELL
        
        # No suitable verification backend available
        logger.info(
            f"Shell backend not available in backend_scope {self._backend_scope}. "
            f"Verification steps will be skipped. EvalAgent will evaluate based on "
            f"execution history and screenshots only."
        )
        return None
    
    async def _run_verification_steps(
        self,
        verification_steps: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Run verification steps using shell backend.  
        If shell backend is not available, verification steps are skipped.
        """
        grounding_client = self.grounding_client
        if not grounding_client:
            logger.warning("No grounding client for verification")
            return []
        
        # Check if shell backend is available
        preferred_backend = self._get_preferred_verification_backend()
        if preferred_backend is None:
            # No verification backend available, skip verification steps
            logger.info(
                f"Skipping {len(verification_steps)} verification step(s) - "
                f"shell backend not available. Will evaluate based on history and screenshots."
            )
            return [{
                "step": "Verification skipped",
                "status": "skipped",
                "reason": "Shell backend not available in backend_scope",
                "note": "Evaluation will be based on execution history and screenshots only"
            }]
        
        logger.info(f"Using shell backend for verification")
        
        results = []
         
        for i, step_desc in enumerate(verification_steps, 1):
            logger.info(f"Running verification step {i}/{len(verification_steps)}: {step_desc}")
            
            try:
                tool_result = await grounding_client.invoke_tool(
                    tool="shell_agent",
                    parameters={"task": step_desc},
                    backend=preferred_backend,
                    keep_session=True
                )
                
                results.append({
                    "step": step_desc,
                    "status": "success" if tool_result.is_success else "error",
                    "output": tool_result.content,
                    "execution_time": tool_result.execution_time,
                })
                
                logger.info(f"Verification step {i} completed - status: {tool_result.status}")
                
            except Exception as e:
                logger.error(f"Verification step {i} failed: {e}")
                results.append({
                    "step": step_desc,
                    "status": "error",
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r.get("status") == "success")
        logger.info(f"Verification completed - {success_count}/{len(results)} successful")
        
        return results

    def _add_evaluation_to_memory(
        self,
        execution_result: Dict[str, Any],
        evaluation: Dict[str, Any]
    ) -> None:
        """Add evaluation to memory."""
        memory = MemoryItem()
        memory.set_value("role", "evaluation")
        memory.set_value("execution_step", execution_result.get("step", "?"))
        memory.set_value("success", evaluation.get("success", False))
        memory.set_value("confidence", evaluation.get("confidence", "unknown"))
        memory.set_value("evaluation", evaluation.get("evaluation", ""))
        memory.set_value("full_result", evaluation)
        
        self._memory.add_memory_item(memory, agent_name=self.name, step=self.step)