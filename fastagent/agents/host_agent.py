"""
Host Agent Module

The Host Agent is responsible for:
- High-level planning and task decomposition
- Managing Kanban board (task coordination)
- Coordinating between Grounding and Eval agents
- No direct grounding capabilities (no backend execution)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastagent.agents.base import BaseAgent
from fastagent.llm import LLMClient
from fastagent.kanban import Kanban, CardType, CardStatus
from fastagent.utils.logging import Logger

if TYPE_CHECKING:
    from fastagent.agents.coordinator import AgentCoordinator

logger = Logger.get_logger(__name__)


class HostAgent(BaseAgent):
    def __init__(
        self,
        name: str = "HostAgent",
        llm_client: Optional[LLMClient] = None,
        coordinator: Optional[AgentCoordinator] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """
        Initialize the Host Agent.
        
        Args:
            name: Agent name
            llm_client: LLM client for planning
            coordinator: AgentCoordinator for resource access
            system_prompt: Custom system prompt for planning
        """
        # Host agent has no backend scope (cannot execute)
        super().__init__(
            name=name,
            backend_scope=[],
            llm_client=llm_client,
            coordinator=coordinator
        )
        
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._backend_descriptions: Dict[str, str] = {}
        
        logger.info(f"Host Agent initialized: {name}")

    @property
    def kanban(self) -> Optional[Kanban]:
        """
        Get the shared Kanban board from coordinator.
        
        The Kanban board is used for task management instead of a separate TodoList.
        Tasks are represented as Kanban cards with TODO/IN_PROGRESS/DONE/BLOCKED status.
        """
        if self._coordinator:
            return self._coordinator.kanban
        return None

    def set_backend_descriptions(self, descriptions: Dict[str, str]) -> None:
        """
        Set backend and server descriptions from system backend.
        
        Args:
            descriptions: Dictionary mapping backend names to their descriptions
        """
        self._backend_descriptions = descriptions
        logger.info(f"Host Agent: Received descriptions for {len(descriptions)} backends")

    def _default_system_prompt(self) -> str:
        """
        Default system prompt for the host agent.
        Backend-specific information will be injected dynamically.
        """
        return """You are a Host Agent responsible for task planning and coordination in a multi-agent system. Analyze requests and decompose into executable subtasks with clear dependencies.

## Card Types

### RESPONSE (User-Facing, max 1 per request)
Two modes:
- **"direct"**: Answer immediately from your knowledge
- **"depends_on_execution"**: Answer needs execution results
  - Set `preliminary_answer` (specific, e.g., "Step 1/2: Searching... Step 2/2: Creating presentation...")
  - Mark supporting EXECUTION cards with `for_response_card: true`
  - System auto-waits for ALL supporting cards to complete

### EXECUTION (Internal Operations, as many as needed)
- Concrete operations, standalone or supporting RESPONSE
- **MUST** have descriptive `temp_id` (e.g., "search_articles", NOT "step1")
- Description **SPECIFIC** to THIS subtask, NOT entire user request
- Use `depends_on: ["other_temp_id"]` to chain dependent steps

## Task Decomposition

Your job: Break down user requests into clear, executable steps.

### When to Split into Multiple EXECUTION Cards

Split when:
1. **Switching backends/tools**: Web search → Terminal commands → GUI automation
2. **Knowledge before action**: "Get info" card → "Use that info" card  
3. **Clear dependencies**: Step B needs Step A's output

Keep together when:
- Single tool/backend handles everything
- Tightly coupled steps (fetch + save data)
- Simple atomic operations

### Task Description Guidelines

Be specific about WHAT, not HOW:

Good:
- "Search web for AI news, save results to 'news.json'"
- "Fetch open issues from HKUDS/VideoRAG repo, save to 'issues.json'"
- "Create PowerPoint from 'news.json' with summary slides"

Bad:
- "Handle the data" (what data? how?)
- "Do the search and make slides" (multiple unclear steps)
- "Use shell_agent to call API 3 times" (too prescriptive)

Key points:
- State inputs/outputs clearly (file names, data sources)
- One clear objective per card
- Let GroundingAgent decide tools and methods

### Dependencies

Use `depends_on: ["card_id"]` when one task needs another's results.

### Optional Metadata

**task_category**: General hint (data_retrieval/content_creation/ui_interaction/analysis)  
**preferred_backend**: Only if user explicitly requests specific tool

Don't specify: tool names, iteration counts, completion criteria

## Output Format

```json
{
    "thought": "Analysis and strategy",
    "needs_user_response": true/false,
    "plan": ["Step 1: ...", "Step 2: ..."],
    "task_updates": [
        {
            "action": "add",
            "temp_id": "descriptive_name",
            "title": "Brief title",
            "description": "Specific description with completion criteria",
            "card_type": "response"/"execution",
            "metadata": {
                // RESPONSE cards:
                "response_mode": "direct"/"depends_on_execution",
                "answer": "..." (direct mode),
                "preliminary_answer": "..." (depends_on_execution),
                "execution_plan": [...],
                
                // EXECUTION cards (all optional):
                "task_category": "data_retrieval/content_creation/ui_interaction/analysis",
                "preferred_backend": "..." (only if explicitly required),
                "depends_on": [...],
                "for_response_card": true/false,
                "step_order": 1
            }
        }
    ],
    "user_response": "Optional message to user"
}
```

**REQUIRED**: `temp_id` (descriptive), `description` (specific), `card_type`

**Card types you can create**: response, execution  
**Initial card status**: All new cards start as **TODO** (managed automatically by system)"""
    

    def construct_messages(
        self,
        user_request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Construct messages for LLM planning.
        """
        messages = [{"role": "system", "content": self._system_prompt}]
        
        # Add available backends with tools and capabilities
        if self._backend_descriptions:
            backend_info_lines = ["## Available Backends\n"]
            backend_info_lines.append("### Guidelines\n")
            backend_info_lines.append("- Use `preferred_backend` to suggest a backend (optional hint)")
            backend_info_lines.append("- GroundingAgent has access to ALL tools and makes final selection")
            backend_info_lines.append("- Choose based on task requirements and tool capabilities below\n")
            
            for backend, desc in self._backend_descriptions.items():
                backend_info_lines.append(f"### {backend.upper()}\n")
                backend_info_lines.append(desc)
                backend_info_lines.append("")  # Empty line between backends
            
            messages.append({"role": "system", "content": "\n".join(backend_info_lines)})
        
        # Check for evaluation feedback (most specific)
        has_eval_feedback = context and "eval_feedback" in context
        
        # Check for BLOCKED tasks (execution failures)
        kanban = self.kanban
        blocked_details = []
        if kanban and context and context.get("include_kanban", True):
            blocked_details = kanban.get_blocked_details()
        
        # Add appropriate context based on priority
        if has_eval_feedback:
            # EVAL FEEDBACK MODE: EvalAgent identified task failure
            eval_feedback_context = self._construct_eval_feedback_context(context["eval_feedback"])
            messages.append({
                "role": "system",
                "content": eval_feedback_context
            })
            logger.info("Host Agent: Eval feedback mode activated - replanning based on evaluation")
            
            # Still show Kanban summary for overall context
            if kanban:
                kanban_summary = kanban.get_summary()
                messages.append({
                    "role": "system",
                    "content": f"## Current Progress (Kanban)\n\n{kanban_summary}"
                })
        
        elif blocked_details:
            # BLOCKED REPLAN MODE: Direct execution failures (not caught by EvalAgent yet)
            replan_context = self._construct_replan_context(blocked_details)
            messages.append({
                "role": "system",
                "content": replan_context
            })
            logger.info(f"Host Agent: Replan mode activated - {len(blocked_details)} BLOCKED task(s) detected")
        
        elif kanban and context and context.get("include_kanban", True):
            # NORMAL MODE: No failures, provide regular Kanban summary
            kanban_summary = kanban.get_summary()
            messages.append({
                "role": "system",
                "content": f"## Current Progress (Kanban)\n\n{kanban_summary}"
            })
        
        # Add user request
        messages.append({"role": "user", "content": user_request})
        
        return messages

    def _construct_eval_feedback_context(self, eval_feedback: Dict[str, Any]) -> str:
        """
        Construct evaluation feedback context for replanning after task failure.
        
        This provides structured feedback from EvalAgent to help LLM make better replanning decisions.
        
        Args:
            eval_feedback: Dictionary with evaluation results:
                - success: bool (always False if we're here)
                - confidence: str (high/medium/low)
                - evaluation: str (evaluation text)
                - issues: List[str] (identified problems)
                - suggestions: List[str] (suggestions for improvement)
                - execution_card_id: str
                - execution_title: str
                - evaluation_type: str (optional, e.g., "status_determination")
        """
        eval_type = eval_feedback.get("evaluation_type", "regular")
        eval_type_label = "STATUS DETERMINATION" if eval_type == "status_determination" else "EVALUATION"
        
        context = f"""## {eval_type_label}: TASK FAILURE DETECTED

**ALERT**: Previous task execution failed evaluation.  
Analyze the failure details below and create alternative approaches.

"""
        
        # Basic information
        context += f"""**Failed Task**: {eval_feedback.get('execution_title', 'Unknown')}  
**Execution Card**: {eval_feedback.get('execution_card_id', 'Unknown')}  
**Evaluation Result**: FAILED (`success=False`)  
**Confidence**: {eval_feedback.get('confidence', 'unknown')}

"""
        
        # Evaluation summary
        if eval_feedback.get('evaluation'):
            context += f"""### Evaluation Summary

{eval_feedback['evaluation']}

"""
        
        # Reasoning (for status determination)
        if eval_feedback.get('reasoning'):
            context += f"""### Reasoning

{eval_feedback['reasoning']}

"""
        
        # Evidence (for status determination)
        if eval_feedback.get('evidence'):
            evidence = eval_feedback['evidence']
            if evidence:
                context += f"### Evidence ({len(evidence)})\n\n"
                for ev in evidence:
                    context += f"- {ev}\n"
                context += "\n"
        
        # Issues identified
        if eval_feedback.get('issues'):
            issues = eval_feedback['issues']
            if issues:
                context += f"### Issues Identified ({len(issues)})\n\n"
                for issue in issues:
                    context += f"- {issue}\n"
                context += "\n"
        
        # Suggestions for improvement
        suggestions = eval_feedback.get('suggestions', [])
        if suggestions:
            context += f"### Suggestions for Improvement ({len(suggestions)})\n\n"
            for suggestion in suggestions:
                context += f"- {suggestion}\n"
            context += "\n"
        
        # Action required
        context += """### ACTION REQUIRED

Create new EXECUTION cards with alternative approaches.  
Address the issues and follow the suggestions provided above.
"""
        
        return context
    
    def _construct_replan_context(self, blocked_details: List[Dict[str, Any]]) -> str:
        """
        Construct detailed replan context for BLOCKED tasks.
        
        This provides comprehensive information about failed tasks to help LLM
        make informed replanning decisions:
        - Why the task failed (error messages)
        - What was attempted (tool calls and parameters)
        - How many times it was tried (iteration count)
        - Related tasks and dependencies
        - Original task context
        
        Args:
            blocked_details: List of detailed BLOCKED card information
        """
        if not blocked_details:
            return ""
        
        # Build header
        context = f"""## REPLAN: BLOCKED TASKS DETECTED

**ALERT**: {len(blocked_details)} BLOCKED task(s) require replanning.  
Analyze failure details below and create alternative approaches.

### Replan Guidelines

- **DO NOT** simply retry - analyze WHY it failed
- Consider alternative tools, parameters, or strategies
- Verify dependencies completed successfully
- Break complex tasks into smaller steps
- If truly impossible, explain why and suggest alternatives

"""
        
        # Build details for each blocked task
        for idx, detail in enumerate(blocked_details, 1):
            context += f"""### BLOCKED TASK #{idx}

**{detail['title']}**

*Task*: {detail['description']}

**ERROR**:
```
{detail['error']}
```

"""
            
            # Tool execution history
            if "tool_executions" in detail:
                tool_execs = detail["tool_executions"]
                context += f"**Attempted Tools** ({len(tool_execs)}):\n"
                for i, exec_info in enumerate(tool_execs, 1):
                    tool_name = exec_info['tool_name']
                    backend = exec_info['backend']
                    status = exec_info['status']
                    error_info = f" - {exec_info['error'][:100]}" if exec_info.get('error') else ""
                    context += f"{i}. `{backend}.{tool_name}` → `{status}`{error_info}\n"
                context += "\n"
            
            # Show iteration count if multiple attempts
            if "result" in detail:
                result = detail["result"]
                iterations = result.get('iterations', 0)
                if iterations > 1:
                    context += f"*Tried {iterations} times before blocking*\n\n"
            
            # Show related tasks only if there are dependencies
            if "related_cards" in detail:
                related = detail["related_cards"]
                incomplete = [c for c in related if c['status'] != 'done']
                if incomplete:
                    context += f"**Incomplete Dependencies** ({len(incomplete)}):\n"
                    for card in incomplete[:3]:
                        status_marker = "[IN_PROGRESS]" if card['status'] == "in_progress" else "[TODO]"
                        context += f"- {status_marker} {card['title']}\n"
                    context += "\n"
        
        # Add final action required
        context += """### ACTION REQUIRED

Create new EXECUTION cards with alternative approaches.  
Address root causes identified above.
"""
        
        return context

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a planning request.
        
        Args:
            context: Context dictionary with:
                - user_request: The user's request
                - eval_feedback: Optional feedback from eval agent
                - include_kanban: Whether to include kanban state
                
        Returns:
            Response dictionary with plan and instructions
        """
        user_request = context.get("user_request", "")
        if not user_request:
            logger.error("Host Agent: No user request provided")
            return {"error": "No user request provided"}
        
        logger.info(f"Host Agent: Processing request at step {self.step}")
        
        # Construct messages
        messages = self.construct_messages(user_request, context)
        
        # Get LLM response
        try:
            response = await self.get_llm_response(messages)
            
            # Extract response content
            response_content = self._extract_response_content(response)
            response_dict = self.response_to_dict(response_content)
            
            logger.info(f"Host Agent LLM Response (full): {response_content}")
            # logger.info(f"Host Agent parsed dict: {response_dict}")
            logger.info(f"Host Agent parsed task_updates: {response_dict.get('task_updates', [])}")
            
            # Update Kanban tasks based on response
            await self._process_task_updates(response_dict.get("task_updates", []))
            
            # Record agent action to recording manager
            if self._coordinator and self._coordinator.recording_manager:
                await self._coordinator.recording_manager.record_agent_action(
                    agent_name=self.name,
                    action_type="plan",
                    input_data={"user_request": user_request},
                    reasoning={
                        "thought": response_dict.get("thought", ""),
                        "plan": response_dict.get("plan", []),
                    },
                    output_data={
                        "task_updates": response_dict.get("task_updates", []),
                        "message_to_grounding": response_dict.get("message_to_grounding", ""),
                        "status": response_dict.get("status", ""),
                    },
                    metadata={
                        "step": self.step,
                        "num_task_updates": len(response_dict.get("task_updates", [])),
                    }
                )
            
            # Increment step
            self.increment_step()
            
            logger.info(f"Host Agent: Generated plan with {len(response_dict.get('plan', []))} steps")
            return response_dict
            
        except Exception as e:
            logger.error(f"Host Agent: Processing failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "status": "error",
                "thought": f"Failed to generate plan: {e}"
            }

    def _extract_response_content(self, response: Dict[str, Any]) -> str:
        """Extract text content from LLM response."""
        # The format returned by LLMClient.complete(): {"message": {...}, "tool_results": [...], "messages": [...]}
        if "message" in response and isinstance(response["message"], dict):
            # The format returned by LLMClient.complete()
            return response["message"].get("content", "")
        elif "choices" in response and response["choices"]:
            # Standard OpenAI format
            choice = response["choices"][0]
            if "message" in choice:
                return choice["message"].get("content", "")
        elif "content" in response:
            return response["content"]
        return json.dumps(response)

    async def _process_task_updates(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process Kanban task updates from LLM response.
        
        Uses coordinator's unified method for Kanban updates to ensure
        centralized tracking, validation, and event handling.
        
        Args:
            updates: List of update operations for Kanban cards
            
        Returns:
            Summary of executed updates
        """
        if not self._coordinator:
            logger.warning("Host Agent: No coordinator available for task updates")
            return {"error": "No coordinator available"}
        
        current_planning_card_id = None
        
        for update in updates:
            if update.get("action") == "add" and update.get("card_type") == "planning":
                # This will be resolved to real ID by coordinator
                current_planning_card_id = update.get("temp_id") or update.get("logical_id")
                break
        
        # If no new PLANNING card, check Kanban for most recent PLANNING card
        if not current_planning_card_id and self.kanban:
            planning_cards = [
                c for c in self.kanban.get_cards_by_type(CardType.PLANNING)
                if c.status in [CardStatus.TODO, CardStatus.IN_PROGRESS, CardStatus.DONE]
            ]
            if planning_cards:
                # Sort by created_at and get most recent
                planning_cards.sort(key=lambda c: c.created_at, reverse=True)
                current_planning_card_id = planning_cards[0].card_id
                logger.debug(
                    f"Using most recent PLANNING card as planning_card_id: {current_planning_card_id[:20]}..."
                )
        
        # Phase 1: Find RESPONSE card logical_id for explicit linking
        response_logical_id = None
        for update in updates:
            if update.get("action") == "add" and update.get("card_type") == "response":
                response_logical_id = update.get("temp_id") or update.get("logical_id")
                break
        
        # Phase 2: Inject metadata (planning_card_id and target_response_card_id)
        if current_planning_card_id:
            for update in updates:
                if update.get("action") == "add":
                    card_type = update.get("card_type", "")
                    
                    if card_type in ["execution", "response"]:
                        # Initialize metadata if not exists
                        if "metadata" not in update:
                            update["metadata"] = {}
                        
                        # Add planning_card_id
                        update["metadata"]["planning_card_id"] = current_planning_card_id
                        
                        logger.debug(
                            f"Injected planning_card_id={current_planning_card_id[:20]}... "
                            f"into {card_type} card: {update.get('title', 'unknown')}"
                        )
                    
                    # For EXECUTION cards supporting RESPONSE, add explicit reference
                    if card_type == "execution" and response_logical_id:
                        if update["metadata"].get("for_response_card"):
                            update["metadata"]["target_response_logical_id"] = response_logical_id
                            logger.debug(
                                f"Linked EXECUTION '{update.get('title', 'unknown')}' "
                                f"to RESPONSE (logical_id={response_logical_id})"
                            )
        
        # Execute updates through coordinator's unified method
        results = await self._coordinator.execute_kanban_updates(
            updates=updates,
            agent_name=self.name,
            step=self.step
        )
        
        if current_planning_card_id and current_planning_card_id in results.get("logical_id_mapping", {}):
            resolved_id = results["logical_id_mapping"][current_planning_card_id]
            logger.info(
                f"Resolved planning_card_id: {current_planning_card_id} → {resolved_id[:20]}..."
        )
        
        # Log summary
        if results.get("added"):
            logger.info(f"Host Agent: Added {len(results['added'])} Kanban cards")
        if results.get("completed"):
            logger.info(f"Host Agent: Completed {len(results['completed'])} Kanban cards")
        if results.get("updated"):
            logger.info(f"Host Agent: Updated {len(results['updated'])} Kanban cards")
        if results.get("errors"):
            logger.warning(f"Host Agent: {len(results['errors'])} errors during Kanban updates")
            for error in results["errors"]:
                logger.warning(f"  - {error}")
        
        return results