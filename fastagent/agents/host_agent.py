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
from fastagent.prompts import HostAgentPrompts
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
        """Default system prompt for the host agent."""
        return HostAgentPrompts.SYSTEM_PROMPT
    

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
            backend_info_lines = [HostAgentPrompts.backend_guidelines()]
            
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
        """Construct evaluation feedback context for replanning after task failure."""
        return HostAgentPrompts.eval_feedback_context(eval_feedback)
    
    def _construct_replan_context(self, blocked_details: List[Dict[str, Any]]) -> str:
        """Construct detailed replan context for BLOCKED tasks."""
        return HostAgentPrompts.replan_context(blocked_details)

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