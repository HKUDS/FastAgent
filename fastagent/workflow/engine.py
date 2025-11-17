"""
Event-driven workflow engine that coordinates agents through Kanban card states.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime

from fastagent.kanban import Kanban, KanbanCard, CardType, CardStatus
from fastagent.agents.content_processor import ContentLevel
from fastagent.utils.logging import Logger
from .rules import WorkflowRule
from .context_manager import ContextManager

if TYPE_CHECKING:
    from fastagent.agents.coordinator import AgentCoordinator
    from fastagent.agents.base import BaseAgent

logger = Logger.get_logger(__name__)


class TaskExecutionContext:
    """Task execution context for tracking task state"""
    
    def __init__(
        self,
        card_id: str,
        rule_name: str,
        started_at: datetime,
        timeout: Optional[float] = None
    ):
        self.card_id = card_id
        self.rule_name = rule_name
        self.started_at = started_at
        self.timeout = timeout
        self.task: Optional[asyncio.Task] = None
    
    def is_timeout(self) -> bool:
        """Check if task has exceeded timeout"""
        if self.timeout is None:
            return False
        elapsed = (datetime.now() - self.started_at).total_seconds()
        return elapsed > self.timeout
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return (datetime.now() - self.started_at).total_seconds()


class WorkflowEngine:
    """
    Workflow Engine: Kanban-driven event loop
    """
    
    def __init__(
        self,
        coordinator: AgentCoordinator,
        poll_interval: float = 1.0,
        task_default_timeout: float = 300.0,
        auto_evaluate: bool = True,
        evaluation_config: Optional[Any] = None,
    ):
        """
        Initialize workflow engine
        
        Args:
            coordinator: Agent coordinator
            poll_interval: Polling interval in seconds
            task_default_timeout: Default task timeout in seconds
            auto_evaluate: Enable automatic evaluation (EvalAgent)
            evaluation_config: Evaluation configuration
        """
        self.coordinator = coordinator
        self.kanban: Kanban = coordinator.kanban
        self.poll_interval = poll_interval
        self.max_concurrent_tasks = 1  # Serial execution only
        self.task_default_timeout = task_default_timeout
        self.auto_evaluate = auto_evaluate
        
        # Evaluation configuration
        self.evaluation_config = evaluation_config
        
        self.rules: List[WorkflowRule] = []
        
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None
        
        # Processed (card_id, rule_name) pairs => timestamp
        self._processed_pairs: Dict[Tuple[str, str], datetime] = {}
        
        # Active task contexts
        self._active_tasks: Dict[str, TaskExecutionContext] = {}
        
        self._stats = {
            "total_processed": 0,
            "total_success": 0,
            "total_failed": 0,
            "total_timeout": 0,
        }
        
        # Context Manager for managing execution context across cards
        from fastagent.workflow.context_manager import ContextManager
        
        if not hasattr(coordinator, 'storage_manager'):
            raise ValueError(
                "WorkflowEngine: coordinator.storage_manager is required. "
                "Please ensure AgentCoordinator is properly initialized."
            )
        
        self._context_manager = ContextManager(
            storage_manager=coordinator.storage_manager
        )
        
        eval_mode = self.evaluation_config.mode if self.evaluation_config else "all"
        logger.info(
            f"WorkflowEngine initialized (evaluation_mode={eval_mode})"
        )
    
    def _mark_pair_processed(self, card_id: str, rule_name: str) -> None:
        """Mark (card_id, rule_name) as processed with timestamp"""
        self._processed_pairs[(card_id, rule_name)] = datetime.now()
    
    @staticmethod
    def _to_datetime(value: Any) -> datetime:
        """Convert string/datetime/None to datetime object"""
        if isinstance(value, datetime):
            return value
        if value is None:
            return datetime.min
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return datetime.min
    
    def add_rule(self, rule: WorkflowRule) -> None:
        """Add workflow rule"""
        existing = next((r for r in self.rules if r.name == rule.name), None)
        if existing:
            logger.warning(f"Rule '{rule.name}' already exists, replacing")
            self.remove_rule(rule.name)
        
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Added rule '{rule.name}' (priority={rule.priority})")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove workflow rule by name"""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                self.rules.pop(i)
                logger.info(f"Removed rule '{rule_name}'")
                return True
        return False
    
    def get_rule(self, rule_name: str) -> Optional[WorkflowRule]:
        """Get rule by name"""
        return next((r for r in self.rules if r.name == rule_name), None)
    
    def _should_evaluate_execution(self, card: KanbanCard) -> bool:
        """
        Check if an EXECUTION card should be evaluated
        Uses EvaluationConfig.should_evaluate() method for unified logic.
        """
        if not self.evaluation_config:
            # No config, evaluate all by default
            return True
        
        # Get backend information
        backend = card.metadata.get("backend") or card.metadata.get("preferred_backend") or "unknown"
        
        # Check if this is the last execution
        is_last = self._is_last_execution(card)
        
        # Use EvaluationConfig's unified evaluation logic
        should_eval = self.evaluation_config.should_evaluate(backend=backend, is_last=is_last)
        
        if should_eval:
            reason = self._get_eval_reason(card, backend, is_last)
            logger.info(f"Card {card.card_id[:30]}... will be evaluated ({reason})")
        else:
            logger.debug(
                f"Card {card.card_id[:30]}... evaluation conditions not met "
                f"(mode={self.evaluation_config.mode}, backend={backend}, is_last={is_last})"
            )
        
        return should_eval
    
    def _is_last_execution(self, card: KanbanCard) -> bool:
        all_cards = list(self.kanban._cards.values())
        execution_cards = [
            c for c in all_cards
            if c.card_type == CardType.EXECUTION
            and c.card_id != card.card_id  # Exclude current card
        ]
        
        # Check if any other EXECUTION is still pending
        pending_executions = [
            c for c in execution_cards
            if c.status in [CardStatus.TODO, CardStatus.IN_PROGRESS]
        ]
        
        is_last = len(pending_executions) == 0
        
        if is_last:
            logger.info(
                f"Card {card.card_id[:30]}... is the LAST execution, will be evaluated"
            )
        
        return is_last
    
    def _get_eval_reason(self, card: KanbanCard, backend: str, is_last: bool) -> str:
        """Get evaluation reason"""
        if not self.evaluation_config:
            return "mode: all (default)"
        
        mode = self.evaluation_config.mode
        
        if mode == "none":
            return "mode: none"
        
        if mode == "all":
            return "mode: all"
        
        if mode == "last_only":
            return "mode: last_only" if is_last else "not last"
        
        if mode == "selective":
            reasons = []
            
            # Check if this is the last execution
            if is_last and self.evaluation_config.always_eval_last:
                reasons.append("last execution")
            
            # Check backend
            if backend and self.evaluation_config.backends and backend in self.evaluation_config.backends:
                reasons.append(f"backend={backend}")
            
            if reasons:
                return f"mode: selective ({', '.join(reasons)})"
            else:
                return "mode: selective"
        
        return "mode: unknown"
    
    async def _link_next_execution_to_current(self, completed_execution: KanbanCard) -> None:
        """
        Link the next TODO EXECUTION to depend on the current completed EXECUTION.
        
        This is used when NO EVALUATION is created (due to evaluation config).
        It ensures sequential execution by making the next EXECUTION wait for the current one.
        
        Args:
            completed_execution: The EXECUTION card that just completed (no evaluation will be created)
        """
        next_execution = await self._find_next_execution(completed_execution)
        
        if not next_execution:
            logger.debug(
                f"No next EXECUTION found for {completed_execution.card_id[:20]}... "
                f"(no evaluation, no further linking needed)"
            )
            return
        
        # Check if next_execution already depends on completed_execution
        current_deps = next_execution.metadata.get("depends_on", [])
        if not isinstance(current_deps, list):
            current_deps = [current_deps] if current_deps else []
        
        if completed_execution.card_id not in current_deps:
            current_deps.append(completed_execution.card_id)
            next_execution.metadata["depends_on"] = current_deps
            logger.info(
                f"✓ Linked EXECUTION {completed_execution.card_id[:20]}... → EXECUTION {next_execution.card_id[:20]}...\n"
                f"  (No evaluation created, but ensuring sequential execution to limit error propagation)"
            )
    
    async def _find_next_execution(self, completed_execution: KanbanCard) -> Optional[KanbanCard]:
        """
        Find the next TODO EXECUTION card that should execute after the current one.
        
        This is a shared helper used by both:
        - _link_evaluation_to_next_execution (when EVALUATION is created)
        - _link_next_execution_to_current (when NO EVALUATION is created)
        
        Args:
            completed_execution: The EXECUTION that just completed
            
        Returns:
            The next TODO EXECUTION card, or None if not found
        """
        planning_card_id = completed_execution.metadata.get("planning_card_id")
        current_step_order = completed_execution.metadata.get("step_order")
        
        if not planning_card_id:
            logger.debug(
                f"No planning_card_id found for EXECUTION {completed_execution.card_id[:20]}..., "
                f"cannot find next execution"
            )
            return None
        
        # Find all TODO EXECUTION cards from the same planning context
        all_cards = list(self.kanban._cards.values())
        candidate_executions = [
            c for c in all_cards
            if c.card_type == CardType.EXECUTION
            and c.status == CardStatus.TODO
            and c.metadata.get("planning_card_id") == planning_card_id
        ]
        
        if not candidate_executions:
            logger.debug(
                f"No TODO EXECUTION cards found in same planning context ({planning_card_id[:20]}...)"
            )
            return None
        
        # Find the next EXECUTION card to execute
        # Strategy: Find the card with the smallest step_order that is larger than current_step_order
        # If no step_order available, use the earliest created_at
        next_execution = None
        
        if current_step_order is not None:
            # Find next by step_order
            next_candidates = [
                c for c in candidate_executions
                if c.metadata.get("step_order") is not None
                and c.metadata.get("step_order") > current_step_order
            ]
            if next_candidates:
                next_candidates.sort(key=lambda c: c.metadata.get("step_order", float('inf')))
                next_execution = next_candidates[0]
        
        # Fallback: Find the earliest created TODO EXECUTION
        if not next_execution:
            # Check if any of the candidates don't have explicit depends_on
            # These are likely to be executed next
            independent_candidates = [
                c for c in candidate_executions
                if not c.metadata.get("depends_on")
            ]
            
            if independent_candidates:
                # Use the earliest created one
                independent_candidates.sort(key=lambda c: c.created_at)
                next_execution = independent_candidates[0]
            elif candidate_executions:
                # All have dependencies, pick the earliest
                candidate_executions.sort(key=lambda c: c.created_at)
                next_execution = candidate_executions[0]
        
        return next_execution
    
    async def _link_evaluation_to_next_execution(
        self,
        eval_card: KanbanCard,
        completed_execution: KanbanCard
    ) -> None:
        """
        Link the EVALUATION card to the next TODO EXECUTION card.
        """
        next_execution = await self._find_next_execution(completed_execution)
        
        if not next_execution:
            logger.debug(
                f"No next EXECUTION found for {completed_execution.card_id[:20]}..., "
                f"evaluation will not block any future executions"
            )
            return
        
        # Update the next EXECUTION card's depends_on to include this EVALUATION
        current_deps = next_execution.metadata.get("depends_on", [])
        if not isinstance(current_deps, list):
            current_deps = [current_deps] if current_deps else []
        
        # Add the EVALUATION card as a dependency if not already present
        if eval_card.card_id not in current_deps:
            current_deps.append(eval_card.card_id)
            next_execution.metadata["depends_on"] = current_deps
            
            logger.info(
                f"✓ Linked EVALUATION {eval_card.card_id[:20]}... → EXECUTION {next_execution.card_id[:20]}...\n"
                f"  This ensures evaluation completes before next execution, preventing error accumulation.\n"
                f"  Execution chain: {completed_execution.title} (DONE) → "
                f"Evaluate (TODO) → {next_execution.title} (WAITING)"
            )
        else:
            logger.debug(
                f"EVALUATION {eval_card.card_id[:20]}... already in dependencies of "
                f"EXECUTION {next_execution.card_id[:20]}..."
            )
    
    def register_default_rules(self) -> None:
        """
        Register default workflow rules
        
        Default rules:
        1. PLANNING/TODO → HostAgent
        2. RESPONSE/TODO → Auto-complete based on response_mode
           - direct: Complete immediately with metadata.answer
           - depends_on_execution: Wait for supporting EXECUTION cards
        3. EXECUTION/TODO → GroundingAgent
        4. EXECUTION/DONE → Update RESPONSE card (if for_response_card=true)
        5. EXECUTION/DONE → Create EVALUATION card (only if auto_evaluate=True and not for_response_card)
        6. EVALUATION/TODO → EvalAgent (only if auto_evaluate=True)
        """
        self.add_rule(WorkflowRule(
            name="planning_to_host",
            card_type=CardType.PLANNING,
            card_status=CardStatus.TODO,
            agent_name="HostAgent",
            priority=100,
        ))
        
        async def complete_response_card(card: KanbanCard) -> None:
            """
            Auto-complete RESPONSE card for direct mode.
            
            Note: depends_on_execution mode is now handled immediately during card creation
            by the Coordinator, so this rule only processes direct answers.
            """
            response_mode = card.metadata.get("response_mode", "direct")
            
            if response_mode == "direct":
                # Direct answer mode - complete immediately
                answer = card.metadata.get("answer", "")
                
                if not answer:
                    logger.warning(f"RESPONSE card {card.card_id} has no answer in direct mode")
                    error_answer = (
                        "Unable to provide an answer. "
                        "The response card was created without an answer field. "
                        "This is likely a system error - please check the logs."
                    )
                    card.result = {
                        "status": "error",
                        "answer": error_answer,
                        "error": "No answer provided by HostAgent",
                    }
                    card.metadata["result"] = card.result
                    card.metadata["error"] = "No answer provided"
                    await self.kanban.update_card_status(card.card_id, CardStatus.DONE)
                    logger.info(f"RESPONSE card {card.card_id} completed with error (degraded)")
                    return
                
                # Complete with direct answer
                card.result = {
                    "status": "completed",
                    "answer": answer,
                    "response_mode": "direct",
                }
                card.metadata["result"] = card.result
                await self.kanban.update_card_status(card.card_id, CardStatus.DONE)
                logger.info(f"RESPONSE card {card.card_id} completed with direct answer")
            
            elif response_mode == "depends_on_execution":
                # If see a TODO RESPONSE with depends_on_execution here, something went wrong
                logger.warning(
                    f"RESPONSE card {card.card_id} with depends_on_execution mode is still TODO. "
                    f"This should have been handled during creation. Skipping."
                )
            
            else:
                logger.error(f"Unknown response_mode '{response_mode}' for RESPONSE card {card.card_id}")
                card.result = {
                    "status": "error",
                    "answer": "Internal error: Unknown response mode",
                    "error": f"Unknown response_mode: {response_mode}",
                }
                card.metadata["result"] = card.result
                await self.kanban.update_card_status(card.card_id, CardStatus.BLOCKED)
        
        self.add_rule(WorkflowRule(
            name="response_auto_complete",
            card_type=CardType.RESPONSE,
            card_status=CardStatus.TODO,
            agent_name=None,  # Hook-only rule, no agent needed
            priority=95,
            pre_hook=complete_response_card,
        ))
        
        self.add_rule(WorkflowRule(
            name="execution_to_grounding",
            card_type=CardType.EXECUTION,
            card_status=CardStatus.TODO,
            agent_name="GroundingAgent",
            priority=90,
        ))
        
        # When supporting EXECUTION completes, update the RESPONSE card
        async def update_response_with_execution(card: KanbanCard) -> None:
            """
            When supporting EXECUTION card completes, integrate results into RESPONSE card.
            """
            # Check if this EXECUTION is for a RESPONSE card
            for_response_card = card.metadata.get("for_response_card", False)
            if not for_response_card:
                # Not a supporting execution, skip
                return
            
            # Find the RESPONSE card that's awaiting this execution
            # Strategy 1: Use explicit target_response_card_id
            target_response_id = card.metadata.get("target_response_card_id")
            response_card = None
            
            if target_response_id:
                response_card = self.kanban.get_card(target_response_id)
                if not response_card:
                    logger.error(
                        f"EXECUTION {card.card_id[:30]}... has target_response_card_id={target_response_id[:30]}... "
                        f"but RESPONSE card not found. Card may have been deleted."
                    )
                    return
                
                if response_card.status not in [CardStatus.IN_PROGRESS, CardStatus.TODO]:
                    logger.warning(
                        f"Target RESPONSE card {target_response_id[:30]}... found but not IN_PROGRESS/TODO "
                        f"(status: {response_card.status}). Execution may have completed already."
                    )
                    return
                
                logger.debug(
                    f"✓ Using explicit link: EXECUTION {card.card_id[:20]}... → RESPONSE {target_response_id[:20]}..."
                )
            else:
                # Strategy 2: Fallback to finding IN_PROGRESS RESPONSE
                logger.debug(
                    f"EXECUTION {card.card_id[:30]}... has no target_response_card_id, "
                    f"falling back to search-based matching"
                )
                
                all_cards = list(self.kanban._cards.values())
                response_cards = [
                    c for c in all_cards
                    if c.card_type == CardType.RESPONSE
                    and c.status == CardStatus.IN_PROGRESS
                    and c.metadata.get("awaiting_execution")
                ]
                
                if not response_cards:
                    logger.error(
                        f"CRITICAL: No awaiting RESPONSE card found for EXECUTION {card.card_id[:30]}... "
                        f"This EXECUTION was marked for_response_card=true but no RESPONSE is waiting. "
                        f"Possible causes: "
                        f"1) RESPONSE completed prematurely (check expected_execution_count), "
                        f"2) RESPONSE was never created, "
                        f"3) RESPONSE already failed/blocked."
                    )
                    return
                
                # Use the most recent RESPONSE card (sort by created_at timestamp)
                response_cards.sort(
                    key=lambda c: c.metadata.get("created_at_timestamp", 0), 
                    reverse=True
                )
                response_card = response_cards[0]
                logger.warning(
                    f"Matched EXECUTION {card.card_id[:20]}... to RESPONSE {response_card.card_id[:20]}... "
                    f"using fallback search (from {len(response_cards)} candidate(s)). "
                    f"Consider using explicit target_response_card_id for reliability."
                )
            
            if not response_card:
                logger.error(f"Failed to find RESPONSE card for execution {card.card_id}")
                return
            
            # Get execution result
            exec_result = getattr(card, 'result', None) or card.metadata.get("result", {})
            
            # Add execution result to RESPONSE card's collection (thread-safe append)
            execution_results = response_card.metadata.get("execution_results", [])
            execution_results.append({
                "execution_card_id": card.card_id,
                "execution_title": card.title,
                "result": exec_result,
            })
            response_card.metadata["execution_results"] = execution_results
            response_card.metadata["execution_count"] = len(execution_results)
            
            # If it's still TODO, something went wrong
            if response_card.status == CardStatus.TODO:
                logger.error(
                    f"RESPONSE card {response_card.card_id[:20]}... is still TODO! "
                    f"This should not happen with the new immediate transition logic. "
                    f"Skipping execution result integration."
                )
                return
            
            # Check if we should complete the RESPONSE card
            # Only complete when all expected executions are done
            expected_count = response_card.metadata.get("expected_execution_count", 1)
            current_count = len(execution_results)
            
            logger.info(
                f"✓ RESPONSE {response_card.card_id[:30]}...: received execution result {current_count}/{expected_count} "
                f"from EXECUTION {card.card_id[:30]}..."
            )
            
            # Check for potential count mismatch
            if current_count > expected_count:
                logger.warning(
                    f"WARNING: RESPONSE {response_card.card_id[:30]}... received MORE executions than expected! "
                    f"Current: {current_count}, Expected: {expected_count}. "
                    f"This indicates expected_execution_count was set incorrectly. "
                    f"Completing anyway to avoid blocking."
                )
            
            if current_count >= expected_count:
                # All executions complete - finalize the RESPONSE with intelligent content processing
                preliminary_answer = response_card.metadata.get("preliminary_answer", "")
                execution_plan = response_card.metadata.get("execution_plan", [])
                
                # Use ContextManager's ContentProcessor to intelligently process and polish response content
                try:
                    # Create ProcessedContent for each execution
                    processed_contents = []
                    for exec_info in execution_results:
                        exec_result = exec_info.get("result", {})
                        exec_title = exec_info.get("execution_title", "Unknown")
                        exec_card_id = exec_info.get("execution_card_id", "")
                        
                        # Get execution card's metadata
                        exec_card = self.kanban.get_card(exec_card_id)
                        exec_metadata = exec_card.metadata if exec_card else {}
                        
                        # Process execution result using ContentProcessor
                        processed = self._context_manager.content_processor.process_execution_result(
                            card_id=exec_card_id,
                            title=exec_title,
                            result=exec_result,
                            metadata=exec_metadata,
                            target_level=ContentLevel.SUMMARY
                        )
                        processed_contents.append(processed)
                    
                    # Polish using ContentProcessor (if LLM available)
                    polished_response = await self._context_manager.content_processor.polish_response(
                        executions=processed_contents,
                        original_task=response_card.metadata.get("original_task", response_card.description),
                        execution_plan=execution_plan
                    )
                    
                    # Build final Response
                    response_card.result = {
                        "status": "completed",
                        "answer": polished_response.get("answer", preliminary_answer),
                        "response_mode": "depends_on_execution",
                        "execution_enhanced": True,
                        "execution_count": current_count,
                        # Keep reference to original execution results (for debugging)
                        "execution_results_summary": [
                            {
                                "card_id": e["execution_card_id"],
                                "title": e["execution_title"],
                                "status": e.get("result", {}).get("status", "unknown")
                            }
                            for e in execution_results
                        ],
                        # Include important data and polished information
                        "important_data": polished_response.get("important_data", []),
                        "key_findings": polished_response.get("key_findings", []),
                        "polished": True,
                    }
                    
                    logger.info(
                        f"✓ Response polished with {len(polished_response.get('important_data', []))} "
                        f"important data items"
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to polish response: {e}", exc_info=True)
                    # Fallback: use simple summary
                    response_card.result = {
                        "status": "completed",
                        "answer": preliminary_answer,
                        "response_mode": "depends_on_execution",
                        "execution_enhanced": True,
                        "execution_results": execution_results,
                        "polished": False,
                        "polish_error": str(e)
                    }
                
                response_card.metadata["result"] = response_card.result
                response_card.metadata["awaiting_execution"] = False
                
                # Mark RESPONSE card as DONE
                await self.kanban.update_card_status(response_card.card_id, CardStatus.DONE)
                logger.info(
                    f"RESPONSE {response_card.card_id[:30]}... completed with {current_count} execution result(s)"
                )
            else:
                # Still waiting for more executions
                logger.info(
                    f"⏳ RESPONSE {response_card.card_id[:30]}... still waiting for {expected_count - current_count} more execution(s)"
                )
        
        self.add_rule(WorkflowRule(
            name="execution_done_update_response",
            card_type=CardType.EXECUTION,
            card_status=CardStatus.DONE,
            agent_name=None,  # Hook-only rule
            priority=85,
            pre_hook=update_response_with_execution,
        ))

        async def link_next_execution_after_done(card: KanbanCard) -> None:
            """
            After EXECUTION completes, link the next TODO EXECUTION to wait for its evaluation.
            
            This prevents error accumulation by ensuring sequential execution with evaluation checks.
            
            Two scenarios:
            1. If EVALUATION will be created: Next EXECUTION will depend on EVALUATION
            2. If NO EVALUATION: Next EXECUTION will depend on this EXECUTION (already done by default)
            
            The linking happens in create_evaluation_card if evaluation is created,
            or here if evaluation is skipped.
            """
            
            # Skip supporting executions for RESPONSE cards
            if card.metadata.get("for_response_card"):
                return
            
            # Skip if evaluation will be created (it will handle linking)
            if self.auto_evaluate and self._should_evaluate_execution(card):
                # EVALUATION will be created, let create_evaluation_card handle linking
                return
            
            # No EVALUATION will be created, link next EXECUTION to this EXECUTION
            await self._link_next_execution_to_current(card)
        
        self.add_rule(WorkflowRule(
            name="execution_done_link_chain",
            card_type=CardType.EXECUTION,
            card_status=CardStatus.DONE,
            agent_name=None,  # Hook-only rule
            priority=82,  # Run after response update but before evaluation creation
            pre_hook=link_next_execution_after_done,
        ))
        
        if self.auto_evaluate:
            async def create_evaluation_card(card: KanbanCard) -> None:
                """
                Create EVALUATION card for completed EXECUTION (regular evaluation)
                
                This is for REGULAR evaluation (after EXECUTION completes successfully).
                It is controlled by EvaluationConfig (mode, backends, etc.).
                
                For STATUS DETERMINATION evaluation (mixed success/failure), 
                see _create_status_evaluation_card() which bypasses config checks.
                """
                # Check if this card already had status determination
                had_status_check = card.metadata.get("status_checked", False)
                if had_status_check:
                    logger.info(
                        f"EXECUTION {card.card_id[:30]}... had status determination - "
                        f"now performing regular quality evaluation"
                    )
                
                # Skip supporting executions for RESPONSE cards
                if card.metadata.get("for_response_card"):
                    logger.debug(
                        f"Skipping evaluation for supporting execution {card.card_id[:30]}... - "
                        f"response-supporting executions don't need evaluation"
                    )
                    return
                
                if not self._should_evaluate_execution(card):
                    logger.info(
                        f"Skipping regular evaluation for EXECUTION {card.card_id[:30]}... "
                        f"(does not meet EvaluationConfig conditions)"
                    )
                    return
                
                # Create EVALUATION card
                backend = card.metadata.get("backend") or card.metadata.get("preferred_backend") or "unknown"
                is_last = self._is_last_execution(card)
                eval_reason = self._get_eval_reason(card, backend, is_last)
                
                eval_type_suffix = " (after status determination)" if had_status_check else ""
                logger.info(
                    f"Creating EVALUATION card for EXECUTION {card.card_id[:30]}... ({eval_reason}){eval_type_suffix}"
                )
                
                eval_card = await self.kanban.add_card(
                    agent_name="EvalAgent",
                    card_type=CardType.EVALUATION,
                    title=f"Evaluate: {card.title}",
                    description=f"Evaluate execution result: {card.description or card.title}",
                    status=CardStatus.TODO,
                    metadata={
                        "execution_card_id": card.card_id,
                        "execution_title": card.title,
                        "eval_reason": eval_reason,  # Record why this was evaluated
                        "triggered_by": "execution_done_to_eval",
                        "planning_card_id": card.metadata.get("planning_card_id"),  # Track planning context
                        "step_order": card.metadata.get("step_order"),  # Track execution order
                        "after_status_determination": had_status_check,  # Mark if this is post-status-check evaluation
                    }
                )
                
                # Find the next TODO EXECUTION card from the same planning context
                # and make it depend on this EVALUATION card to ensure sequential evaluation
                await self._link_evaluation_to_next_execution(eval_card, card)
            
            self.add_rule(WorkflowRule(
                name="execution_done_to_eval",
                card_type=CardType.EXECUTION,
                card_status=CardStatus.DONE,
                agent_name=None,
                priority=80,
                pre_hook=create_evaluation_card,
            ))
            
            async def eval_result_handler(card: KanbanCard, result: Dict[str, Any]) -> None:
                """
                Handle evaluation results and update EXECUTION card status.
                
                For STATUS DETERMINATION evaluations:
                - Update EXECUTION card status to DONE (success) or BLOCKED (failure)
                - This provides immediate visual feedback of the task state
                
                For REGULAR evaluations:
                - Create new PLANNING card if needed (re-planning)
                
                Re-planning is triggered ONLY when:
                1. success=False: Task explicitly failed
                
                NOT triggered when:
                - status=error: Evaluation itself failed (not the task) - handled separately
                - success=True: Task succeeded
                """
                # Check if this is a status_determination evaluation
                is_status_determination = result.get("evaluation_type") == "status_determination"
                
                # First check if evaluation itself failed
                if result.get("status") == "error":
                    logger.info(
                        f"Evaluation process failed (not task failure) - "
                        f"error: {result.get('error', 'unknown')}"
                    )
                    
                    # For status_determination, update EXECUTION card to BLOCKED
                    if is_status_determination:
                        execution_card_id = result.get("execution_card_id")
                        if execution_card_id:
                            execution_card = self.kanban.get_card(execution_card_id)
                            if execution_card:
                                execution_card.metadata["status_checked"] = True
                                execution_card.metadata["evaluation_failed"] = True
                                execution_card.metadata["evaluation_error"] = result.get("error", "unknown")
                                await self.kanban.update_card_status(execution_card_id, CardStatus.BLOCKED)
                                logger.warning(
                                    f"Status Determination failed: Updated EXECUTION card {execution_card_id} to BLOCKED"
                                )
                    return
                
                # Handle STATUS DETERMINATION: Update EXECUTION card status
                if is_status_determination:
                    success = result.get("success", False)
                    confidence = result.get("confidence", "low")
                    execution_card_id = result.get("execution_card_id")
                    
                    if execution_card_id:
                        execution_card = self.kanban.get_card(execution_card_id)
                        if execution_card:
                            execution_card.metadata["status_checked"] = True
                            new_status = CardStatus.DONE if success else CardStatus.BLOCKED
                            await self.kanban.update_card_status(execution_card_id, new_status)
                            logger.info(
                                f"Status Determination: Updated EXECUTION card {execution_card_id} to {new_status.value} "
                                f"(success={success}, confidence={confidence})"
                            )
                    
                    # If status determination failed, trigger re-planning
                    if success is False:
                        suggestions = result.get("suggestions_for_host", [])
                        reasoning = result.get("reasoning", "")
                        evidence = result.get("evidence", [])
                        
                        logger.info(
                            f"Status Determination failed (success=False) - triggering re-planning"
                        )
                        
                        # Prepare eval_feedback data for HostAgent
                        eval_feedback_data = {
                            "success": success,
                            "confidence": result.get("confidence", "unknown"),
                            "evaluation": "[STATUS DETERMINATION] Task had mixed results",
                            "reasoning": reasoning,
                            "evidence": evidence,
                            "issues": [],
                            "suggestions": suggestions,
                            "execution_card_id": execution_card_id,
                            "execution_title": card.title,
                            "evaluation_type": "status_determination",
                        }
                        
                        await self.kanban.add_card(
                            agent_name="HostAgent",
                            card_type=CardType.PLANNING,
                            title="Adjust Plan (Status Determination Failure)",
                            description="Status determination indicates task needs re-execution. See eval_feedback for details.",
                            status=CardStatus.TODO,
                            metadata={
                                "triggered_by": card.card_id,
                                "eval_result": result,
                                "adjustment_reason": "status_determination_failure",
                                "eval_feedback": eval_feedback_data,  # Store for context injection
                            }
                        )
                    else:
                        logger.info(
                            f"Status Determination succeeded (success=True) - "
                            f"regular evaluation will follow if configured"
                        )
                    
                    return
                
                success = result.get("success", True)
                confidence = result.get("confidence", "high")
                suggestions = result.get("suggestions_for_host", [])
                
                # Re-planning decision: Only based on success field
                # confidence is used by EvalAgent internally for verification, not for re-planning
                needs_replanning = False
                feedback_parts = []
                
                # Only re-plan if task explicitly failed
                if success is False:
                    needs_replanning = True
                    feedback_parts.append("[FAILURE] Task execution failed or incomplete")
                    logger.info(
                        f"Task failed (success=False, confidence={confidence}) - triggering re-planning"
                    )
                
                # success=True: Task succeeded, no re-planning needed
                elif success is True:
                    logger.info(
                        f"Task completed successfully (success=True, confidence={confidence}) - "
                        f"no re-planning needed"
                    )
                
                # success=None: Evaluation inconclusive, do NOT re-plan (avoid infinite loop)
                elif success is None:
                    logger.warning(
                        f"Evaluation inconclusive (success=None, confidence={confidence}). "
                        f"Not triggering re-planning to avoid potential infinite loop. "
                        f"Task will be considered complete."
                    )
                
                # Note: success=True with medium/high confidence does NOT trigger re-planning
                # The task is considered complete even if it could be improved
                
                if needs_replanning and suggestions:
                    feedback_parts.append("\nSuggestions for improvement:")
                    for s in suggestions:
                        feedback_parts.append(f"- {s}")
                
                # Create new PLANNING card only for failures
                if needs_replanning and feedback_parts:
                    feedback = "\n".join(feedback_parts)
                    
                    reason = "failure"
                    
                    logger.info(
                        f"Creating new PLANNING task - "
                        f"success={success}, confidence={confidence}, "
                        f"reason={reason}"
                    )
                    
                    # Prepare eval_feedback data for HostAgent
                    eval_feedback_data = {
                        "success": success,
                        "confidence": confidence,
                        "evaluation": result.get("evaluation", ""),
                        "issues": result.get("issues", []),
                        "suggestions": suggestions,
                        "execution_card_id": card.card_id,
                        "execution_title": card.title,
                    }
                    
                    await self.kanban.add_card(
                        agent_name="HostAgent",
                        card_type=CardType.PLANNING,
                        title="Adjust Plan (Evaluation Feedback)",
                        description="Evaluation indicates task needs re-execution. See eval_feedback for details.",
                        status=CardStatus.TODO,
                        metadata={
                            "triggered_by": card.card_id,
                            "eval_result": result,
                            "adjustment_reason": reason,
                            "eval_feedback": eval_feedback_data,  # Store for context injection
                        }
                    )
            
            self.add_rule(WorkflowRule(
                name="evaluation_to_eval_agent",
                card_type=CardType.EVALUATION,
                card_status=CardStatus.TODO,
                agent_name="EvalAgent",
                priority=75,
                post_hook=eval_result_handler,
            ))
            
            logger.info("Registered default rules with evaluation enabled")
        else:
            logger.info("Registered default rules without evaluation")
    
    async def start(self) -> None:
        """Start workflow engine"""
        if self._running:
            logger.warning("WorkflowEngine already running")
            return
        
        await self._reset_inprogress_cards()
        
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("WorkflowEngine started")
    
    async def stop(self) -> None:
        """Stop workflow engine"""
        if not self._running:
            return
        
        self._running = False
        
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        
        if self._active_tasks:
            logger.info(f"Waiting for {len(self._active_tasks)} active tasks to complete...")
            tasks = [ctx.task for ctx in self._active_tasks.values() if ctx.task]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("WorkflowEngine stopped")
    
    async def _reset_inprogress_cards(self) -> None:
        """Reset IN_PROGRESS cards to TODO and clear active tasks"""
        for card in self.kanban._cards.values():
            if card.status == CardStatus.IN_PROGRESS:
                await self.kanban.update_card_status(card.card_id, CardStatus.TODO)
        self._active_tasks.clear()
    
    async def _poll_loop(self) -> None:
        """Main polling loop"""
        logger.info("Starting poll loop")
        
        while self._running:
            try:
                await self._check_timeouts()
                await self._cleanup_completed_tasks()
                await self._process_kanban_cards()
                await asyncio.sleep(self.poll_interval)
                
            except asyncio.CancelledError:
                logger.info("Poll loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in poll loop: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)
    
    async def _check_timeouts(self) -> None:
        """Check and handle task timeouts"""
        timeout_tasks = [(card_id, ctx) for card_id, ctx in self._active_tasks.items() if ctx.is_timeout()]
        
        for card_id, ctx in timeout_tasks:
            logger.warning(
                f"Task timeout for card {card_id} "
                f"(rule: {ctx.rule_name}, elapsed: {ctx.get_elapsed_time():.1f}s)"
            )
            
            if ctx.task and not ctx.task.done():
                ctx.task.cancel()
                try:
                    await ctx.task
                except asyncio.CancelledError:
                    pass
            
            await self.kanban.update_card_status(card_id, CardStatus.BLOCKED)
            
            card = self.kanban.get_card(card_id)
            if card:
                card.metadata["error"] = "timeout"
            
            self._stats["total_timeout"] += 1
            self._stats["total_processed"] += 1
            self._stats["total_failed"] += 1
        
        # Check RESPONSE cards that are waiting too long
        all_cards = list(self.kanban._cards.values())
        waiting_response_cards = [
            c for c in all_cards
            if c.card_type == CardType.RESPONSE
            and c.status == CardStatus.IN_PROGRESS
            and c.metadata.get("awaiting_execution")
        ]
        
        for response_card in waiting_response_cards:
            # Check if RESPONSE has been waiting for more than task_default_timeout
            created_timestamp = response_card.metadata.get("created_at_timestamp", 0)
            if created_timestamp > 0:
                elapsed = datetime.now().timestamp() - created_timestamp
                if elapsed > self.task_default_timeout:
                    # RESPONSE card timeout - complete it with partial results
                    execution_results = response_card.metadata.get("execution_results", [])
                    expected_count = response_card.metadata.get("expected_execution_count", 1)
                    current_count = len(execution_results)
                    
                    logger.error(
                        f"TIMEOUT: RESPONSE {response_card.card_id[:30]}... has been waiting for {elapsed:.1f}s. "
                        f"Received {current_count}/{expected_count} executions. "
                        f"Completing with partial results to avoid blocking."
                    )
                    
                    preliminary_answer = response_card.metadata.get("preliminary_answer", "")
                    response_card.result = {
                        "status": "timeout",
                        "answer": preliminary_answer,
                        "response_mode": "depends_on_execution",
                        "execution_enhanced": True if execution_results else False,
                        "execution_results": execution_results,
                        "error": f"Timeout after {elapsed:.1f}s, received {current_count}/{expected_count} executions"
                    }
                    response_card.metadata["result"] = response_card.result
                    response_card.metadata["awaiting_execution"] = False
                    response_card.metadata["error"] = "timeout"
                    
                    await self.kanban.update_card_status(response_card.card_id, CardStatus.BLOCKED)
                    self._stats["total_timeout"] += 1
                    self._stats["total_failed"] += 1
    
    async def _process_kanban_cards(self) -> None:
        """Process Kanban cards, find matching rules and trigger execution"""
        all_cards = list(self.kanban._cards.values())
        if not all_cards:
            return
        
        all_cards.sort(key=lambda c: c.created_at)
        
        for card in all_cards:
            # Clear processed record if card was reset to TODO
            if card.status == CardStatus.TODO:
                card_updated_at = self._to_datetime(getattr(card, "updated_at", None))
                related_pairs = [k for k in self._processed_pairs.keys() if k[0] == card.card_id]
                for p_key in related_pairs:
                    if card_updated_at > self._processed_pairs[p_key]:
                        del self._processed_pairs[p_key]

            if card.status == CardStatus.IN_PROGRESS:
                continue
            
            if len(self._active_tasks) >= self.max_concurrent_tasks:
                continue
            
            matched_rule = self._find_matching_rule(card)
            
            if matched_rule:
                pair_key = (card.card_id, matched_rule.name)
                if pair_key in self._processed_pairs:
                    logger.debug(
                        f"Skipping already processed pair: card={card.card_id[:20]}..., "
                        f"rule={matched_rule.name}"
                    )
                    continue
                
                if not await self._check_dependencies(card):
                    continue
                
                await self._trigger_task(card, matched_rule)
            else:
                # Log if no rule matches for DONE cards
                if card.status == CardStatus.DONE:
                    logger.debug(
                        f"No rule matched for DONE card: {card.card_id[:20]}... "
                        f"[{card.card_type.value}/{card.status.value}]"
                    )
    
    def _find_matching_rule(self, card: KanbanCard) -> Optional[WorkflowRule]:
        """Find matching rule for card (sorted by priority)"""
        for rule in self.rules:
            if rule.matches(card):
                return rule
        return None
    
    async def _check_dependencies(self, card: KanbanCard) -> bool:
        """
        Check if card dependencies are satisfied
        
        Reads dependency card IDs from metadata.depends_on and verifies
        all dependent cards have DONE status
        """
        depends_on = card.metadata.get("depends_on", [])
        if not depends_on:
            return True
        
        if not isinstance(depends_on, list):
            depends_on = [depends_on]
        
        for dep_card_id in depends_on:
            dep_card = self.kanban.get_card(dep_card_id)
            if not dep_card:
                logger.warning(f"Dependency card {dep_card_id} not found for {card.card_id}")
                return False
            
            if dep_card.status != CardStatus.DONE:
                return False
        
        return True
    
    async def _trigger_task(self, card: KanbanCard, rule: WorkflowRule) -> None:
        """
        Trigger task execution
        
        If rule.agent_name is None, only execute hooks without triggering agent.
        """
        if rule.agent_name is None:
            logger.info(
                f"Triggering rule '{rule.name}' for card {card.card_id} "
                f"[{card.card_type.value}/{card.status.value}] (hook-only rule)"
            )
            
            self._mark_pair_processed(card.card_id, rule.name)
            
            if rule.pre_hook:
                try:
                    await self._run_hook(rule.pre_hook, card)
                    logger.info(f"Pre-hook completed for hook-only rule '{rule.name}'")
                except Exception as e:
                    logger.error(f"Pre-hook failed for rule '{rule.name}': {e}", exc_info=True)
            
            if rule.post_hook:
                try:
                    await self._run_hook(rule.post_hook, card, {})
                    logger.debug(f"Post-hook completed for rule '{rule.name}'")
                except Exception as e:
                    logger.error(f"Post-hook failed for rule '{rule.name}': {e}", exc_info=True)
            return
        
        agent = self.coordinator.get_agent(rule.agent_name)
        if not agent:
            logger.warning(f"Agent '{rule.agent_name}' not found for rule '{rule.name}'")
            return
        
        logger.info(
            f"Triggering rule '{rule.name}' for card {card.card_id} "
            f"[{card.card_type.value}/{card.status.value}] → {rule.agent_name}"
        )
        
        pair_key = (card.card_id, rule.name)
        self._mark_pair_processed(card.card_id, rule.name)
        
        timeout = rule.timeout or self.task_default_timeout
        ctx = TaskExecutionContext(
            card_id=card.card_id,
            rule_name=rule.name,
            started_at=datetime.now(),
            timeout=timeout
        )
        
        try:
            await self.kanban.update_card_status(card.card_id, CardStatus.IN_PROGRESS)
        except Exception as e:
            logger.error(f"Failed to update card status: {e}")
            self._processed_pairs.pop(pair_key, None)
            return
        
        if rule.pre_hook:
            try:
                await self._run_hook(rule.pre_hook, card)
            except Exception as e:
                logger.error(f"Pre-hook failed for rule '{rule.name}': {e}")
        
        try:
            task = asyncio.create_task(self._execute_agent_task(card, agent, rule))
            ctx.task = task
            self._active_tasks[card.card_id] = ctx
        except Exception as e:
            logger.error(f"Failed to create task for card {card.card_id}: {e}")
            await self.kanban.update_card_status(card.card_id, CardStatus.TODO)
            self._processed_pairs.pop(pair_key, None)
    
    async def _execute_agent_task(
        self,
        card: KanbanCard,
        agent: BaseAgent,
        rule: WorkflowRule
    ) -> None:
        """Execute agent task with error handling"""
        card_id = card.card_id
        
        try:
            context = self._build_task_context(card, agent_name=rule.agent_name)
            result = await agent.process(context)
            
            card.result = result
            card.metadata["result"] = result
            
            # Check if task needs STATUS DETERMINATION evaluation (mixed success/failure results)
            if result.get("status") == "needs_eval":
                logger.info(
                    f"Task needs status determination for card {card_id} (mixed results) - "
                    f"creating FORCED evaluation (bypasses config)"
                )
                
                # Create FORCED EVALUATION card to determine final status (bypasses config)
                await self._create_status_evaluation_card(card, result)
                self._stats["total_processed"] += 1
                
            elif result.get("error") or result.get("status") == "error":
                await self.kanban.update_card_status(card_id, CardStatus.BLOCKED)
                error_msg = result.get('error', 'unknown error')
                logger.warning(f"Task failed for card {card_id}: {error_msg}")
                
                # Store error in card metadata for failure reporting
                card.metadata["error"] = str(error_msg)
                
                # If this is an EVALUATION card that failed, also update the corresponding EXECUTION card
                await self._handle_evaluation_failure(card, result)
                
                self._stats["total_failed"] += 1
                self._stats["total_processed"] += 1
            else:
                if card.card_type == CardType.PLANNING:
                    self._on_planning_completed(card, result)
                elif card.card_type == CardType.EXECUTION:
                    await self._on_execution_completed(card, result)
                elif card.card_type == CardType.RESPONSE:
                    self._on_response_completed(card, result)
                
                # For EVALUATION cards, check success field not just error
                if card.card_type == CardType.EVALUATION:
                    # Check if the evaluation itself succeeded (not the execution being evaluated)
                    is_status_determination = card.metadata.get("evaluation_type") == "status_determination"

                    eval_success = result.get("success", None)
                    
                    # If success is None, it means LLM didn't return this field
                    if eval_success is None:
                        logger.warning(
                            f"Evaluation result missing 'success' field for card {card_id}. "
                            f"This may indicate LLM parsing issue."
                        )
                    
                    # Log evaluation type for debugging
                    if is_status_determination:
                        logger.info(
                            f"Status Determination evaluation completed for card {card_id}: "
                            f"eval_success={eval_success}, confidence={result.get('confidence', 'unknown')}, "
                            f"EXECUTION card already updated to {'DONE' if eval_success else 'BLOCKED'}"
                        )
                    else:
                        logger.info(
                            f"Regular evaluation completed for card {card_id}: "
                            f"eval_success={eval_success}, confidence={result.get('confidence', 'unknown')}"
                        )
                
                await self.kanban.update_card_status(card_id, CardStatus.DONE)
                logger.info(f"Task completed for card {card_id}")
                self._stats["total_success"] += 1
                self._stats["total_processed"] += 1
            
            if rule.post_hook:
                try:
                    await self._run_hook(rule.post_hook, card, result)
                except Exception as e:
                    logger.error(f"Post-hook failed for rule '{rule.name}': {e}")
        
        except asyncio.CancelledError:
            logger.warning(f"Task cancelled for card {card_id}")
            await self.kanban.update_card_status(card_id, CardStatus.BLOCKED)
            
            # If this is an EVALUATION card that was cancelled, update EXECUTION card
            await self._handle_evaluation_failure(card, {"error": "cancelled"})
            raise
        
        except Exception as e:
            logger.error(f"Unexpected error executing task for card {card_id}: {e}", exc_info=True)
            await self.kanban.update_card_status(card_id, CardStatus.BLOCKED)
            card.metadata["error"] = str(e)
            
            # If this is an EVALUATION card that raised an exception, update EXECUTION card
            await self._handle_evaluation_failure(card, {"error": str(e)})
            
            self._stats["total_failed"] += 1
    
    async def _handle_evaluation_failure(self, eval_card: KanbanCard, result: Dict[str, Any]) -> None:
        """
        Handle EVALUATION card failure by updating the corresponding EXECUTION card.
        """
        if eval_card.card_type != CardType.EVALUATION:
            return
        
        execution_card_id = eval_card.metadata.get("execution_card_id")
        if not execution_card_id:
            logger.debug(f"EVALUATION card {eval_card.card_id} has no execution_card_id to update")
            return
        
        execution_card = self.kanban.get_card(execution_card_id)
        if not execution_card:
            logger.warning(f"EXECUTION card {execution_card_id} not found for failed EVALUATION card {eval_card.card_id}")
            return
        
        # Only update if EXECUTION card is still IN_PROGRESS
        # If EvalAgent already updated it (e.g., in status_determination), we don't override
        if execution_card.status == CardStatus.IN_PROGRESS:
            try:
                execution_card.metadata["evaluation_failed"] = True
                execution_card.metadata["evaluation_error"] = result.get("error", "unknown")
                await self.kanban.update_card_status(execution_card_id, CardStatus.BLOCKED)
                logger.warning(
                    f"Updated EXECUTION card {execution_card_id} to BLOCKED "
                    f"because its EVALUATION card {eval_card.card_id} failed"
                )
            except Exception as update_error:
                logger.error(
                    f"Failed to update EXECUTION card {execution_card_id} after EVALUATION failure: {update_error}"
                )
    
    def _get_workspace_directory(self) -> Optional[str]:
        """
        Get the workspace directory for file operations.
        
        Priority:
        1. Recording manager's trajectory directory (if available)
        2. Default workspace directory in recording log dir
        """
        from pathlib import Path
        
        # Try to get from recording manager
        if self.coordinator.recording_manager:
            recorder = getattr(self.coordinator.recording_manager, '_recorder', None)
            if recorder and hasattr(recorder, 'trajectory_dir'):
                # Use workspace subdirectory within trajectory
                workspace_dir = Path(recorder.trajectory_dir) / "workspace"
                workspace_dir.mkdir(parents=True, exist_ok=True)
                return str(workspace_dir.absolute())
        
        # Fallback: use default workspace in logs
        try:
            default_workspace = Path("./logs/workspace")
            default_workspace.mkdir(parents=True, exist_ok=True)
            return str(default_workspace.absolute())
        except Exception as e:
            logger.warning(f"Failed to create workspace directory: {e}")
            return None
    
    def _build_task_context(self, card: KanbanCard, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Build execution context for task"""
        context = {
            "card_id": card.card_id,
            "card_type": card.card_type.value,
            "card_status": card.status.value,
            "title": card.title,
            "description": card.description,
            "metadata": card.metadata,
            "step": card.step,
        }
        
        # Inject workspace directory for file operations
        # Use recording directory if available, otherwise use default workspace
        workspace_dir = self._get_workspace_directory()
        if workspace_dir:
            context["workspace_dir"] = workspace_dir
        
        # Automatically inject dependency results if card has depends_on
        depends_on = card.metadata.get("depends_on", [])
        if depends_on:
            if not isinstance(depends_on, list):
                depends_on = [depends_on]
            
            dependency_results = []
            for dep_card_id in depends_on:
                dep_card = self.kanban.get_card(dep_card_id)
                if dep_card:
                    dep_result = getattr(dep_card, 'result', None) or dep_card.metadata.get("result", {})
                    dependency_results.append({
                        "card_id": dep_card_id,
                        "title": dep_card.title,
                        "description": dep_card.description,
                        "card_type": dep_card.card_type.value,
                        "status": dep_card.status.value,
                        "result": dep_result,
                        "metadata": dep_card.metadata
                    })
                else:
                    logger.warning(
                        f"Dependency card {dep_card_id} not found for card {card.card_id}. "
                        f"This may indicate the dependency was deleted or the ID is incorrect."
                    )
            
            if dependency_results:
                context["dependency_results"] = dependency_results
                logger.debug(
                    f"Injected {len(dependency_results)} dependency result(s) for card {card.card_id[:20]}..."
                )
        
        if card.card_type == CardType.PLANNING:
            context["user_request"] = card.description or card.title
            context["include_kanban"] = True
            
            # Inject eval_feedback if available
            if "eval_feedback" in card.metadata:
                context["eval_feedback"] = card.metadata["eval_feedback"]
                logger.debug(f"Injected eval_feedback for PLANNING card {card.card_id[:20]}...")
        
        elif card.card_type == CardType.EXECUTION:
            if agent_name == "EvalAgent":
                exec_result = getattr(card, 'result', None) or card.metadata.get("result")
                if not exec_result:
                    logger.warning(f"No execution result found for card {card.card_id}")
                    exec_result = {}
                
                context["execution_result"] = exec_result
                context["execution_card"] = card.to_dict()
                context["instruction"] = card.description or card.title
            else:
                context["instruction"] = card.description or card.title
                context["auto_execute"] = True
                
                planning_card_id = card.metadata.get("planning_card_id")
                if planning_card_id:
                    task_ctx = self._context_manager.get_task_context(planning_card_id)
                    if task_ctx:
                        # Get remaining EXECUTION cards (TODO status, excluding current card)
                        remaining_executions = []
                        all_cards = list(self.kanban._cards.values())
                        execution_cards = [
                            c for c in all_cards
                            if c.card_type == CardType.EXECUTION
                            and c.status == CardStatus.TODO
                            and c.metadata.get("planning_card_id") == planning_card_id
                            and c.card_id != card.card_id  # Exclude current card
                        ]
                        # Sort by step_order if available
                        execution_cards.sort(key=lambda c: c.metadata.get("step_order", 999))
                        remaining_executions = [
                            {
                                "card_id": c.card_id,
                                "title": c.title,
                                "description": c.description,
                                "step_order": c.metadata.get("step_order")
                            }
                            for c in execution_cards
                        ]
                        
                        accumulated_context = task_ctx.get_context_for_next_execution(
                            next_task_description=card.description or card.title,
                            remaining_executions=remaining_executions,
                            max_context_length=10000
                        )
                        context["accumulated_context"] = accumulated_context
                        logger.debug(
                            f"Injected accumulated context for EXECUTION {card.card_id[:20]}...: "
                            f"{len(accumulated_context.get('previous_results', []))} previous results, "
                            f"{len(remaining_executions)} remaining steps"
                        )
                    
                    # Inject sibling execution results (for cross-execution awareness)
                    sibling_executions = []
                    all_cards = list(self.kanban._cards.values())
                    for c in all_cards:
                        if (c.card_type == CardType.EXECUTION
                            and c.metadata.get("planning_card_id") == planning_card_id
                            and c.card_id != card.card_id  # Exclude current card
                            and c.status in [CardStatus.DONE, CardStatus.BLOCKED]):
                            
                            sibling_result = getattr(c, 'result', None) or c.metadata.get("result", {})
                            sibling_executions.append({
                                "title": c.title,
                                "status": c.status.value,
                                "step_order": c.metadata.get("step_order"),
                                "result": sibling_result
                            })
                    
                    if sibling_executions:
                        # Sort by step_order
                        sibling_executions.sort(key=lambda x: x.get("step_order") or 999)
                        context["related_executions"] = sibling_executions
                        logger.debug(
                            f"Injected {len(sibling_executions)} sibling execution(s) for EXECUTION {card.card_id[:20]}..."
                        )
                
                if "backend" in card.metadata:
                    context["backend"] = card.metadata["backend"]
                if "tool" in card.metadata:
                    context["tool"] = card.metadata["tool"]
        
        elif card.card_type == CardType.EVALUATION:
            execution_card_id = card.metadata.get("execution_card_id")
            if execution_card_id:
                execution_card = self.kanban.get_card(execution_card_id)
                if execution_card:
                    exec_result = getattr(execution_card, 'result', None) or execution_card.metadata.get("result", {})
                    context["execution_result"] = exec_result
                    context["execution_card"] = execution_card.to_dict()

                    planning_card_id = execution_card.metadata.get("planning_card_id")
                    if planning_card_id:
                        task_ctx = self._context_manager.get_task_context(planning_card_id)
                        if task_ctx:
                            # Get remaining EXECUTION cards (TODO status)
                            remaining_executions = []
                            all_cards = list(self.kanban._cards.values())
                            execution_cards = [
                                c for c in all_cards
                                if c.card_type == CardType.EXECUTION
                                and c.status == CardStatus.TODO
                                and c.metadata.get("planning_card_id") == planning_card_id
                            ]
                            # Sort by step_order if available
                            execution_cards.sort(key=lambda c: c.metadata.get("step_order", 999))
                            remaining_executions = [
                                {
                                    "card_id": c.card_id,
                                    "title": c.title,
                                    "description": c.description,
                                    "step_order": c.metadata.get("step_order")
                                }
                                for c in execution_cards
                            ]
                            
                            # Use a dedicated method for evaluation context
                            accumulated_context = task_ctx.get_context_for_evaluation(
                                current_execution_card_id=execution_card_id,
                                current_execution_result=exec_result,
                                remaining_executions=remaining_executions,
                                max_context_length=15000  # Slightly larger for evaluation
                            )
                            context["accumulated_context"] = accumulated_context
                            
                            # Extract key fields for backward compatibility
                            context["original_task"] = accumulated_context.get("original_task")
                            context["is_last_execution"] = accumulated_context.get("is_last_execution", False)
                            
                            logger.debug(
                                f"Injected accumulated context for EVALUATION {card.card_id[:20]}...: "
                                f"{len(accumulated_context.get('previous_results', []))} previous results, "
                                f"is_last={accumulated_context.get('is_last_execution', False)}"
                            )
            
            context["run_verification"] = True
        
        return context
    
    async def _create_status_evaluation_card(self, execution_card: KanbanCard, result: Dict[str, Any]) -> None:
        """
        Create a FORCED EVALUATION card for mixed success/failure results.
        
        This is STATUS DETERMINATION evaluation - it determines the final status
        of an EXECUTION card when there are mixed success/failure tool calls.
        """
        # Get the real original task from TaskContext (if available)
        planning_card_id = execution_card.metadata.get("planning_card_id")
        original_task = execution_card.description or execution_card.title  # Fallback
        if planning_card_id:
            task_ctx = self._context_manager.get_task_context(planning_card_id)
            if task_ctx:
                original_task = task_ctx.original_task  # Real user's original request
        
        eval_card = await self.kanban.add_card(
            agent_name="EvalAgent",
            card_type=CardType.EVALUATION,
            title=f"[Status Check] {execution_card.title}",
            description=f"Determine final status for: {execution_card.description or execution_card.title}",
            status=CardStatus.TODO,
            metadata={
                "execution_card_id": execution_card.card_id,
                "execution_title": execution_card.title,
                "triggered_by": "needs_eval_status",
                "evaluation_type": "status_determination",  # FORCED evaluation, bypasses config
                "forced_evaluation": True,  # Flag to indicate this bypasses EvaluationConfig
                "mixed_results": True,
                "failed_count": result.get("failed_count", 0),
                "original_task": original_task,  # Real original task from PLANNING card
                "execution_task": execution_card.description or execution_card.title,  # Current execution's sub-task
                "planning_card_id": planning_card_id,  # Track planning context
                "step_order": execution_card.metadata.get("step_order"),  # Track execution order
            }
        )
        logger.info(
            f"Created FORCED status determination evaluation for {execution_card.card_id} "
            f"(bypasses EvaluationConfig)"
        )
        
        # Link to next execution to prevent error accumulation (same as regular evaluation)
        await self._link_evaluation_to_next_execution(eval_card, execution_card)
    
    def _on_planning_completed(self, card: KanbanCard, result: Dict[str, Any]) -> None:
        """
        Handle PLANNING card completion.
        
        Creates a TaskContext to track execution state across subsequent cards.
        """
        try:
            task_ctx = self._context_manager.create_task_context(
                task_id=card.card_id,
                original_task=card.description or card.title
            )
            logger.info(
                f"Created task context for PLANNING {card.card_id[:30]}...: "
                f"'{task_ctx.original_task[:50]}...'"
            )
            # Store task_context_id in card metadata for future reference
            card.metadata["task_context_id"] = card.card_id
        except Exception as e:
            logger.error(f"Failed to create task context: {e}")
    
    async def _on_execution_completed(self, card: KanbanCard, result: Dict[str, Any]) -> None:
        """
        Handle EXECUTION card completion.
        """
        try:
            planning_card_id = card.metadata.get("planning_card_id")
            if not planning_card_id:
                logger.debug(
                    f"EXECUTION {card.card_id[:20]}... has no planning_card_id, "
                    f"skipping context save"
                )
                return
            
            task_ctx = self._context_manager.get_task_context(planning_card_id)
            if not task_ctx:
                logger.warning(
                    f"Task context not found for planning_card_id {planning_card_id}, "
                    f"cannot save execution context"
                )
                return
            
            # Process and record execution (saves ProcessedContent to AgentStorage and adds card_id to TaskStorage)
            # Get agent_name from card metadata or default to GroundingAgent
            agent_name = card.metadata.get("agent_name", "GroundingAgent")
            
            # Use storage_manager's async method to process and record execution
            record = await self._context_manager.storage_manager.process_and_record_execution_async(
                agent_name=agent_name,
                card_id=card.card_id,
                title=card.title,
                result=result,
                metadata=card.metadata,
                task_id=planning_card_id,
                target_level=ContentLevel.SUMMARY
            )
            
            logger.info(
                f"Saved execution result for {card.card_id[:30]}...: "
                f"type={record.processed.content_type.value}, status={record.processed.status}, "
                f"total_executions={len(task_ctx._execution_card_ids)}"
            )
            
            # Optional: Update remaining task if result includes completed_subtask
            if "completed_subtask" in result:
                task_ctx.update_remaining_task(result["completed_subtask"])
                logger.debug(f"Updated remaining task: {task_ctx.remaining_task[:50]}...")

            if card.status != CardStatus.DONE and result.get("status") not in ["needs_eval", "error"]:
                try:
                    asyncio.create_task(
                        self.kanban.update_card_status(card.card_id, CardStatus.DONE)
                    )
                    logger.info(
                        f"EXECUTION card {card.card_id[:30]}... marked as DONE (soft finish)"
                    )
                except Exception as _exc:
                    logger.error(f"Failed to mark card {card.card_id} DONE: {_exc}")
        except Exception as e:
            logger.error(f"Failed to save execution context: {e}")
    
    def _on_response_completed(self, card: KanbanCard, result: Dict[str, Any]) -> None:
        """
        Handle RESPONSE card completion.
        """
        try:
            planning_card_id = card.metadata.get("planning_card_id")
            if not planning_card_id:
                logger.debug(
                    f"RESPONSE {card.card_id[:20]}... has no planning_card_id, "
                    f"skipping context cleanup"
                )
                return
            
            self._context_manager.remove_task_context(planning_card_id)
            logger.info(
                f"Cleaned up task context for planning_card_id {planning_card_id[:30]}... "
                f"after RESPONSE completion"
            )
        except Exception as e:
            logger.error(f"Failed to cleanup task context: {e}")
    
    async def _run_hook(self, hook: Callable, *args, **kwargs) -> Any:
        """Run hook function (supports both sync and async)"""
        if inspect.iscoroutinefunction(hook):
            return await hook(*args, **kwargs)
        else:
            return hook(*args, **kwargs)
    
    async def _cleanup_completed_tasks(self) -> None:
        """Clean up completed tasks"""
        completed = [card_id for card_id, ctx in self._active_tasks.items() if ctx.task and ctx.task.done()]
        for card_id in completed:
            del self._active_tasks[card_id]
    
    def get_status(self) -> Dict[str, Any]:
        """Get workflow engine status"""
        return {
            "running": self._running,
            "active_tasks": len(self._active_tasks),
            "processed_pairs": len(self._processed_pairs),
            "rules_count": len(self.rules),
            "stats": self._stats.copy(),
            "rules": [
                {
                    "name": rule.name,
                    "card_type": rule.card_type.value,
                    "card_status": rule.card_status.value,
                    "agent": rule.agent_name,
                    "priority": rule.priority,
                    "timeout": rule.timeout,
                }
                for rule in self.rules
            ],
            "active_task_details": [
                {
                    "card_id": card_id,
                    "rule_name": ctx.rule_name,
                    "elapsed_time": ctx.get_elapsed_time(),
                    "timeout": ctx.timeout,
                }
                for card_id, ctx in self._active_tasks.items()
            ],
        }
    
    def reset_processed_pairs(self) -> None:
        """Reset processed (card_id, rule_name) pairs list"""
        self._processed_pairs.clear()
        logger.info("Reset processed pairs list")
    
    def has_processed_pair(self, card_id: str, rule_name: str) -> bool:
        """
        Check if a (card_id, rule_name) pair has been processed
        """
        return (card_id, rule_name) in self._processed_pairs
    
    def get_stats(self) -> Dict[str, int]:
        """Get workflow statistics"""
        return self._stats.copy()
    
    def reset_stats(self) -> None:
        """Reset workflow statistics"""
        self._stats = {
            "total_processed": 0,
            "total_success": 0,
            "total_failed": 0,
            "total_timeout": 0,
        }
        logger.info("Reset statistics")