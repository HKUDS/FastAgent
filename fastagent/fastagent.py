from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastagent.agents import (
    AgentCoordinator,
    HostAgent,
    GroundingAgent,
    EvalAgent,
)
from fastagent.llm import LLMClient
from fastagent.grounding.core.grounding_client import GroundingClient
from fastagent.kanban import CardType, CardStatus
from fastagent.workflow import WorkflowEngine
from fastagent.config import get_config, load_config, get_agent_config, get_workflow_config
from fastagent.recording import RecordingManager
from fastagent.utils.logging import Logger
from datetime import datetime
from pathlib import Path
import traceback

logger = Logger.get_logger(__name__)


@dataclass
class EvaluationConfig:
    """
    Evaluation modes:
    - "none": Do not evaluate any execution
    - "all": Evaluate all executions
    - "last_only": Only evaluate the last execution
    - "selective": Selective evaluation (based on backends and always_eval_last)
    """
    # Evaluation mode
    mode: str = "selective"  # none/all/last_only/selective
    
    # The following parameters are only effective when mode="selective":
    
    # Backend filter: only evaluate executions using these backends
    # None means all backends
    # Example: ["gui", "mcp"] - only evaluate executions using gui or mcp
    backends: Optional[List[str]] = None
    
    # Always evaluate the last execution (regardless of backend)
    always_eval_last: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        valid_modes = ["none", "all", "last_only", "selective"]
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got: {self.mode}")
    
    @classmethod
    def none(cls) -> "EvaluationConfig":
        """Do not evaluate any execution"""
        return cls(mode="none")
    
    @classmethod
    def all(cls) -> "EvaluationConfig":
        """Evaluate all executions"""
        return cls(mode="all")
    
    @classmethod
    def last_only(cls) -> "EvaluationConfig":
        """Only evaluate the last execution"""
        return cls(mode="last_only")
    
    @classmethod
    def selective(
        cls,
        backends: Optional[List[str]] = None,
        always_eval_last: bool = True
    ) -> "EvaluationConfig":
        """Selective evaluation (recommended)"""
        return cls(
            mode="selective",
            backends=backends,
            always_eval_last=always_eval_last
        )
    
    def should_evaluate(self, backend: str, is_last: bool) -> bool:
        if self.mode == "none":
            return False
        
        if self.mode == "all":
            return True
        
        if self.mode == "last_only":
            return is_last
        
        if self.mode == "selective":
            # If it's the last one and always_eval_last=True, evaluate directly
            if is_last and self.always_eval_last:
                return True
            
            # If no backend restriction, evaluate all
            if self.backends is None:
                return True
            
            # Check if backend is in the list
            return backend in self.backends
        
        # Default to not evaluate
        return False


@dataclass
class FastAgentConfig:
    """
    FastAgent Configuration Class
    
    Contains all necessary configuration parameters for running the entire system.
    
    Configuration Loading Priority (low to high):
    1. DataClass defaults          ← Lowest priority (hardcoded fallback)
    2. config_workflow.json        ← Workflow default config file
    3. User config (--config)      ← User-defined config (JSON)
    4. CLI arguments               ← Command-line argument overrides
    5. Runtime modifications       ← Highest priority (direct code changes)
    
    Key Points:
    - Agent backend_scope configs are loaded from config_agents.json
    - Timeout control unified using max_execution_time to avoid parameter confusion
    - Logging optimized to print only on state changes or every N iterations
    - LLM API Keys set via environment variables, LLMClient uses litellm to auto-load from .env
    """
    
    # LLM Configuration
    llm_model: str = "anthropic/claude-sonnet-4-5"
    llm_enable_thinking: bool = False
    llm_rate_limit_delay: float = 0.0
    llm_max_retries: int = 3
    llm_timeout: float = 120.0  # LLM API call timeout in seconds
    llm_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Grounding Configuration
    grounding_config_path: Optional[str] = None  # If None, use default config
    workflow_config_path: Optional[str] = None  # If None, use default config_workflow.json
    
    # Workflow Configuration (can be overridden by CLI or config_workflow.json)
    enable_workflow: bool = True
    poll_interval: float = 1.0
    task_default_timeout: float = 1800.0
    
    # Agent Configuration
    host_system_prompt: Optional[str] = None
    grounding_system_prompt: Optional[str] = None
    eval_system_prompt: Optional[str] = None
    
    # Execution Configuration
    max_execution_time: float = 1000.0  # Maximum execution time (seconds), unified timeout control
    max_iterations: Optional[int] = None  # Maximum iterations (optional, auto-calculated from max_execution_time if None)
    auto_evaluate: bool = True  # Whether to enable auto-evaluation
    
    # Evaluation Configuration
    evaluation_config: Optional[EvaluationConfig] = None  # Evaluation config (includes strategy mode)
    
    # Recording Configuration
    enable_recording: bool = True  # Whether to enable recording
    recording_backends: Optional[List[str]] = None  # Backends to record (None = all)
    enable_screenshot: bool = True  # Whether to enable screenshots
    enable_video: bool = True  # Whether to enable video recording
    recording_log_dir: str = "./logs/recordings"  # Recording log directory
    
    # Logging Configuration
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file_path: Optional[str] = None

    # Runtime logging detail control
    log_every_n_iterations: int = 10  # Iteration log interval, can be overridden by CLI
    poll_sleep_multiplier: float = 2.0  # Poll sleep multiplier
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.llm_model:
            raise ValueError("llm_model is required")
        
        # Set default evaluation config if not specified
        if self.auto_evaluate and self.evaluation_config is None:
            self.evaluation_config = EvaluationConfig.selective(
                backends=["gui", "mcp"],  # Only evaluate executions using GUI or MCP
                always_eval_last=True,    # Always evaluate the last execution
            )
        
        logger.info(f"Using LLM model: {self.llm_model}")
        logger.debug("API keys should be set in environment variables (e.g., ANTHROPIC_API_KEY, OPENAI_API_KEY)")
    
    @classmethod
    def load(
        cls,
        workflow_config_path: Optional[str] = None,
        user_config_path: Optional[str] = None,
        cli_overrides: Optional[Dict[str, Any]] = None,
    ) -> "FastAgentConfig":

        from fastagent.config import load_workflow_config
        import json

        if workflow_config_path:
            workflow_cfg = load_workflow_config(workflow_config_path)
        else:
            workflow_cfg = get_workflow_config()
        
        wf = workflow_cfg.get("workflow", {})
        exec_cfg = workflow_cfg.get("execution", {})
        
        # Build base config dict from workflow config
        config_dict = {
            "enable_workflow": wf.get("enable", True),
            "poll_interval": wf.get("poll_interval", 1.0),
            "task_default_timeout": wf.get("task_default_timeout", 1800.0),
            "auto_evaluate": wf.get("auto_evaluate", True),
            "log_every_n_iterations": wf.get("log_every_n_iterations", 10),
            "poll_sleep_multiplier": wf.get("poll_sleep_multiplier", 2.0),
            "max_execution_time": exec_cfg.get("max_execution_time", 3600.0),
            "max_iterations": exec_cfg.get("max_iterations", None),
        }

        if user_config_path:
            try:
                with open(user_config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                
                # User config overrides workflow config
                config_dict.update(user_config)
                logger.info(f"Loaded from user config file: {user_config_path}")
            except FileNotFoundError:
                logger.error(f"Config file not found: {user_config_path}")
                raise
            except json.JSONDecodeError as e:
                logger.error(f"Config file format error: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")
                raise
        

        if cli_overrides:
            # Only override non-None values
            for key, value in cli_overrides.items():
                if value is not None:
                    config_dict[key] = value
            logger.debug(f"CLI overrides applied: {list(cli_overrides.keys())}")
        
        return cls(**config_dict)
    
    @classmethod
    def from_workflow_config(cls, config_path: Optional[str] = None, **overrides) -> "FastAgentConfig":
        """
        Create config from workflow config file (backward compatibility)
        Note: Recommended to use FastAgentConfig.load() instead of this method
        
        Args:
            config_path: Workflow config file path
            **overrides: Parameters to override
        """
        logger.warning("from_workflow_config() is deprecated, use FastAgentConfig.load() instead")
        return cls.load(workflow_config_path=config_path, cli_overrides=overrides)


class FastAgent:
    """
    FastAgent Unified Entry Class
    
    Usage Flow:
    1. Create FastAgentConfig
    2. Create FastAgent instance
    3. Call initialize() to initialize all components
    4. Call run() to execute tasks
    5. Call cleanup() to clean up resources
    """
    
    def __init__(self, config: Optional[FastAgentConfig] = None):
        """
        Initialize FastAgent
        """
        self.config = config or FastAgentConfig()
        
        # Core components (lazy initialization)
        self._llm_client: Optional[LLMClient] = None
        self._grounding_client: Optional[GroundingClient] = None
        self._recording_manager: Optional[RecordingManager] = None
        self._coordinator: Optional[AgentCoordinator] = None
        self._workflow_engine: Optional[WorkflowEngine] = None
        
        # Note: Agents auto-register to coordinator after creation
        
        # State
        self._initialized = False
        self._running = False
        
        logger.debug("FastAgent instance created")
    
    async def initialize(self) -> None:
        """
        Initialize all components
        
        Initialization order:
        1. LLMClient
        2. GroundingClient
        3. AgentCoordinator
        4. Agents (HostAgent, GroundingAgent, EvalAgent)
        5. WorkflowEngine (if enabled)
        """
        if self._initialized:
            logger.warning("FastAgent already initialized")
            return
        
        logger.info("Initializing FastAgent...")
        
        try:
            # 1. Initialize LLM Client
            self._llm_client = LLMClient(
                model=self.config.llm_model,
                enable_thinking=self.config.llm_enable_thinking,
                rate_limit_delay=self.config.llm_rate_limit_delay,
                max_retries=self.config.llm_max_retries,
                timeout=self.config.llm_timeout,
                **self.config.llm_kwargs
            )
            logger.info(f"LLM Client: {self.config.llm_model} (timeout: {self.config.llm_timeout}s)")
            
            # 2. Initialize Grounding Client
            if self.config.grounding_config_path:
                grounding_config = load_config(self.config.grounding_config_path)
            else:
                grounding_config = get_config()
            
            self._grounding_client = GroundingClient(config=grounding_config)
            await self._grounding_client.initialize_all_providers()
            backends = list(self._grounding_client.list_providers().keys())
            logger.info(f"Grounding Client: {len(backends)} backends")
            logger.debug(f"  Backends: {backends}")
            
            # 3. Initialize Agent Coordinator
            self._coordinator = AgentCoordinator(
                grounding_client=self._grounding_client,
                recording_manager=None,  # Will be set after RecordingManager is created
                llm_client=self._llm_client,
                enable_workflow=self.config.enable_workflow,
                poll_interval=self.config.poll_interval,
                task_default_timeout=self.config.task_default_timeout,
                auto_evaluate=self.config.auto_evaluate,
                evaluation_config=self.config.evaluation_config,
            )
            
            # 4. Initialize Recording Manager (if enabled) - after coordinator for kanban access
            if self.config.enable_recording:
                self._recording_manager = RecordingManager(
                    enabled=True,
                    task_id="",  # Will be set in run()
                    log_dir=self.config.recording_log_dir,
                    backends=self.config.recording_backends,
                    enable_screenshot=self.config.enable_screenshot,
                    enable_video=self.config.enable_video,
                    agent_name="GroundingAgent",
                    kanban=self._coordinator.kanban,  # Pass kanban for event recording
                )
                # Set recording manager in coordinator
                self._coordinator.recording_manager = self._recording_manager
                logger.info(f"Recording: {len(self._recording_manager.backends or [])} backends")
                logger.debug(f"  Screenshot: {self.config.enable_screenshot}, Video: {self.config.enable_video}")
            
            # 5. Initialize Agents
            # Load agent configurations from config_agents.json
            host_config = get_agent_config("HostAgent")
            grounding_config = get_agent_config("GroundingAgent")
            eval_config = get_agent_config("EvalAgent")
            
            # Host Agent
            host_backend_scope = host_config.get("backend_scope", []) if host_config else []
            host_agent = HostAgent(
                name="HostAgent",
                llm_client=self._llm_client,
                coordinator=self._coordinator,
                system_prompt=self.config.host_system_prompt,
            )
            
            # Grounding Agent
            grounding_backend_scope = grounding_config.get("backend_scope", ["shell", "gui", "mcp"]) if grounding_config else ["shell", "gui", "mcp"]
            grounding_agent = GroundingAgent(
                name="GroundingAgent",
                backend_scope=grounding_backend_scope,
                llm_client=self._llm_client,
                coordinator=self._coordinator,
                system_prompt=self.config.grounding_system_prompt,
            )
            
            # Set backend descriptions for HostAgent
            backend_descriptions = await grounding_agent.get_backend_descriptions()
            host_agent.set_backend_descriptions(backend_descriptions)
            
            # Eval Agent
            if self.config.auto_evaluate:
                eval_backend_scope = eval_config.get("backend_scope", ["shell", "system"]) if eval_config else ["shell", "system"]
                EvalAgent(
                    name="EvalAgent",
                    backend_scope=eval_backend_scope,
                    llm_client=self._llm_client,
                    coordinator=self._coordinator,
                    system_prompt=self.config.eval_system_prompt,
                )
            
            agents = self._coordinator.list_agents()
            logger.info(f"Agents: {', '.join(agents)}")
            
            # 6. Link Recording Manager to Kanban (for real-time updates)
            if self._recording_manager:
                self._recording_manager.kanban = self._coordinator.kanban
                
                # Start Recording Manager
                task_id = f"init_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self._recording_manager.task_id = task_id
                await self._recording_manager.start()

                logger.debug(f"Recording started: {task_id}")
            
            # 7. Start Workflow Engine
            if self.config.enable_workflow:
                await self._coordinator.start_workflow()
                self._workflow_engine = self._coordinator.workflow_engine
                
                workflow_status = self._coordinator.get_workflow_status()
                logger.info(f"Workflow: {workflow_status['rules_count']} rules, {self.config.poll_interval}s interval")
            
            self._initialized = True
            logger.info("="*60)
            logger.info("FastAgent initialization complete")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Failed to initialize FastAgent: {e}")
            await self.cleanup()
            raise
    
    async def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        wait_for_completion: bool = True,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute user query
        
        Args:
            query: User query/task description
            context: Additional context information
            wait_for_completion: Whether to wait for task completion
            timeout: Timeout in seconds (uses config.max_execution_time if None)
        """
        if not self._initialized:
            raise RuntimeError("FastAgent not initialized. Call initialize() first.")
        
        if self._running:
            raise RuntimeError("FastAgent is already running a task.")
        
        logger.info("="*60)
        logger.info(f"Task: {query}")
        logger.info("="*60)
        
        self._running = True
        start_time = asyncio.get_event_loop().time()
        timeout = timeout or self.config.max_execution_time
        
        try:
            # Create initial PLANNING card
            planning_card = await self._coordinator.kanban.add_card(
                agent_name="HostAgent",
                card_type=CardType.PLANNING,
                title=f"User request: {query[:50]}...",
                description=query,
                status=CardStatus.TODO,
                metadata={
                    "priority": "high",
                    "user_query": query,
                    "context": context or {},
                }
            )
            logger.debug(f"PLANNING card created: {planning_card.card_id}")
            
            if not wait_for_completion:
                return {
                    "status": "running",
                    "message": "Task started, running in background",
                    "planning_card_id": planning_card.card_id,
                }
            
            # Wait for task completion
            completed = False
            iterations = 0
            # Calculate max iterations: use user-specified value or auto-calculate from timeout
            if self.config.max_iterations is not None:
                max_iterations = self.config.max_iterations
                logger.debug(f"Max iterations: {max_iterations} (user specified)")
            else:
                # Auto-calculate based on timeout and poll interval
                # Each iteration waits 2x poll interval to ensure sufficient time for state changes
                max_iterations = int(timeout / (self.config.poll_interval * 2)) + 10
                logger.debug(f"Max iterations: {max_iterations} (auto from timeout={timeout}s)")
            
            # Log folding: only print on state changes or every N iterations
            last_active_cards_count = -1
            log_every_n_iterations = self.config.log_every_n_iterations
            
            while not completed and iterations < max_iterations:
                iterations += 1
                elapsed_time = asyncio.get_event_loop().time() - start_time
                
                # Check timeout
                if elapsed_time > timeout:
                    logger.warning(f"Timeout after {elapsed_time:.1f}s")
                    return {
                        "status": "timeout",
                        "message": f"Task execution timeout after {elapsed_time:.1f}s",
                        "user_response": "Task execution timeout. Check kanban_summary for current progress.",
                        "kanban_summary": self._coordinator.kanban.get_summary(),
                        "workflow_stats": self._get_workflow_stats(),
                        "execution_time": elapsed_time,
                    }
                
                # Wait for workflow processing
                await asyncio.sleep(self.config.poll_interval * self.config.poll_sleep_multiplier)
                
                # Check task status
                # Rule: All PLANNING, EXECUTION, RESPONSE cards are DONE or BLOCKED
                # And all EVALUATION cards (if enabled) are also DONE or BLOCKED
                all_cards = list(self._coordinator.kanban._cards.values())
                
                # Check PLANNING, EXECUTION, RESPONSE and EVALUATION cards
                # RESPONSE is included to ensure it completes before system exit
                # (even though it's auto-processed by WorkflowEngine)
                card_types_to_check = [CardType.PLANNING, CardType.EXECUTION, CardType.RESPONSE]
                if self.config.auto_evaluate:
                    card_types_to_check.append(CardType.EVALUATION)
                
                active_cards = [
                    card for card in all_cards
                    if card.card_type in card_types_to_check
                    and card.status in [CardStatus.TODO, CardStatus.IN_PROGRESS]
                ]
                
                # Filter out None markers
                active_cards = [c for c in active_cards if c is not None]
                
                if not active_cards:
                    # Check if we need to wait for auto-evaluation
                    
                    # IMPORTANT: This block only executes when active_cards is empty, which means:
                    # All PLANNING, EXECUTION, RESPONSE cards are DONE or BLOCKED
                    # All EVALUATION cards are DONE or BLOCKED (if auto_evaluate=True)
                    should_wait_for_eval = False
                    
                    if self.config.auto_evaluate:
                        # Check for DONE EXECUTION cards (exclude supporting cards for RESPONSE)
                        done_execution_cards = [
                            card for card in all_cards
                            if card.card_type == CardType.EXECUTION
                            and card.status == CardStatus.DONE
                            and not card.metadata.get("for_response_card")  # Skip supporting cards
                        ]
                        
                        if done_execution_cards:
                            # Check if there's a corresponding EVALUATION card for each DONE EXECUTION
                            for exec_card in done_execution_cards:
                                # Look for EVALUATION card that references this EXECUTION
                                # Note: We check for existence, not status. If EVALUATION exists
                                # (regardless of TODO/IN_PROGRESS/DONE/BLOCKED), it means either:
                                # 1. It's being processed (TODO/IN_PROGRESS) - in which case it would
                                #    be in active_cards and we wouldn't be in this block
                                # 2. It's completed (DONE/BLOCKED) - in which case we're truly done
                                has_eval_card = any(
                                    card.card_type == CardType.EVALUATION
                                    and card.metadata.get("execution_card_id") == exec_card.card_id
                                    for card in all_cards
                                )
                                
                                if not has_eval_card:
                                    # Found a DONE EXECUTION without EVALUATION card
                                    # This means the rule hasn't been triggered yet, need to wait
                                    should_wait_for_eval = True
                                    logger.debug(
                                        f"EXECUTION card {exec_card.card_id} is DONE but no EVALUATION card found, "
                                        f"waiting for engine to trigger evaluation..."
                                    )
                                    break
                    
                    if should_wait_for_eval:
                        # Wait for engine to process and create EVALUATION card
                        await asyncio.sleep(self.config.poll_interval * 4)
                        # Don't mark as completed, let next iteration check again
                        logger.debug("Waiting completed, will check again in next iteration")
                    else:
                        # Check for failed (BLOCKED) cards
                        blocked_cards = [
                            card for card in all_cards
                            if card.card_type in card_types_to_check
                            and card.status == CardStatus.BLOCKED
                        ]
                        
                        # No need to wait, all EXECUTION cards have their EVALUATIONs
                        completed = True
                        if blocked_cards:
                            logger.warning(
                                f"⚠ Completed with {len(blocked_cards)} failed task(s): "
                                f"{iterations} iterations, {elapsed_time:.1f}s"
                            )
                            for card in blocked_cards:
                                error_msg = card.metadata.get("error", "Unknown error")
                                logger.warning(f"  - {card.card_type.value} card '{card.title}': {error_msg}")
                        else:
                            logger.info(f"Completed: {iterations} iterations, {elapsed_time:.1f}s")
                else:
                    # Only log on state changes or every N iterations
                    current_count = len(active_cards)
                    should_log = (
                        current_count != last_active_cards_count or
                        iterations % log_every_n_iterations == 0 or
                        iterations == 1
                    )
                    
                    if should_log:
                        card_summary = [f'{c.card_type.value}/{c.status.value}' for c in active_cards]
                        logger.info(f"Iter {iterations}: {current_count} active - {card_summary}")
                        last_active_cards_count = current_count
            
            # If max_iterations reached but tasks still in progress, wait for them to complete
            if not completed and iterations >= max_iterations:
                logger.info("Max iterations reached, waiting for IN_PROGRESS tasks...")
                # Wait up to 30 seconds for in-progress tasks to complete
                extra_wait_iterations = 15
                for extra_iter in range(extra_wait_iterations):
                    await asyncio.sleep(2)
                    
                    all_cards = list(self._coordinator.kanban._cards.values())
                    # Check EVALUATION cards based on auto_evaluate setting
                    card_types_to_check = [CardType.PLANNING, CardType.EXECUTION, CardType.RESPONSE]
                    if self.config.auto_evaluate:
                        card_types_to_check.append(CardType.EVALUATION)
                    
                    in_progress_cards = [
                        card for card in all_cards
                        if card.card_type in card_types_to_check
                        and card.status == CardStatus.IN_PROGRESS
                    ]
                    
                    if not in_progress_cards:
                        logger.debug(f"All IN_PROGRESS tasks completed after {extra_iter + 1} extra iterations")
                        
                        # Give WorkflowEngine time to detect DONE status and trigger evaluation
                        if self.config.auto_evaluate:
                            await asyncio.sleep(self.config.poll_interval * 3)
                        
                        # Double-check if really completed
                        all_cards = list(self._coordinator.kanban._cards.values())
                        active_cards = [
                            card for card in all_cards
                            if card.card_type in [CardType.PLANNING, CardType.EXECUTION, CardType.RESPONSE]
                            and card.status in [CardStatus.TODO, CardStatus.IN_PROGRESS]
                        ]
                        
                        # If auto_evaluate enabled, also wait for EVALUATION tasks
                        if self.config.auto_evaluate:
                            eval_cards = [
                                card for card in all_cards
                                if card.card_type == CardType.EVALUATION
                                and card.status in [CardStatus.TODO, CardStatus.IN_PROGRESS]
                            ]
                            if eval_cards:
                                logger.debug(f"Waiting for {len(eval_cards)} EVALUATION tasks...")
                                active_cards.extend(eval_cards)
                        
                        if not active_cards:
                            completed = True
                        break
                    else:
                        logger.debug(f"Extra wait {extra_iter + 1}/{extra_wait_iterations}: {len(in_progress_cards)} IN_PROGRESS")
            
            # Collect results
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Get all completed cards
            done_cards = [
                card for card in all_cards
                if card.status == CardStatus.DONE
            ]
            
            # Get blocked (failed) cards
            blocked_cards = [
                card for card in all_cards
                if card.card_type in [CardType.PLANNING, CardType.EXECUTION, CardType.RESPONSE, CardType.EVALUATION]
                and card.status == CardStatus.BLOCKED
            ]
            
            # Get evaluation results
            eval_results = []
            if self.config.auto_evaluate:
                eval_cards = [
                    card for card in all_cards
                    if card.card_type == CardType.EVALUATION
                ]
                for card in eval_cards:
                    eval_results.append({
                        "card_id": card.card_id,
                        "title": card.title,
                        "metadata": card.metadata,
                    })
            
            # Collect response cards and generate user response
            response_cards = [
                card for card in all_cards
                if card.card_type == CardType.RESPONSE
            ]
            
            user_response = self._generate_user_response(
                response_cards=response_cards,
                done_cards=done_cards,
                completed=completed
            )
            
            # Determine final status based on failures
            has_failures = len(blocked_cards) > 0
            if not completed:
                final_status = "max_iterations_reached"
                final_message = f"Max iterations ({max_iterations}) reached"
            elif has_failures:
                final_status = "completed_with_failures"
                final_message = f"Completed with {len(blocked_cards)} task(s) failed"
            else:
                final_status = "completed"
                final_message = "Task execution completed successfully"
            
            result = {
                "status": final_status,
                "message": final_message,
                "execution_time": execution_time,
                "iterations": iterations,
                "completed_tasks": len(done_cards),
                "failed_tasks": len(blocked_cards),
                "evaluation_results": eval_results,
                "user_response": user_response, 
                "task_type": "response" if response_cards else "operation",  
                "kanban_summary": self._coordinator.kanban.get_summary(),
                "workflow_stats": self._get_workflow_stats(),
            }
            
            # Add failure details if any
            if has_failures:
                result["failures"] = [
                    {
                        "card_type": card.card_type.value,
                        "card_id": card.card_id,
                        "title": card.title,
                        "error": card.metadata.get("error", "Unknown error"),
                    }
                    for card in blocked_cards
                ]
            
            logger.info("="*60)
            if has_failures:
                logger.warning(
                    f"Summary: {result['status']} | {result['execution_time']:.2f}s | "
                    f"{result['iterations']} iters | {result['completed_tasks']} completed / {result['failed_tasks']} failed"
                )
                for failure in result['failures']:
                    logger.warning(f"  Failed: {failure['card_type']} - {failure['title']}: {failure['error']}")
            else:
                logger.info(
                    f"Summary: {result['status']} | {result['execution_time']:.2f}s | "
                    f"{result['iterations']} iters | {result['completed_tasks']} tasks"
                )
            logger.info("="*60)
            
            # IMPORTANT: Record all DONE cards' final state before stopping recording
            # This ensures RESPONSE and EVALUATION cards' results are captured
            if self._recording_manager and self._recording_manager._is_started:
                try:
                    # Get all relevant cards (including RESPONSE and EVALUATION)
                    final_cards = [
                        card for card in all_cards
                        if card.card_type in [CardType.PLANNING, CardType.EXECUTION, CardType.RESPONSE, CardType.EVALUATION]
                        and card.status in [CardStatus.DONE, CardStatus.BLOCKED]
                    ]
                    
                    # Record each card's final state
                    for card in final_cards:
                        await self._recording_manager.record_kanban_event(
                            event_type="card_final_state",
                            agent_name=card.agent_name,
                            card_id=card.card_id,
                            card_data=card.to_dict(),
                            changes={"status": "final"},
                        )
                    
                    # Give a small delay to ensure all writes are flushed
                    await asyncio.sleep(0.5)
                    
                    logger.debug(
                        f"Recorded final state for {len(final_cards)} cards "
                        f"(RESPONSE: {len([c for c in final_cards if c.card_type == CardType.RESPONSE])}, "
                        f"EVALUATION: {len([c for c in final_cards if c.card_type == CardType.EVALUATION])})"
                    )
                except Exception as e:
                    logger.warning(f"Failed to record final card states: {e}")
            
            return result
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            tb = traceback.format_exc(limit=10)
            logger.error(f"Task execution failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "traceback": tb,
                "user_response": f"Task execution error: {str(e)}",
                "execution_time": execution_time,
                "kanban_summary": self._coordinator.kanban.get_summary() if self._coordinator else None,
            }
        
        finally:
            self._running = False
    
    def _get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow statistics"""
        if self._workflow_engine:
            return self._workflow_engine.get_stats()
        return {}
    
    def _save_research_to_file(
        self,
        content: str,
        card_id: str,
        base_answer: str = ""
    ) -> str:
        """
        Save research content to a markdown file.
        
        Args:
            content: Research content to save
            card_id: Knowledge card ID
            base_answer: Base answer from HostAgent (if any)
            
        Returns:
            Path to the saved file
        """
        try:
            # Use recording directory if available, otherwise use default
            if self._recording_manager and hasattr(self._recording_manager, '_recorder'):
                recorder = self._recording_manager._recorder
                if recorder and hasattr(recorder, 'trajectory_dir'):
                    output_dir = Path(recorder.trajectory_dir) / "research_results"
                else:
                    output_dir = Path(self.config.recording_log_dir) / "research_results"
            else:
                output_dir = Path(self.config.recording_log_dir) / "research_results"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_{card_id}_{timestamp}.md"
            filepath = output_dir / filename
            
            # Create markdown content
            markdown_content = f"# Research Results\n\n"
            markdown_content += f"**Card ID:** {card_id}\n"
            markdown_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            if base_answer:
                markdown_content += f"## Base Answer\n\n{base_answer}\n"
            
            # Only add Research Details section if there's actual content
            if content:
                markdown_content += f"\n## Research Details\n\n{content}\n"
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Research results saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save research results: {e}")
            return ""
    
    def _generate_user_response(
        self,
        response_cards: List,
        done_cards: List,
        completed: bool
    ) -> str:
        """
        For tasks requiring user response, integrates execution results
        into a comprehensive answer.
        """
        if not completed:
            return "Task incomplete or timeout. Check kanban_summary for details."
        
        # For tasks requiring user response, return the answer (possibly enhanced with execution)
        if response_cards:
            response_parts = []
            
            for card in response_cards:
                if card.status == CardStatus.DONE:
                    # Get base answer from card
                    base_answer = None
                    if hasattr(card, 'result') and card.result:
                        base_answer = card.result.get('answer')
                    if not base_answer:
                        base_answer = card.metadata.get('answer')
                    
                    if base_answer:
                        response_parts.append(base_answer)
                    
                    # Check if execution was enhanced
                    execution_enhanced = False
                    execution_results = None
                    if hasattr(card, 'result') and card.result:
                        execution_enhanced = card.result.get('execution_enhanced', False)
                        execution_results = card.result.get('execution_results', [])
                    
                    # Always save response task results to file (regardless of execution enhancement)
                    saved_path = ""
                    
                    if execution_enhanced and execution_results:
                        # Extract execution results from RESPONSE card's result
                        response_parts.append("\n\nEXECUTION RESULTS:")
                        
                        # Process each execution result
                        all_content = []
                        for exec_result in execution_results:
                            result = exec_result.get('result', {})
                            content = result.get('response', '')
                            
                            if content:
                                # Convert to string if needed
                                content_str = content if isinstance(content, str) else str(content)
                                all_content.append(content_str)
                        
                        # Combine all execution content
                        combined_content = "\n---\n".join(all_content) if all_content else ""
                        
                        if combined_content:
                            # Save complete results to file (always)
                            saved_path = self._save_research_to_file(
                                content=combined_content,
                                card_id=card.card_id,
                                base_answer=base_answer or ""
                            )
                            
                            # Terminal display strategy: show summary for long content
                            if len(combined_content) > 1000:
                                # For long content, show summary in terminal
                                preview = combined_content[:500] + "\n... (content continues) ..."
                                response_parts.append(preview)
                                if saved_path:
                                    response_parts.append(f"\n[Complete results saved to: `{saved_path}`]")
                            else:
                                # For short content, show full result
                                response_parts.append(combined_content)
                                if saved_path:
                                    response_parts.append(f"\n[Saved to: `{saved_path}`]")
                        
                        # Add execution count details
                        response_parts.append(f"\n[Used {len(execution_results)} execution(s)]")
                    
                    elif base_answer:
                        # Direct answer (no execution) - also save to file
                        saved_path = self._save_research_to_file(
                            content="",  # No additional execution content
                            card_id=card.card_id,
                            base_answer=base_answer
                        )
                        if saved_path:
                            response_parts.append(f"\nSaved to: `{saved_path}`")
                    
                    if not base_answer:
                        logger.warning(f"RESPONSE card {card.card_id} completed but has no answer")
            
            if response_parts:
                return "\n".join(response_parts)
            else:
                return "Response task completed but no answer found."
        
        # For operation-type tasks, return a summary
        execution_cards = [
            card for card in done_cards 
            if card.card_type == CardType.EXECUTION
        ]
        
        if execution_cards:
            summary_parts = [f"Successfully completed {len(execution_cards)} operation task(s):"]
            for i, card in enumerate(execution_cards[:5], 1):  # Show first 5
                summary_parts.append(f"{i}. {card.title}")
            
            if len(execution_cards) > 5:
                summary_parts.append(f"... and {len(execution_cards) - 5} more task(s)")
            
            summary_parts.append("\nSee kanban_summary for detailed results.")
            return "\n".join(summary_parts)
        
        return "Task completed."
    
    async def cleanup(self) -> None:
        """
        Clean up all resources
        
        Cleanup order:
        1. Stop Workflow Engine
        2. Close all Grounding sessions
        3. Stop RecordingManager (if enabled)
        4. Clean up Coordinator
        """
        logger.info("Cleaning up FastAgent resources...")
        
        try:
            # 1. Stop Workflow Engine
            if self._coordinator and self._workflow_engine:
                await self._coordinator.stop_workflow()
                logger.debug("Workflow Engine stopped")
            
            # 2. Close all Grounding sessions
            if self._grounding_client:
                await self._grounding_client.close_all_sessions()
                logger.debug("Grounding sessions closed")

            # 3. Stop RecordingManager (if enabled)
            if self._recording_manager:
                try:
                    await self._recording_manager.stop()
                    logger.debug("Recording manager stopped")
                except Exception as e:
                    logger.warning(f"Failed to stop RecordingManager: {e}")
            
            # 4. Clean up Coordinator
            if self._coordinator:
                await self._coordinator.cleanup()
                logger.debug("Coordinator cleaned up")
            
            # Reset state
            self._initialized = False
            self._running = False
            
            logger.info("Cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
    
    @property
    def coordinator(self) -> Optional[AgentCoordinator]:
        """Get Coordinator"""
        return self._coordinator
    
    @property
    def kanban(self):
        """Get Kanban board"""
        if self._coordinator:
            return self._coordinator.kanban
        return None
    
    @property
    def workflow_engine(self) -> Optional[WorkflowEngine]:
        """Get Workflow Engine"""
        return self._workflow_engine
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status
        
        Returns:
            Status dictionary containing:
            - initialized: Whether initialized
            - running: Whether running a task
            - agents: List of registered agents
            - workflow_status: Workflow status
            - kanban_summary: Kanban summary
        """
        status = {
            "initialized": self._initialized,
            "running": self._running,
        }
        
        if self._coordinator:
            status["agents"] = self._coordinator.list_agents()
            status["workflow_status"] = self._coordinator.get_workflow_status()
            status["kanban_summary"] = self._coordinator.kanban.get_summary()
        
        return status
    
    def print_status(self) -> None:
        """Print current status (formatted output)"""
        status = self.get_status()
        
        print("\n" + "="*60)
        print("FastAgent Status")
        print("="*60)
        print(f"Initialized: {status['initialized']}")
        print(f"Running: {status['running']}")
        
        if "agents" in status:
            print(f"\nRegistered Agents: {', '.join(status['agents'])}")
        
        if "workflow_status" in status:
            ws = status["workflow_status"]
            print(f"\nWorkflow Status:")
            print(f"  Running: {ws.get('running', False)}")
            print(f"  Active tasks: {ws.get('active_tasks', 0)}")
            print(f"  Rules: {ws.get('rules_count', 0)}")
        
        if "kanban_summary" in status:
            print(f"\n{status['kanban_summary']}")
        
        print("="*60)
    
    async def __aenter__(self):
        """Support async with syntax"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Support async with syntax"""
        await self.cleanup()
        return False