"""
Agent Coordinator Module

Provides centralized coordination for multi-agent systems.
Manages agent lifecycle and shared resources.

Key responsibilities:
- Agent registration and lifecycle management
- Shared resource management (GroundingClient, RecordingManager, Kanban)
- State synchronization across agents
- Workflow engine coordination
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Any, Tuple

from fastagent.kanban import Kanban
from fastagent.utils.logging import Logger

if TYPE_CHECKING:
    from fastagent.agents.base import BaseAgent
    from fastagent.grounding.core.grounding_client import GroundingClient
    from fastagent.recording import RecordingManager
    from fastagent.llm import LLMClient
    from fastagent.memory import StorageManager

logger = Logger.get_logger(__name__)


class AgentCoordinator:
    """
    Central coordinator for multi-agent systems.
    """
    
    def __init__(
        self,
        grounding_client: Optional[GroundingClient] = None,
        recording_manager: Optional[RecordingManager] = None,
        llm_client: Optional[LLMClient] = None,
        enable_workflow: bool = False,
        poll_interval: float = 1.0,
        task_default_timeout: float = 300.0,
        auto_evaluate: bool = True,
        evaluation_config: Optional[Any] = None,
    ):
        """
        Initialize the coordinator.
        
        Args:
            grounding_client: Shared grounding client for backend access
            recording_manager: Shared recording manager
            llm_client: Shared LLM client for agents
            enable_workflow: Enable event-driven workflow engine
            poll_interval: Workflow polling interval in seconds
            task_default_timeout: Default timeout for workflow tasks in seconds
            auto_evaluate: Enable automatic evaluation by EvalAgent
            evaluation_config: Evaluation configuration (includes evaluation strategy)
        """
        self._agents: Dict[str, BaseAgent] = {}
        
        # Shared resources
        self._grounding_client = grounding_client
        self._recording_manager = recording_manager
        self._llm_client = llm_client

        from fastagent.memory import StorageManager
        self.storage_manager = StorageManager(llm_client=llm_client)
        logger.info("AgentCoordinator: StorageManager initialized")
        
        # Connect recording_manager to grounding_client for GUI intermediate step recording
        if self._grounding_client and self._recording_manager:
            self._grounding_client.recording_manager = self._recording_manager
        
        # Shared state
        self._kanban = Kanban(coordinator=self)  # Pass coordinator to support dynamic agent memory access
        self._active_task_id: Optional[str] = None
        
        # Workflow engine
        self._workflow_engine: Optional[Any] = None
        if enable_workflow:
            from fastagent.workflow import WorkflowEngine
            self._workflow_engine = WorkflowEngine(
                coordinator=self,
                poll_interval=poll_interval,
                task_default_timeout=task_default_timeout,
                auto_evaluate=auto_evaluate,
                evaluation_config=evaluation_config,
            )
            logger.info("AgentCoordinator: WorkflowEngine enabled")
        
        logger.info("AgentCoordinator initialized")
    
    @property
    def grounding_client(self) -> Optional[GroundingClient]:
        """Get shared grounding client."""
        return self._grounding_client
    
    @grounding_client.setter
    def grounding_client(self, client: GroundingClient) -> None:
        """Set shared grounding client."""
        self._grounding_client = client
        if client and self._recording_manager:
            client.recording_manager = self._recording_manager
        logger.info("Coordinator: GroundingClient updated")
    
    @property
    def recording_manager(self) -> Optional[RecordingManager]:
        """Get shared recording manager."""
        return self._recording_manager
    
    @recording_manager.setter
    def recording_manager(self, manager: RecordingManager) -> None:
        """Set shared recording manager."""
        self._recording_manager = manager
        if self._grounding_client:
            self._grounding_client.recording_manager = manager
        logger.info("Coordinator: RecordingManager updated")
    
    @property
    def llm_client(self) -> Optional[LLMClient]:
        """Get shared LLM client."""
        return self._llm_client
    
    @llm_client.setter
    def llm_client(self, client: LLMClient) -> None:
        """Set shared LLM client."""
        self._llm_client = client
        logger.info("Coordinator: LLMClient updated")
    
    @property
    def workflow_engine(self) -> Optional[Any]:
        """Get workflow engine if enabled."""
        return self._workflow_engine
    
    @property
    def kanban(self) -> Kanban:
        """
        Get shared Kanban board for task coordination.
        
        The Kanban board is a system-wide shared resource used for:
        - Task management across all agents (TODO/IN_PROGRESS/DONE/BLOCKED)
        - High-level progress tracking
        - Multi-agent coordination
        
        This is separate from individual agent Memory.
        """
        return self._kanban
    
    async def execute_kanban_updates(
        self,
        updates: List[Dict[str, Any]],
        agent_name: str,
        step: int
    ) -> Dict[str, Any]:
        """
        Execute Kanban updates from any agent.
        
        EXECUTION cards automatically depend on the most recent pending EXECUTION
        to ensure sequential execution. Can be overridden with explicit 'depends_on'.
        
        Args:
            updates: List of update operations
            agent_name: Name of the agent requesting updates
            step: Current step number
        """
        from fastagent.kanban import CardType, CardStatus
        import uuid
        
        results = {
            "added": [],
            "completed": [],
            "updated": [],
            "errors": [],
            "logical_id_mapping": {}  # logical_id -> real_card_id
        }
        
        # Phase 1: Collect all cards to be created and build logical_id mapping
        logical_id_map = {}  # logical_id -> real_card_id
        add_operations = []  # Store add operations for processing
        
        for update in updates:
            action = update.get("action")
            
            if action == "add":
                logical_id = update.get("temp_id") or update.get("logical_id")
                if not logical_id:
                    # Require logical_id for all cards
                    logger.error(
                        f"Card '{update.get('title', 'unknown')}' is missing logical_id (temp_id). "
                        f"All cards must have a logical identifier for dependency tracking."
                    )
                    results["errors"].append(
                        f"Missing logical_id for card '{update.get('title', 'unknown')}'"
                    )
                    continue
                
                # Store for later processing
                add_operations.append(update)
                logical_id_map[logical_id] = None  # Will be filled with real card_id
            elif action in ["complete", "update"]:
                # Non-add operations processed later
                pass
        
        # Phase 2: Create all cards and build logical_id → real_id mapping
        # Track EXECUTION cards by planning_card_id and step_order for automatic chaining
        execution_by_planning: Dict[str, List[Tuple[int, str]]] = {}  # planning_card_id -> [(step_order, logical_id), ...]
        
        for update in add_operations:
            try:
                # Extract card information
                title = update.get("title", "")
                description = update.get("description", "")
                priority = update.get("priority", "medium")
                card_type_str = update.get("card_type", "planning")
                logical_id = update.get("temp_id") or update.get("logical_id")
                
                # Parse card_type
                try:
                    card_type = CardType(card_type_str)
                except ValueError:
                    card_type = CardType.PLANNING
                
                # Start with metadata from update (if any), then add priority
                metadata = update.get("metadata", {}).copy() if "metadata" in update else {}
                metadata["priority"] = priority
                
                # Store logical_id in metadata for tracking
                metadata["logical_id"] = logical_id
                
                # Handle depends_on (resolve logical_ids to real card_ids)
                depends_on = metadata.get("depends_on", update.get("depends_on"))
                
                # AUTOMATIC DEPENDENCY CHAIN for EXECUTION cards with step_order
                if card_type == CardType.EXECUTION:
                    step_order = metadata.get("step_order")
                    planning_card_id = metadata.get("planning_card_id")
                    
                    # If this EXECUTION has step_order and planning_card_id, and NO explicit depends_on
                    if step_order is not None and planning_card_id and not depends_on:
                        # Track this execution for potential dependency chain
                        if planning_card_id not in execution_by_planning:
                            execution_by_planning[planning_card_id] = []
                        execution_by_planning[planning_card_id].append((step_order, logical_id))
                        
                        # Find the previous EXECUTION card (with smaller step_order)
                        same_planning_execs = execution_by_planning[planning_card_id]
                        # Sort by step_order
                        same_planning_execs_sorted = sorted([e for e in same_planning_execs if e[0] < step_order], key=lambda x: x[0])
                        
                        if same_planning_execs_sorted:
                            # Depend on the immediately previous step
                            prev_step_order, prev_logical_id = same_planning_execs_sorted[-1]
                            depends_on = [prev_logical_id]
                            logger.info(
                                f"Auto-chaining EXECUTION '{title}' (step_order={step_order}) "
                                f"Depends on previous step (step_order={prev_step_order})"
                            )
                
                if depends_on:
                    # Replace logical_ids with real card_ids
                    resolved_depends_on = []
                    for dep_id in (depends_on if isinstance(depends_on, list) else [depends_on]):
                        if dep_id in logical_id_map:
                            # This is a logical_id, need to resolve it
                            real_id = logical_id_map.get(dep_id)
                            if real_id:
                                resolved_depends_on.append(real_id)
                            else:
                                # Dependency hasn't been created yet
                                logger.warning(
                                    f"Dependency logical_id '{dep_id}' not yet resolved for card '{title}'. "
                                    f"Ensure dependent cards are listed before cards that depend on them."
                                )
                                # Try to find it anyway (in case it was created in previous batch)
                                resolved_depends_on.append(dep_id)
                        else:
                            # This is already a real card_id or unknown logical_id
                            resolved_depends_on.append(dep_id)
                    
                    metadata["depends_on"] = resolved_depends_on
                    logger.debug(f"Resolved dependencies for '{title}': {resolved_depends_on}")
                
                planning_card_id = metadata.get("planning_card_id")
                if planning_card_id and planning_card_id in logical_id_map:
                    # Resolve temp_id to real card_id
                    resolved_planning_id = logical_id_map.get(planning_card_id)
                    if resolved_planning_id:
                        metadata["planning_card_id"] = resolved_planning_id
                        logger.debug(
                            f"Resolved planning_card_id for '{title}': "
                            f"{planning_card_id} → {resolved_planning_id[:20]}..."
                        )
                
                # Resolve target_response_logical_id to real card_id
                target_response_logical_id = metadata.get("target_response_logical_id")
                if target_response_logical_id and target_response_logical_id in logical_id_map:
                    resolved_response_id = logical_id_map.get(target_response_logical_id)
                    if resolved_response_id:
                        metadata["target_response_card_id"] = resolved_response_id
                        # Remove the logical_id field
                        metadata.pop("target_response_logical_id", None)
                        logger.debug(
                            f"Resolved target_response_card_id for '{title}': "
                            f"{target_response_logical_id} → {resolved_response_id[:20]}..."
                        )
                
                # Generate UUID-based card_id
                card_id = f"{agent_name}_{card_type.value}_{uuid.uuid4().hex[:12]}"
                
                # Create the card with the generated card_id
                card = await self._kanban.add_card(
                    agent_name=agent_name,
                    card_type=card_type,
                    title=title,
                    description=description,
                    status=CardStatus.TODO,
                    step=step,
                    metadata=metadata,
                    card_id=card_id  # Pass the generated card_id
                )
                
                # Store logical_id → card_id mapping
                if logical_id:
                    logical_id_map[logical_id] = card.card_id
                    results["logical_id_mapping"][logical_id] = card.card_id
                    logger.info(f"✓ Created card '{logical_id}' → {card.card_id[:20]}...")
                
                # IMMEDIATE TRANSITION: If RESPONSE card depends on execution, mark as IN_PROGRESS immediately
                if card_type == CardType.RESPONSE:
                    response_mode = metadata.get("response_mode", "direct")
                    if response_mode == "depends_on_execution":
                        # Initialize tracking metadata
                        preliminary_answer = metadata.get("preliminary_answer", "Processing...")
                        execution_plan = metadata.get("execution_plan", [])
                        expected_count = metadata.get("expected_execution_count", 1)
                        
                        card.metadata["awaiting_execution"] = True
                        card.metadata["execution_count"] = 0
                        card.metadata["execution_results"] = []
                        card.metadata["expected_execution_count"] = expected_count
                        card.metadata["response_card_id"] = card.card_id
                        card.metadata["created_at_timestamp"] = __import__('datetime').datetime.now().timestamp()
                        
                        # Immediately transition to IN_PROGRESS
                        await self._kanban.update_card_status(card.card_id, CardStatus.IN_PROGRESS)
                        logger.info(
                            f"🚀 RESPONSE card {card.card_id[:30]}... immediately marked as IN_PROGRESS, "
                            f"waiting for {expected_count} execution(s) (plan: {execution_plan})"
                        )
                
                results["added"].append({
                    "card_id": card.card_id,
                    "title": title,
                    "logical_id": logical_id
                })
                logger.debug(f"Coordinator: Added Kanban card '{title}' ({logical_id}) for {agent_name}")
                
            except Exception as e:
                error_msg = f"Failed to add card '{update.get('title', 'unknown')}': {e}"
                results["errors"].append(error_msg)
                logger.error(f"Coordinator: {error_msg}")
        
        # Phase 3: Process non-add operations (complete, update)
        for update in updates:
            action = update.get("action")
            
            try:
                if action == "complete":
                    # Mark task as done
                    card_id = update.get("card_id")
                    if card_id:
                        # Resolve logical_id if provided
                        if card_id in logical_id_map:
                            card_id = logical_id_map[card_id]
                        
                        await self._kanban.update_card_status(card_id, CardStatus.DONE)
                        results["completed"].append(card_id)
                        logger.debug(f"Coordinator: Completed Kanban card {card_id}")
                        
                elif action == "update":
                    # Update task status
                    card_id = update.get("card_id")
                    status_str = update.get("status")
                    if card_id and status_str:
                        # Resolve logical_id if provided
                        if card_id in logical_id_map:
                            card_id = logical_id_map[card_id]
                        
                        try:
                            status = CardStatus(status_str)
                            await self._kanban.update_card_status(card_id, status)
                            results["updated"].append({
                                "card_id": card_id,
                                "status": status_str
                            })
                            logger.debug(f"Coordinator: Updated Kanban card {card_id} to {status_str}")
                        except ValueError as e:
                            results["errors"].append(f"Invalid status '{status_str}' for card {card_id}: {e}")
                            
                elif action != "add":
                    results["errors"].append(f"Unknown action: {action}")
                    
            except Exception as e:
                error_msg = f"Failed to execute {action} for {agent_name}: {e}"
                results["errors"].append(error_msg)
                logger.error(f"Coordinator: {error_msg}")
        
        # Log summary if logical_ids were used
        if results["logical_id_mapping"]:
            logger.info(
                f"✓ Created {len(results['logical_id_mapping'])} cards with logical_id mapping: "
                f"{list(results['logical_id_mapping'].keys())}"
            )
        
        return results
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the coordinator."""
        if agent.name in self._agents:
            logger.warning(f"Agent '{agent.name}' already registered, replacing")
        
        self._agents[agent.name] = agent
        
        # Inject shared resources if not already set
        if not agent.llm_client and self._llm_client:
            agent.llm_client = self._llm_client
        
        logger.info(f"Registered agent: {agent.name}")
    
    def unregister_agent(self, agent_name: str) -> None:
        """Unregister an agent."""
        if agent_name in self._agents:
            self._agents.pop(agent_name)
            logger.info(f"Unregistered agent: {agent_name}")
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self._agents.get(agent_name)
    
    def list_agents(self) -> List[str]:
        """Get list of registered agent names."""
        return list(self._agents.keys())
    
    async def start_workflow(self) -> None:
        """
        Start the event-driven workflow engine.
        
        Usage:
        1. coordinator.start_workflow() - Start the workflow engine
        2. Add task cards to Kanban by agents
        3. WorkflowEngine automatically monitors and triggers the corresponding agents
        4. After the agent is completed, update the card status and trigger downstream tasks
        """
        if not self._workflow_engine:
            logger.warning("Coordinator: WorkflowEngine not enabled")
            return
        
        host_agent = self.get_agent("HostAgent")
        grounding_agent = self.get_agent("GroundingAgent")
        if host_agent and grounding_agent:
            try:
                logger.info("Coordinator: Updating backend descriptions for HostAgent...")
                backend_descriptions = await grounding_agent.get_backend_descriptions()
                host_agent.set_backend_descriptions(backend_descriptions)
                logger.info(f"Coordinator: Backend descriptions updated for {len(backend_descriptions)} backends")
                
                # Log summary of available backends and servers
                for backend_name, desc in backend_descriptions.items():
                    # Count servers (for MCP) or show simple info
                    if "servers):" in desc:
                        # Extract server count from description
                        import re
                        match = re.search(r'(\d+) servers', desc)
                        if match:
                            logger.info(f"  - {backend_name}: {match.group(1)} servers available")
                    else:
                        logger.info(f"  - {backend_name}: available")
            except Exception as e:
                logger.warning(f"Coordinator: Failed to update backend descriptions: {e}")
        
        # Register default rules
        self._workflow_engine.register_default_rules()
        
        # Start the workflow engine
        await self._workflow_engine.start()
        logger.info("Coordinator: Workflow engine started")
    
    async def stop_workflow(self) -> None:
        """Stop the workflow engine"""
        if self._workflow_engine:
            await self._workflow_engine.stop()
            logger.info("Coordinator: Workflow engine stopped")
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get the status of the workflow engine"""
        if not self._workflow_engine:
            return {"enabled": False}
        
        status = self._workflow_engine.get_status()
        status["enabled"] = True
        return status
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        # Stop workflow engine if running
        if self._workflow_engine:
            await self.stop_workflow()
        
        # Stop recording if active
        if self._recording_manager and self._active_task_id:
            await self._recording_manager.stop()
        
        logger.info("Coordinator cleanup complete")