"""
Event-driven workflow engine for coordinating multiple agents.

Main Components:
- WorkflowEngine: Workflow engine that polls Kanban and triggers tasks
- WorkflowRule: Workflow rule that defines task triggering conditions and execution logic
- TaskExecutionContext: Context for tracking task execution state

Usage Example:
    from fastagent import AgentCoordinator
    from fastagent.workflow import WorkflowEngine, WorkflowRule
    from fastagent.kanban import CardType, CardStatus
    
    # Create coordinator with workflow enabled
    coordinator = AgentCoordinator(enable_workflow=True)
    
    # Get workflow engine
    engine = coordinator.workflow_engine
    
    # Add custom rule
    rule = WorkflowRule(
        name="custom_rule",
        card_type=CardType.EXECUTION,
        card_status=CardStatus.TODO,
        agent_name="GroundingAgent",
        priority=100,
    )
    engine.add_rule(rule)
    
    # Start workflow
    await coordinator.start_workflow()
"""

from .engine import WorkflowEngine, TaskExecutionContext
from .rules import WorkflowRule

__all__ = [
    "WorkflowEngine",
    "WorkflowRule",
    "TaskExecutionContext",
]