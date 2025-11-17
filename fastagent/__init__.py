from importlib import import_module as _imp
from typing import Dict as _Dict, Any as _Any

__all__ = [
    # Main API
    "FastAgent",
    "FastAgentConfig",
    "EvaluationConfig",

    # Agents
    "HostAgent",
    "GroundingAgent",
    "EvalAgent",
    "AgentCoordinator",

    # Workflow
    "WorkflowEngine",
    "WorkflowRule",

    # Core Components
    "GroundingClient",
    "LLMClient",

    # Memory System
    "Kanban",
    "Memory",
    "MemoryItem",

    # Enums
    "KanbanEvent",
    "CardType",
    "CardStatus",

    # Recording System
    "RecordingManager",
    "RecordingViewer",

    # UI System
    "FastAgentUI",
    "create_ui",
    "AgentStatus",
    "UIIntegration",
]

# Map attribute → sub-module that provides it.
_attr_to_module: _Dict[str, str] = {
    # Main API
    "FastAgent": "fastagent.fastagent",
    "FastAgentConfig": "fastagent.fastagent",
    "EvaluationConfig": "fastagent.fastagent",

    # Agents
    "HostAgent": "fastagent.agents",
    "GroundingAgent": "fastagent.agents",
    "EvalAgent": "fastagent.agents",
    "AgentCoordinator": "fastagent.agents.coordinator",

    # Workflow
    "WorkflowEngine": "fastagent.workflow",
    "WorkflowRule": "fastagent.workflow",

    # Core Components
    "GroundingClient": "fastagent.grounding.core.grounding_client",
    "LLMClient": "fastagent.llm",   # May require optional deps (e.g. litellm)

    # Memory System
    "Kanban": "fastagent.memory.kanban",
    "Memory": "fastagent.memory.kanban",
    "MemoryItem": "fastagent.memory.kanban",

    # Enums
    "KanbanEvent": "fastagent.memory.enums",
    "CardType": "fastagent.memory.enums",
    "CardStatus": "fastagent.memory.enums",

    # Recording System
    "RecordingManager": "fastagent.recording",
    "RecordingViewer": "fastagent.recording.viewer",

    # UI System
    "FastAgentUI": "fastagent.utils.ui",
    "create_ui": "fastagent.utils.ui",
    "AgentStatus": "fastagent.utils.ui",
    "UIIntegration": "fastagent.utils.ui_integration",
}


def __getattr__(name: str) -> _Any:
    """Dynamically import sub-modules on first attribute access.

    This keeps the *initial* package import lightweight and avoids raising
    `ModuleNotFoundError` for optional / heavy dependencies until the
    corresponding functionality is explicitly used.
    """
    if name not in _attr_to_module:
        raise AttributeError(f"module 'fastagent' has no attribute '{name}'")

    module_name = _attr_to_module[name]
    module = _imp(module_name)
    value = getattr(module, name)
    globals()[name] = value 
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_attr_to_module.keys()))