"""
    RecordingManager
      ├── internal management of platform.RecordingClient
      ├── internal management of platform.ScreenshotClient  
      ├── internal management of TrajectoryRecorder
      ├── internal management of ActionRecorder
      └── internal management of KanbanEventRecorder
"""

# Auto-record the tool execution
from .manager import RecordingManager

# Low-level components (advanced users)
from .recorder import TrajectoryRecorder
from .action_recorder import ActionRecorder
from .kanban_recorder import KanbanEventRecorder

# Utility functions
from .utils import (
    load_trajectory_from_jsonl,
    load_metadata,
    format_trajectory_for_export,
    analyze_trajectory,
    load_recording_session,
    filter_trajectory,
    extract_errors,
    generate_summary_report,
)

# New utility functions for agent actions and kanban events
from .action_recorder import (
    load_agent_actions,
    analyze_agent_actions,
    format_agent_actions,
)

from .kanban_recorder import (
    load_kanban_events,
    analyze_kanban_events,
    format_kanban_events,
    reconstruct_kanban_timeline,
)

__all__ = [
    # Manager
    'RecordingManager',
    
    # Recorders
    'TrajectoryRecorder',
    'ActionRecorder',
    'KanbanEventRecorder',
    
    # Trajectory utils
    'load_trajectory_from_jsonl',
    'load_metadata',
    'format_trajectory_for_export',
    'analyze_trajectory',
    'load_recording_session',
    'filter_trajectory',
    'extract_errors',
    'generate_summary_report',
    
    # Agent action utils
    'load_agent_actions',
    'analyze_agent_actions',
    'format_agent_actions',
    
    # Kanban event utils
    'load_kanban_events',
    'analyze_kanban_events',
    'format_kanban_events',
    'reconstruct_kanban_timeline',
]