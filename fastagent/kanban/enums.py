from enum import Enum


class CardType(str, Enum):
    """Types of cards on the Kanban board"""
    PLANNING = "planning"
    EXECUTION = "execution"
    EVALUATION = "evaluation"
    RESPONSE = "response"


class CardStatus(str, Enum):
    """Status of cards on the Kanban board"""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"


class KanbanEvent(str, Enum):
    """Event types for Kanban hooks"""
    ON_CARD_ADDED = "on_card_added"
    ON_CARD_UPDATED = "on_card_updated"
    ON_CARD_DELETED = "on_card_deleted"
    ON_STEP_RECORDED = "on_step_recorded"