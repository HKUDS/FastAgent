"""
Unified context management using StorageManager.

Usage:
    storage_manager = StorageManager(llm_client=llm)
    context_manager = ContextManager(storage_manager=storage_manager)
"""

from typing import Dict, Any, List, Optional

from fastagent.utils.logging import Logger
from fastagent.memory import StorageManager, TaskStorage
from fastagent.agents.content_processor import (
    ContentType,
    ContentLevel,
    ContentProcessor,
    ProcessedContent
)

logger = Logger.get_logger(__name__)


class ContextManager:
    """Global context manager using unified StorageManager."""
    
    def __init__(self, storage_manager: StorageManager):
        self.storage_manager = storage_manager
        logger.info("ContextManager initialized (using shared StorageManager)")
    
    @property
    def content_processor(self) -> ContentProcessor:
        """Get shared ContentProcessor instance."""
        return self.storage_manager.content_processor
    
    def set_llm_client(self, llm_client):
        """Set LLM client."""
        self.storage_manager.set_llm_client(llm_client)
        logger.info("ContextManager: LLM client set")
    
    def create_task_context(
        self,
        task_id: str,
        original_task: str,
        agent_name: str = "GroundingAgent"
    ) -> TaskStorage:
        """
        Create new task context.
        
        Args:
            task_id: Task ID (usually the PLANNING card ID)
            original_task: Original task description
            agent_name: Associated agent name for querying execution records
        """
        return self.storage_manager.create_task_storage(
            task_id=task_id,
            original_task=original_task,
            agent_name=agent_name
        )
    
    def get_task_context(self, task_id: str) -> Optional[TaskStorage]:
        return self.storage_manager.get_task_storage(task_id)
    
    def remove_task_context(self, task_id: str):
        self.storage_manager.remove_task_storage(task_id)
    
    def infer_content_type(
        self,
        card_title: str,
        card_metadata: Dict[str, Any]
    ) -> ContentType:
        """Infer content type using ContentProcessor."""
        return self.content_processor.infer_content_type(card_title, card_metadata)
    
    def process_execution_result(
        self,
        card: Any,
        result: Dict[str, Any],
        target_level: ContentLevel = ContentLevel.SUMMARY
    ) -> ProcessedContent:
        """
        Process execution result.
        
        Note: This method only processes, does not store.
        Storage is handled by StorageManager.process_and_record_execution.
        """
        return self.content_processor.process_execution_result(
            card_id=card.card_id,
            title=card.title,
            result=result,
            metadata=card.metadata,
            target_level=target_level
        )
    
    async def polish_response_for_task(
        self,
        task_id: str,
        execution_plan: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate polished response for specified task.
        """
        task_storage = self.get_task_context(task_id)
        if not task_storage:
            logger.warning(f"Task context not found: {task_id}")
            return None
        
        executions = task_storage.get_executions()
        
        polished = await self.content_processor.polish_response(
            executions=executions,
            original_task=task_storage.original_task,
            execution_plan=execution_plan
        )
        
        return polished