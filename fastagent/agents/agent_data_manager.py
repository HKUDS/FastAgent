from __future__ import annotations

from typing import Dict, Any, List, Optional, TYPE_CHECKING

from fastagent.memory import Memory, MemoryItem
from fastagent.agents.content_processor import ContentProcessor, ContentLevel
from fastagent.utils.logging import Logger

if TYPE_CHECKING:
    from fastagent.memory import StorageManager

logger = Logger.get_logger(__name__)


class AgentDataManager:
    def __init__(
        self,
        agent_name: str,
        storage_manager: "StorageManager",
        max_memory_items: int = 50,
        compression_threshold: int = 30,
        auto_compress: bool = True,
    ):
        self.agent_name = agent_name
        self.storage_manager = storage_manager
        
        self.agent_storage = storage_manager.get_or_create_agent_storage(
            agent_name=agent_name,
            llm_client=storage_manager.content_processor.llm_client,
            max_memory_items=max_memory_items,
            compression_threshold=compression_threshold,
            auto_compress=auto_compress
        )
        
        logger.info(f"AgentDataManager initialized for {agent_name}")
    
    @property
    def memory(self) -> Memory:
        return self.agent_storage.memory
    
    @property
    def content_processor(self) -> ContentProcessor:
        return self.storage_manager.content_processor
    
    async def record_llm_interaction(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryItem:
        return await self.agent_storage.record_llm_interaction(
            role=role,
            content=content,
            metadata=metadata
        )
    
    async def record_execution(
        self,
        card_id: str,
        title: str,
        result: Dict[str, Any],
        metadata: Dict[str, Any],
        task_id: Optional[str] = None,
        target_level: ContentLevel = ContentLevel.SUMMARY
    ) -> Dict[str, Any]:
        record = await self.storage_manager.process_and_record_execution_async(
            agent_name=self.agent_name,
            card_id=card_id,
            title=title,
            result=result,
            metadata=metadata,
            task_id=task_id,
            target_level=target_level
        )
        
        return {
            "processed_content": record.processed,
            "memory_item": record.to_memory_item().to_dict(),
            "filter_stats": None
        }
    
    def build_llm_context(
        self,
        system_prompt: str,
        current_task: str,
        include_summaries: bool = True,
        max_recent_items: int = 10
    ) -> List[Dict[str, Any]]:
        return self.agent_storage.build_llm_context(
            system_prompt=system_prompt,
            current_task=current_task,
            include_summaries=include_summaries,
            max_recent_items=max_recent_items
        )
    
    def get_recent_executions(self, count: int = 5) -> List[Dict[str, Any]]:
        executions = []
        for item in reversed(self.memory.content):
            if item.get_value("type") == "execution":
                executions.append(item.to_dict())
                if len(executions) >= count:
                    break
        
        return list(reversed(executions))
    
    def get_stats(self) -> Dict[str, Any]:
        return self.agent_storage.get_stats()
    
    def clear(self) -> None:
        self.agent_storage.clear()
        logger.info(f"{self.agent_name}: Data manager cleared")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "stats": self.agent_storage.get_stats(),
            "memory": self.memory.to_list_of_dicts(),
            "summarizer": self.agent_storage.summarizer.to_dict()
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        self.agent_name = data.get("agent_name", self.agent_name)
        
        memory_data = data.get("memory", [])
        self.memory.from_list_of_dicts(memory_data)
        
        summarizer_data = data.get("summarizer", {})
        self.agent_storage.summarizer.from_dict(summarizer_data)
        
        logger.info(
            f"{self.agent_name}: Data manager restored - "
            f"{len(memory_data)} memory items, "
            f"{len(summarizer_data.get('summaries', []))} summaries"
        )