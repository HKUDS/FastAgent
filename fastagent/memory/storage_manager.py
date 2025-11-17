from __future__ import annotations

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from fastagent.memory.memory import Memory, MemoryItem
from fastagent.memory.summarizer import MemorySummarizer
from fastagent.agents.content_processor import (
    ContentProcessor,
    ContentLevel,
    ProcessedContent,
    ContentType
)
from fastagent.utils.logging import Logger

if TYPE_CHECKING:
    from fastagent.llm import LLMClient

logger = Logger.get_logger(__name__)


@dataclass
class ExecutionRecord:
    card_id: str
    agent_name: str
    step: int
    
    processed: ProcessedContent
    
    task_id: Optional[str] = None  
    timestamp: Optional[str] = None
    
    def to_memory_item(self) -> MemoryItem:
        item = MemoryItem()
        item.set_value("type", "execution")
        item.set_value("card_id", self.card_id)
        item.set_value("agent_name", self.agent_name)
        item.set_value("step", self.step)
        item.set_value("task_id", self.task_id)
        item.set_value("timestamp", self.timestamp)
        
        summary_data = self.processed.to_dict(ContentLevel.SUMMARY)
        item.add_values_from_dict(summary_data)
        
        item.set_value("_processed_full", self.processed)
        
        return item
    
    @classmethod
    def from_memory_item(cls, item: MemoryItem) -> Optional["ExecutionRecord"]:
        if item.get_value("type") != "execution":
            return None
        
        processed = item.get_value("_processed_full")
        if not processed:
            logger.warning("Missing _processed_full, attempting to reconstruct")
            return None
        
        return cls(
            card_id=item.get_value("card_id"),
            agent_name=item.get_value("agent_name"),
            step=item.get_value("step"),
            processed=processed,
            task_id=item.get_value("task_id"),
            timestamp=item.get_value("timestamp")
        )


class AgentStorage:
    def __init__(
        self,
        agent_name: str,
        llm_client: Optional[LLMClient] = None,
        max_memory_items: int = 50,
        compression_threshold: int = 30,
        auto_compress: bool = True
    ):
        self.agent_name = agent_name
        self.auto_compress = auto_compress
        
        # Core components
        self.memory = Memory()
        self.summarizer = MemorySummarizer(
            llm_client=llm_client,
            max_memory_items=max_memory_items,
            compression_threshold=compression_threshold
        )
        
        # Statistics
        self._stats = {
            "total_items": 0,
            "llm_interactions": 0,
            "executions": 0,
            "compressions": 0,
        }
        
        logger.info(f"AgentStorage initialized for {agent_name}")
    
    async def record_llm_interaction(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryItem:
        item = MemoryItem()
        item.set_value("type", "llm_interaction")
        item.set_value("role", role)
        item.set_value("content", content)
        
        if metadata:
            item.add_values_from_dict(metadata)
        
        self.memory.add_memory_item(
            item,
            agent_name=self.agent_name,
            step=self._stats["total_items"]
        )
        
        self._stats["total_items"] += 1
        self._stats["llm_interactions"] += 1
        
        # Auto-compression check
        if self.auto_compress:
            await self._check_and_compress()
        
        return item
    
    async def record_execution(
        self,
        record: ExecutionRecord
    ) -> MemoryItem:
        item = record.to_memory_item()
        
        self.memory.add_memory_item(
            item,
            agent_name=self.agent_name,
            step=self._stats["total_items"]
        )
        
        self._stats["total_items"] += 1
        self._stats["executions"] += 1
        
        logger.debug(
            f"{self.agent_name}: Recorded execution '{record.card_id[:20]}...' "
            f"(type: {record.processed.content_type.value})"
        )
        
        # Auto-compression check
        if self.auto_compress:
            await self._check_and_compress()
        
        return item
    
    async def _check_and_compress(self) -> None:
        if self.memory.length >= self.summarizer.compression_threshold:
            logger.info(
                f"{self.agent_name}: Memory size ({self.memory.length}) reached threshold, compressing..."
            )
            
            result = await self.summarizer.compress_memory(self.memory)
            
            if result.get("compressed"):
                self._stats["compressions"] += 1
                logger.info(
                    f"{self.agent_name}: Memory compressed - "
                    f"{result['original_size']} → {result['new_size']} items"
                )
    
    def build_llm_context(
        self,
        system_prompt: str,
        current_task: str,
        include_summaries: bool = True,
        max_recent_items: int = 10
    ) -> List[Dict[str, Any]]:
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        if include_summaries:
            summaries = self.summarizer.get_summaries()
            if summaries:
                summary_sections = []
                for i, summary_info in enumerate(summaries, 1):
                    step_range = summary_info.get("step_range", (0, 0))
                    summary_text = summary_info.get("summary", "")
                    summary_sections.append(
                        f"### Historical Summary {i} (Steps {step_range[0]}-{step_range[1]})\n"
                        f"{summary_text}"
                    )
                
                combined_summary = "\n\n".join(summary_sections)
                messages.append({
                    "role": "system",
                    "content": f"## Previous Work Context\n\n{combined_summary}"
                })
        
        recent_items = self.memory.content[-max_recent_items:] if self.memory.length > 0 else []
        
        if recent_items:
            for item in recent_items:
                item_type = item.get_value("type")
                
                if item_type == "llm_interaction":
                    messages.append({
                        "role": item.get_value("role"),
                        "content": item.get_value("content")
                    })
                elif item_type == "execution":
                    summary = item.get_value("summary", "")
                    status = item.get_value("status", "unknown")
                    if summary:
                        messages.append({
                            "role": "system",
                            "content": f"[Execution Result - {status}] {summary}"
                        })
        
        messages.append({
            "role": "user",
            "content": current_task
        })
        
        return messages
    
    def get_execution_by_card_id(self, card_id: str) -> Optional[ExecutionRecord]:
        for item in self.memory.content:
            if item.get_value("card_id") == card_id:
                return ExecutionRecord.from_memory_item(item)
        return None
    
    def get_executions_by_task_id(self, task_id: str) -> List[ExecutionRecord]:
        records = []
        for item in self.memory.content:
            if item.get_value("task_id") == task_id:
                record = ExecutionRecord.from_memory_item(item)
                if record:
                    records.append(record)
        return records
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "current_memory_size": self.memory.length,
            "summaries_count": len(self.summarizer.get_summaries())
        }
    
    def clear(self) -> None:
        self.memory.clear()
        self.summarizer.clear_summaries()
        self._stats = {
            "total_items": 0,
            "llm_interactions": 0,
            "executions": 0,
            "compressions": 0,
        }


class TaskStorage:
    def __init__(
        self,
        task_id: str,
        original_task: str,
        agent_storage: AgentStorage
    ):
        self.task_id = task_id
        self.original_task = original_task
        self.remaining_task = original_task
        self.response_card_id: Optional[str] = None
        
        self._execution_card_ids: List[str] = []
        
        self._agent_storage = agent_storage
        
        logger.debug(f"TaskStorage created for task {task_id}")
    
    def add_execution(self, card_id: str):
        self._execution_card_ids.append(card_id)
        logger.debug(f"TaskStorage: Added execution ref {card_id[:20]}... ({len(self._execution_card_ids)} total)")
    
    def get_executions(self) -> List[ProcessedContent]:
        results = []
        for card_id in self._execution_card_ids:
            record = self._agent_storage.get_execution_by_card_id(card_id)
            if record:
                results.append(record.processed)
        return results
    
    def get_context_for_next_execution(
        self,
        next_task_description: str,
        remaining_executions: Optional[List[Dict[str, Any]]] = None,
        max_context_length: int = 10000
    ) -> Dict[str, Any]:
        previous_results = []
        for processed in self.get_executions():
            previous_results.append(processed.to_dict(ContentLevel.SUMMARY))
        
        context = {
            "task_description": next_task_description,
            "original_task": self.original_task,
            "remaining_executions": remaining_executions or [],
            "previous_results": previous_results,
            "completed_steps": len(self._execution_card_ids),
        }
        
        # Check length and compress if needed
        context_str = str(context)
        if len(context_str) > max_context_length:
            logger.warning(f"Context too long ({len(context_str)} chars), compressing")
            context = self._compress_context(context, max_context_length)
        
        return context
    
    def _compress_context(
        self,
        context: Dict[str, Any],
        max_length: int
    ) -> Dict[str, Any]:
        
        from fastagent.agents.content_processor import ContentType
        
        previous_results = context.get("previous_results", [])
        
        if not previous_results:
            return context
        
        data_results = []
        operation_results = []
        other_results = []
        
        for result in previous_results:
            content_type = result.get("content_type", "")
            if content_type == ContentType.DATA_RETRIEVAL.value:
                data_results.append(result)
            elif content_type == ContentType.OPERATION.value:
                operation_results.append({
                    "title": result.get("title", ""),
                    "status": result.get("status", ""),
                    "summary": result.get("summary", "")[:100] 
                })
            else:
                other_results.append(result)
        
        compressed_results = data_results + other_results + operation_results
        
        context_copy = {**context}
        context_copy["previous_results"] = compressed_results
        
        if len(str(context_copy)) > max_length:
            keep_others = min(5, len(other_results) + len(operation_results))
            
            compressed_results = data_results + (other_results + operation_results)[-keep_others:]
            context_copy["previous_results"] = compressed_results
            context_copy["_compressed"] = True
            context_copy["_original_count"] = len(previous_results)
            context_copy["_kept_data_count"] = len(data_results)
            context_copy["_kept_other_count"] = keep_others
            
            logger.info(
                f"Context compressed: kept {len(data_results)} data results + "
                f"{keep_others} recent results (out of {len(previous_results)} total)"
            )
        
        return context_copy
    
    def get_context_for_evaluation(
        self,
        current_execution_card_id: str,
        current_execution_result: Optional[Dict[str, Any]] = None,
        remaining_executions: Optional[List[Dict[str, Any]]] = None,
        max_context_length: int = 15000
    ) -> Dict[str, Any]:
        previous_results = []
        for processed in self.get_executions():
            previous_results.append(processed.to_dict(ContentLevel.SUMMARY))
        
        is_last_execution = not remaining_executions or len(remaining_executions) == 0
        
        context = {
            "original_task": self.original_task,
            "remaining_executions": remaining_executions or [],
            "previous_results": previous_results,
            "completed_steps": len(self._execution_card_ids),
            "is_last_execution": is_last_execution,
            "current_execution_card_id": current_execution_card_id,
        }
        
        if current_execution_result:
            context["current_execution_result"] = current_execution_result
        
        return context
    
    def get_response_summary(self) -> Dict[str, Any]:
        executions = self.get_executions()
        
        summary = {
            "task_id": self.task_id,
            "original_task": self.original_task,
            "total_executions": len(executions),
            "status": "success" if all(p.status == "success" for p in executions) else "partial",
        }
        
        data_contents = []
        operation_summary = []
        verification_results = []
        
        for processed in executions:
            if processed.content_type == ContentType.DATA_RETRIEVAL:
                data_contents.append({
                    "source": processed.title,
                    "data": processed.full_data,
                    "summary": processed.summary,
                })
            elif processed.content_type == ContentType.OPERATION:
                operation_summary.append(processed.summary or processed.title)
            elif processed.content_type == ContentType.VERIFICATION:
                verification_results.append({
                    "item": processed.title,
                    "status": processed.status,
                })
        
        summary["data_contents"] = data_contents
        summary["operations"] = operation_summary
        summary["verifications"] = verification_results
        
        return summary


class StorageManager:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.content_processor = ContentProcessor(llm_client=llm_client)
        self._agent_storages: Dict[str, AgentStorage] = {}      
        self._task_storages: Dict[str, TaskStorage] = {}
        
        logger.info("StorageManager initialized with shared ContentProcessor")
    
    def set_llm_client(self, llm_client: LLMClient):
        self.content_processor.llm_client = llm_client
        logger.info("StorageManager: LLM client updated")
    
    def get_or_create_agent_storage(
        self,
        agent_name: str,
        llm_client: Optional[LLMClient] = None,
        **kwargs
    ) -> AgentStorage:
        if agent_name not in self._agent_storages:
            self._agent_storages[agent_name] = AgentStorage(
                agent_name=agent_name,
                llm_client=llm_client,
                **kwargs
            )
            logger.info(f"Created AgentStorage for {agent_name}")
        return self._agent_storages[agent_name]
    
    def get_agent_storage(self, agent_name: str) -> Optional[AgentStorage]:
        return self._agent_storages.get(agent_name)
    
    def create_task_storage(
        self,
        task_id: str,
        original_task: str,
        agent_name: str
    ) -> TaskStorage:
        agent_storage = self.get_or_create_agent_storage(agent_name)
        
        task_storage = TaskStorage(
            task_id=task_id,
            original_task=original_task,
            agent_storage=agent_storage
        )
        
        self._task_storages[task_id] = task_storage
        logger.info(f"Created TaskStorage for task {task_id}")
        
        return task_storage
    
    def get_task_storage(self, task_id: str) -> Optional[TaskStorage]:
        return self._task_storages.get(task_id)
    
    def remove_task_storage(self, task_id: str):
        if task_id in self._task_storages:
            del self._task_storages[task_id]
            logger.info(f"Removed TaskStorage for {task_id}")
    
    def process_and_record_execution(
        self,
        agent_name: str,
        card_id: str,
        title: str,
        result: Dict[str, Any],
        metadata: Dict[str, Any],
        task_id: Optional[str] = None,
        target_level: ContentLevel = ContentLevel.SUMMARY
    ) -> ExecutionRecord:
        processed = self.content_processor.process_execution_result(
            card_id=card_id,
            title=title,
            result=result,
            metadata=metadata,
            target_level=target_level
        )
        
        agent_storage = self.get_or_create_agent_storage(agent_name)
        record = ExecutionRecord(
            card_id=card_id,
            agent_name=agent_name,
            step=agent_storage._stats["total_items"],
            processed=processed,
            task_id=task_id
        )
        
        # 3. Record to AgentStorage
        import asyncio
        try:
            # Try to use asyncio.run in a new event loop
            asyncio.run(agent_storage.record_execution(record))
        except RuntimeError as e:
            # If already running in an event loop, log a warning
            logger.warning(
                f"Cannot use sync method in async context: {e}. "
                f"Consider using process_and_record_execution_async instead."
            )
            # Try to create a task in the current loop (without awaiting the result)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(agent_storage.record_execution(record))
                    logger.warning("Created background task for recording (result not awaited)")
            except Exception as e2:
                logger.error(f"Failed to record execution: {e2}")
        
        if task_id:
            task_storage = self.get_task_storage(task_id)
            if not task_storage:
                logger.warning(
                    f"TaskStorage for task_id={task_id} not found, auto-creating. "
                    f"Consider creating TaskStorage explicitly before recording executions."
                )
                task_storage = self.create_task_storage(
                    task_id=task_id,
                    original_task="", 
                    agent_name=agent_name
                )
            task_storage.add_execution(card_id)
        
        logger.info(
            f"Processed and recorded execution: {card_id[:20]}... "
            f"(agent={agent_name}, task={task_id}, type={processed.content_type.value})"
        )
        
        return record
    
    async def process_and_record_execution_async(
        self,
        agent_name: str,
        card_id: str,
        title: str,
        result: Dict[str, Any],
        metadata: Dict[str, Any],
        task_id: Optional[str] = None,
        target_level: ContentLevel = ContentLevel.SUMMARY
    ) -> ExecutionRecord:
        processed = self.content_processor.process_execution_result(
            card_id=card_id,
            title=title,
            result=result,
            metadata=metadata,
            target_level=target_level
        )
        
        agent_storage = self.get_or_create_agent_storage(agent_name)
        record = ExecutionRecord(
            card_id=card_id,
            agent_name=agent_name,
            step=agent_storage._stats["total_items"],
            processed=processed,
            task_id=task_id
        )
        
        await agent_storage.record_execution(record)
        
        if task_id:
            task_storage = self.get_task_storage(task_id)
            if not task_storage:
                logger.warning(
                    f"TaskStorage for task_id={task_id} not found, auto-creating. "
                    f"Consider creating TaskStorage explicitly before recording executions."
                )
                task_storage = self.create_task_storage(
                    task_id=task_id,
                    original_task="",  
                    agent_name=agent_name
                )
            task_storage.add_execution(card_id)
        
        return record
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_agents": len(self._agent_storages),
            "total_tasks": len(self._task_storages),
            "agent_stats": {
                name: storage.get_stats()
                for name, storage in self._agent_storages.items()
            }
        }