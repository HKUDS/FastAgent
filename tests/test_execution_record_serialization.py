import json
import unittest

from fastagent.agents.content_processor import (
    ContentLevel,
    ContentType,
    ProcessedContent,
)
from fastagent.memory import Memory
from fastagent.memory.storage_manager import ExecutionRecord


class ExecutionRecordSerializationTests(unittest.TestCase):
    def test_memory_round_trip_preserves_processed_content(self):
        processed = ProcessedContent(
            content_type=ContentType.DATA_RETRIEVAL,
            level=ContentLevel.FULL,
            title="Synthetic retrieval",
            status="success",
            structured_data={"value": "synthetic"},
            full_data={"value": "synthetic"},
            summary="synthetic summary",
            metadata={"source": "synthetic"},
        )
        memory = Memory()
        memory.add_memory_item(
            ExecutionRecord(
                card_id="synthetic-card",
                agent_name="SyntheticAgent",
                step=1,
                processed=processed,
            ).to_memory_item()
        )

        restored_memory = Memory()
        restored_memory.from_list_of_dicts(json.loads(memory.to_json()))
        restored_record = ExecutionRecord.from_memory_item(
            restored_memory.get_latest_item()
        )

        self.assertIsInstance(restored_record.processed, ProcessedContent)
        self.assertEqual(
            restored_record.processed.to_dict(ContentLevel.FULL),
            processed.to_dict(ContentLevel.FULL),
        )
        self.assertEqual(restored_record.processed.metadata, {"source": "synthetic"})


if __name__ == "__main__":
    unittest.main()
