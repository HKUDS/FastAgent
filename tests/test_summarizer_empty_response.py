import asyncio
import unittest

from fastagent.memory import Memory, MemoryItem
from fastagent.memory.summarizer import MemorySummarizer


class EmptySummaryLLM:
    async def complete(self, messages):
        return {"content": ""}


class MemorySummarizerTests(unittest.TestCase):
    def test_empty_summary_does_not_discard_memory(self):
        async def compress():
            memory = Memory()
            for step in range(3):
                item = MemoryItem()
                item.set_value("step", step)
                item.set_value("content", f"synthetic item {step}")
                memory.add_memory_item(item, auto_timestamp=False)

            summarizer = MemorySummarizer(
                llm_client=EmptySummaryLLM(),
                max_memory_items=2,
                compression_threshold=1,
                summary_window=1,
            )
            result = await summarizer.compress_memory(memory)
            return result, memory, summarizer

        result, memory, summarizer = asyncio.run(compress())

        self.assertFalse(result["compressed"])
        self.assertEqual(memory.length, 3)
        self.assertEqual(summarizer.get_summaries(), [])


if __name__ == "__main__":
    unittest.main()
