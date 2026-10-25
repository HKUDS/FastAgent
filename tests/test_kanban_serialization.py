import asyncio
import json
import unittest

from fastagent.kanban import CardType, Kanban


class KanbanSerializationTests(unittest.TestCase):
    def test_round_trip_preserves_card_result(self):
        async def create_board():
            board = Kanban()
            card = await board.add_card(
                agent_name="ResponseAgent",
                card_type=CardType.RESPONSE,
                title="Synthetic response",
                description="Synthetic task result",
            )
            card.result = {"status": "success", "answer": "synthetic answer"}
            return board

        board = asyncio.run(create_board())
        serialized = json.loads(board.to_json())

        restored = Kanban()
        restored.from_dict(serialized)

        restored_card = restored.get_card(next(iter(serialized["cards"])))
        self.assertEqual(
            restored_card.result,
            {"status": "success", "answer": "synthetic answer"},
        )
        self.assertEqual(
            json.loads(restored.to_json())["cards"][restored_card.card_id]["result"],
            {"status": "success", "answer": "synthetic answer"},
        )


if __name__ == "__main__":
    unittest.main()
