import json
import unittest
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage

from myclaw.core.logger import JSONLEventLogger


class TestJSONLEventLogger(unittest.TestCase):

    def test_serialize_langchain_messages(self):
        payload = {
            "content": [
                HumanMessage(content="你好"),
                AIMessage(
                    content="我来帮你",
                    additional_kwargs={"meta": {"score": 1}},
                ),
            ]
        }

        serialized = JSONLEventLogger._serialize_for_json(payload)

        self.assertEqual(serialized["content"][0]["type"], "human")
        self.assertEqual(serialized["content"][1]["type"], "ai")
        self.assertEqual(serialized["content"][1]["content"], "我来帮你")
        json.dumps(serialized, ensure_ascii=False)

    def test_serialize_datetime_and_bytes(self):
        payload = {
            "when": datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc),
            "raw": "hello".encode("utf-8"),
        }

        serialized = JSONLEventLogger._serialize_for_json(payload)

        self.assertEqual(serialized["when"], "2026-04-21T12:00:00+00:00")
        self.assertEqual(serialized["raw"], "hello")
        json.dumps(serialized, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
