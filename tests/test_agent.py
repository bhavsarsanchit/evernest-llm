import os
import unittest
from unittest.mock import MagicMock, patch

from evernest_assistant.agent import ask
from evernest_assistant.config import api_settings


class AskTests(unittest.TestCase):
    def test_refuses_when_the_agent_cannot_see_the_listing(self):
        os.environ.pop("API_KEY", None)
        with patch("evernest_assistant.agent.search") as search:
            result = ask("A-01", "L-1003", "Wie hoch ist die Kaltmiete?")
        search.assert_not_called()
        self.assertTrue(result["refused"])
        self.assertEqual(result["sources"], [])
        self.assertIn("access", result["answer"].lower())

    def test_host_comes_from_env_and_falls_back_to_the_spec(self):
        saved = {name: os.environ.get(name) for name in ("API_KEY", "API_BASE_URL")}
        try:
            os.environ["API_KEY"] = "test-key"
            os.environ["API_BASE_URL"] = "https://openrouter.ai/api/v1"
            self.assertEqual(
                api_settings("https://example.invalid/v1"),
                ("test-key", "https://openrouter.ai/api/v1"),
            )
            os.environ["API_BASE_URL"] = ""
            self.assertEqual(
                api_settings("https://openrouter.ai/api/v1"),
                ("test-key", "https://openrouter.ai/api/v1"),
            )
            os.environ["API_KEY"] = ""
            self.assertIsNone(api_settings("https://openrouter.ai/api/v1"))
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_trace_records_duration_request_tokens_and_cost(self):
        saved = {name: os.environ.get(name) for name in ("API_KEY", "API_BASE_URL")}
        os.environ["API_KEY"] = "test-key"
        os.environ["API_BASE_URL"] = "https://openrouter.ai/api/v1"

        class Usage:
            def model_dump(self):
                return {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                    "cost": 0.002,
                }

        class Message:
            content = "78.40"

        class Choice:
            message = Message()

        class Completion:
            usage = Usage()
            choices = [Choice()]

        client = MagicMock()
        client.chat.completions.create.return_value = Completion()
        trace = MagicMock()
        opik_client = MagicMock()
        opik_client.trace.return_value = trace
        try:
            with patch("evernest_assistant.agent.search", return_value=[]), patch(
                "openai.OpenAI", return_value=client
            ), patch("opik.Opik", return_value=opik_client):
                result = ask("A-01", "L-1001", "Wie groß ist die Wohnfläche?")
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

        self.assertFalse(result["refused"])
        sent = client.chat.completions.create.call_args.kwargs
        self.assertEqual(sent["extra_body"], {"usage": {"include": True}})
        logged = opik_client.trace.call_args.kwargs
        self.assertEqual(logged["input"]["agent_id"], "A-01")
        self.assertEqual(logged["input"]["listing_id"], "L-1001")
        self.assertEqual(logged["input"]["question"], "Wie groß ist die Wohnfläche?")
        self.assertIn("temperature", logged["input"])
        self.assertIn("max_tokens", logged["input"])
        self.assertLess(logged["start_time"], logged["end_time"])
        span = trace.span.call_args.kwargs
        self.assertEqual(span["type"], "llm")
        self.assertEqual(span["provider"], "openrouter.ai")
        self.assertEqual(span["usage"]["prompt_tokens"], 11)
        self.assertEqual(span["usage"]["completion_tokens"], 7)
        self.assertEqual(span["usage"]["total_tokens"], 18)
        self.assertEqual(span["total_cost"], 0.002)
        self.assertEqual(span["start_time"], logged["start_time"])
        self.assertEqual(span["end_time"], logged["end_time"])
