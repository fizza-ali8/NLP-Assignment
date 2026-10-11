"""Prompt assembly and context-overflow handling (Assignment 2, Phase II + IV)."""

import unittest

from tests import support  # noqa: F401
import conversation_manager as cm
from prompts import SYSTEM_PROMPT


def new_session():
    m = cm.ConversationManager()
    return m, m.create_session()


def total_chars(msgs):
    return sum(len(x["content"]) for x in msgs)


class PromptAssembly(unittest.TestCase):
    def test_layout_is_system_then_history_then_current_turn(self):
        m, sid = new_session()
        m.add_user_message(sid, "first question")
        m.add_assistant_message(sid, "first answer")
        m.add_user_message(sid, "second question")
        msgs, turns = m.build_prompt_messages(sid, "[Source: X]\nfact")
        self.assertEqual([x["role"] for x in msgs], ["system", "user", "assistant", "user"])
        self.assertEqual(msgs[0]["content"], SYSTEM_PROMPT)
        self.assertTrue(msgs[-1]["content"].startswith("second question"))
        self.assertIn("LIBRARY DOCUMENTS", msgs[-1]["content"])
        self.assertIn("fact", msgs[-1]["content"])
        self.assertEqual(turns, cm.MAX_TURNS)

    def test_system_prompt_is_identical_with_and_without_context(self):
        """Required for the local model server to reuse its cached prefix -> no extra latency from RAG."""
        m, sid = new_session()
        m.add_user_message(sid, "hi")
        a = m.build_prompt_messages(sid, "")[0][0]["content"]
        b = m.build_prompt_messages(sid, "lots of retrieved text " * 50)[0][0]["content"]
        self.assertEqual(a, b)

    def test_no_context_means_exactly_the_patrons_words(self):
        m, sid = new_session()
        m.add_user_message(sid, "hello")
        msgs, _ = m.build_prompt_messages(sid, "")
        self.assertEqual(msgs[-1]["content"], "hello")

    def test_stored_history_never_contains_retrieved_text(self):
        m, sid = new_session()
        m.add_user_message(sid, "overdue fine?")
        m.build_prompt_messages(sid, "SECRET EXCERPT")
        self.assertEqual(m.get_full_history(sid)[-1]["content"], "overdue fine?")

    def test_requires_a_pending_user_message(self):
        m, sid = new_session()
        with self.assertRaises(ValueError):
            m.build_prompt_messages(sid)

    def test_previous_user_message(self):
        m, sid = new_session()
        self.assertIsNone(m.previous_user_message(sid))
        m.add_user_message(sid, "one")
        self.assertIsNone(m.previous_user_message(sid))
        m.add_assistant_message(sid, "r")
        m.add_user_message(sid, "two")
        self.assertEqual(m.previous_user_message(sid), "one")


class ContextOverflow(unittest.TestCase):
    def fill(self, m, sid, pairs, size):
        for _ in range(pairs):
            m.add_user_message(sid, "U" * size)
            m.add_assistant_message(sid, "A" * size)

    def test_long_history_plus_max_context_stays_in_budget_oldest_dropped_first(self):
        m, sid = new_session()
        for i in range(12):
            m.add_user_message(sid, f"user-{i:02d} " + "u" * 1900)
            m.add_assistant_message(sid, f"asst-{i:02d} " + "a" * 1900)
        m.add_user_message(sid, "current question")
        msgs, turns = m.build_prompt_messages(sid, "C" * 1500)
        self.assertLessEqual(total_chars(msgs), cm.CONTEXT_CHAR_BUDGET)
        self.assertLess(turns, cm.MAX_TURNS, "history should have been trimmed")
        joined = " ".join(x["content"] for x in msgs)
        self.assertIn("asst-11", joined, "newest history is kept")
        self.assertNotIn("user-00", joined, "oldest history is dropped")
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertTrue(msgs[-1]["content"].startswith("current question"))

    def test_light_conversation_is_not_trimmed(self):
        m, sid = new_session()
        self.fill(m, sid, 6, 100)
        m.add_user_message(sid, "now")
        _, turns = m.build_prompt_messages(sid, "C" * 1500)
        self.assertEqual(turns, cm.MAX_TURNS)

    def test_current_turn_is_never_dropped_even_with_a_tiny_budget(self):
        old = cm.CONTEXT_CHAR_BUDGET
        cm.CONTEXT_CHAR_BUDGET = 100
        try:
            m, sid = new_session()
            self.fill(m, sid, 4, 500)
            m.add_user_message(sid, "please answer me")
            msgs, turns = m.build_prompt_messages(sid, "excerpt " * 100)
            self.assertEqual(turns, 0)
            self.assertEqual([x["role"] for x in msgs], ["system", "user"])
            self.assertTrue(msgs[-1]["content"].startswith("please answer me"))
        finally:
            cm.CONTEXT_CHAR_BUDGET = old

    def test_absurdly_large_context_is_truncated_last(self):
        m, sid = new_session()
        self.fill(m, sid, 3, 1900)
        m.add_user_message(sid, "q" * 1900)
        msgs, _ = m.build_prompt_messages(sid, "C" * 100000)
        self.assertLessEqual(total_chars(msgs), cm.CONTEXT_CHAR_BUDGET)
        self.assertTrue(msgs[-1]["content"].startswith("q" * 100))

    def test_per_message_cap(self):
        m, sid = new_session()
        m.add_user_message(sid, "x" * 50000)
        self.assertEqual(len(m.get_full_history(sid)[-1]["content"]), cm.MAX_CHARS_PER_MSG)

    def test_budget_leaves_room_inside_the_model_context_window(self):
        import llm_engine
        approx_tokens = cm.CONTEXT_CHAR_BUDGET / 3.5  # conservative chars-per-token for English prose
        self.assertLess(approx_tokens, llm_engine.NUM_CTX - 1000, "prompt budget must leave >= 1000 tokens for the reply")


if __name__ == "__main__":
    unittest.main()
