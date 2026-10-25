"""Tools, the prompt-based tool gate, sentence splitting, and end-of-speech."""

import asyncio
import os
import tempfile
import threading
import time
import unittest
from datetime import date

from tools import calendar_tool, catalogue, crm, fine
from tools.orchestrator import StreamGate, ToolCall, detect, dispatch, execute, parse_tool_call
from voice.endpoint import SpeechBuffer
from voice.sentences import SentenceSplitter


class FineTool(unittest.TestCase):
    def test_four_days_is_one_dollar(self):
        text = fine.calculate({"days": 4, "items": 1})
        self.assertIn("$1.00", text)

    def test_cap_and_block_and_lost(self):
        text = fine.calculate({"days": 50, "items": 2})
        self.assertIn("$20.00", text)  # $10 cap x 2 items
        self.assertIn("blocked", text)
        self.assertIn("lost", text)

    def test_bad_arguments_do_not_raise(self):
        self.assertIn("whole number", fine.calculate({"days": "nope"}))
        self.assertIn("number of days", fine.calculate({}))


class CalendarTool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = os.environ.get("LIBRARY_DATA_DIR")
        os.environ["LIBRARY_DATA_DIR"] = self.tmp.name
        self.today = date(2026, 10, 26)  # a Monday

    def tearDown(self):
        if self._old is None:
            os.environ.pop("LIBRARY_DATA_DIR", None)
        else:
            os.environ["LIBRARY_DATA_DIR"] = self._old
        self.tmp.cleanup()

    def test_book_list_conflict_and_cancel(self):
        booked = calendar_tool.run(
            {"action": "book", "date": "2026-10-26", "start": "14:00", "hours": 2, "room": 1, "name": "Ada", "user_id": "u1"},
            today=self.today,
        )
        self.assertIn("Booked study room 1", booked)
        again = calendar_tool.run(
            {"action": "book", "date": "2026-10-26", "start": "15:00", "hours": 1, "room": 1, "user_id": "u2"},
            today=self.today,
        )
        self.assertIn("already booked", again)
        listed = calendar_tool.run({"action": "list", "date": "2026-10-26"}, today=self.today)
        self.assertIn("Room 1", listed)
        cancelled = calendar_tool.run({"action": "cancel", "user_id": "u1"}, today=self.today)
        self.assertIn("Cancelled", cancelled)

    def test_sunday_and_too_far_ahead_are_refused(self):
        sunday = calendar_tool.run(
            {"action": "book", "date": "2026-10-25", "start": "14:00", "hours": 1},
            today=self.today,
        )
        self.assertIn("7 days", sunday)
        closed = calendar_tool.run(
            {"action": "book", "date": "2026-11-01", "start": "14:00", "hours": 1},
            today=date(2026, 11, 1),
        )
        self.assertIn("closed", closed)

    def test_two_threads_do_not_double_book_the_same_room(self):
        errors = []

        def book():
            try:
                calendar_tool.run(
                    {"action": "book", "date": "2026-10-27", "start": "10:00", "hours": 1, "room": 2, "user_id": "t"},
                    today=self.today,
                )
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=book) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        listed = calendar_tool.run({"action": "list", "date": "2026-10-27"}, today=self.today)
        self.assertEqual(listed.count("Room 2"), 1)


class CatalogueTool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = os.environ.get("LIBRARY_DATA_DIR")
        os.environ["LIBRARY_DATA_DIR"] = self.tmp.name

    def tearDown(self):
        if self._old is None:
            os.environ.pop("LIBRARY_DATA_DIR", None)
        else:
            os.environ["LIBRARY_DATA_DIR"] = self._old
        self.tmp.cleanup()

    def test_clean_code_copy_count(self):
        text = catalogue.search({"query": "Clean Code"})
        self.assertIn("Clean Code", text)
        self.assertIn("2 of 2", text)


class CrmTool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = os.environ.get("LIBRARY_DATA_DIR")
        os.environ["LIBRARY_DATA_DIR"] = self.tmp.name

    def tearDown(self):
        if self._old is None:
            os.environ.pop("LIBRARY_DATA_DIR", None)
        else:
            os.environ["LIBRARY_DATA_DIR"] = self._old
        self.tmp.cleanup()

    def test_update_recall_and_last_question(self):
        saved = crm.run({"action": "update", "user_id": "ada", "name": "Ada", "email": "ada@example.com", "preference": "mystery"})
        self.assertIn("Ada", saved)
        self.assertIn("ada@example.com", crm.run({"action": "recall", "user_id": "ada"}))
        crm.note_interaction("ada", "How much is the fine for 4 days?")
        block = crm.prompt_block("ada")
        self.assertIn("Ada", block)
        self.assertIn("fine for 4 days", block)
        self.assertEqual(crm.prompt_block("nobody"), "")

    def test_a_tool_call_cannot_read_another_chats_saved_name(self):
        crm.run({"action": "update", "user_id": "shared", "name": "Fizza"})
        blocked = dispatch(ToolCall("patron_memory", {"action": "recall", "user_id": "shared"}, "shared"))
        self.assertNotIn("Fizza", blocked)
        self.assertIn("not available", blocked)


class Detector(unittest.TestCase):
    def test_fine_phrase_is_a_tool_call(self):
        calls = detect("How much is the fine for 4 days overdue?", "u")
        self.assertEqual(calls[0].name, "calculate_fine")
        self.assertEqual(calls[0].arguments["days"], 4)

    def test_policy_question_without_a_number_is_not_a_fine_call(self):
        self.assertEqual(detect("What is the overdue fine per day?"), [])

    def test_pronoun_follow_up_is_not_a_catalogue_search(self):
        self.assertEqual(detect("how many copies of it?"), [])

    def test_catalogue_title_is_kept(self):
        calls = detect("How many copies of Clean Code?")
        self.assertEqual(calls[0].name, "search_catalogue")
        self.assertIn("Clean Code", calls[0].arguments["query"])

    def test_saying_a_name_does_not_save_it_for_other_chats(self):
        self.assertEqual(detect("My name is Fizza.", "user-1"), [])

    def test_saved_memory_is_read_only_when_the_patron_asks(self):
        calls = detect("Please save my name for later. My name is Fizza.", "user-1")
        self.assertEqual(calls[0].name, "patron_memory")
        self.assertTrue(calls[0].allow_saved_memory)

    def test_model_tool_call_line_parses(self):
        call = parse_tool_call('TOOL_CALL: {"name":"calculate_fine","arguments":{"days":4,"items":1}}\n', user_id="u")
        self.assertEqual(call.name, "calculate_fine")
        self.assertEqual(call.user_id, "u")

    def test_stream_gate_releases_normal_text_and_holds_a_tool_call(self):
        plain = StreamGate()
        self.assertEqual(plain.feed("Sure"), ("token", "Sure"))
        tool = StreamGate()
        self.assertEqual(tool.feed("TOOL_"), ("hold", ""))
        kind, payload = tool.feed('CALL: {"name":"calculate_fine","arguments":{"days":4}}\n')
        self.assertEqual(kind, "tool")
        self.assertEqual(parse_tool_call(payload).arguments["days"], 4)


class Timeouts(unittest.TestCase):
    def test_a_slow_tool_returns_a_message_and_does_not_block_the_loop(self):
        def slow(_call):
            time.sleep(0.4)
            return "too late"

        async def run():
            import tools.orchestrator as orch
            original = orch.dispatch
            orch.dispatch = slow
            try:
                t0 = time.perf_counter()

                async def other():
                    await asyncio.sleep(0.05)
                    return "alive"

                message, alive = await asyncio.gather(
                    execute(ToolCall("calculate_fine", {"days": 1}), timeout=0.15),
                    other(),
                )
                elapsed = time.perf_counter() - t0
            finally:
                orch.dispatch = original
            return message, alive, elapsed

        message, alive, elapsed = asyncio.run(run())
        self.assertIn("too long", message)
        self.assertEqual(alive, "alive")
        self.assertLess(elapsed, 0.35)

    def test_a_crashing_tool_returns_a_message(self):
        async def run():
            import tools.orchestrator as orch
            original = orch.dispatch

            def boom(_call):
                raise RuntimeError("disk full")

            orch.dispatch = boom
            try:
                return await execute(ToolCall("study_room", {"action": "list"}))
            finally:
                orch.dispatch = original

        text = asyncio.run(run())
        self.assertIn("could not finish", text)
        self.assertIn("disk full", text)


class SentencesAndEndpoint(unittest.TestCase):
    def test_first_sentence_is_ready_before_the_rest(self):
        splitter = SentenceSplitter()
        self.assertEqual(splitter.feed("Hello there. "), ["Hello there."])
        self.assertEqual(splitter.feed("More is coming"), [])
        self.assertEqual(splitter.flush(), ["More is coming"])

    def test_a_price_does_not_split_the_sentence(self):
        splitter = SentenceSplitter()
        self.assertEqual(splitter.feed("The rate is $0.25 per day"), [])
        self.assertEqual(splitter.flush(), ["The rate is $0.25 per day"])

    def test_silence_after_speech_is_the_end(self):
        import numpy as np
        buf = SpeechBuffer()
        loud = (np.ones(1600, dtype=np.int16) * 8000).tobytes()
        quiet = np.zeros(12800, dtype=np.int16).tobytes()
        self.assertEqual(buf.add(loud), "speech")
        self.assertEqual(buf.add(quiet), "endpoint")


if __name__ == "__main__":
    unittest.main()
