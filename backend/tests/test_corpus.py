"""Corpus checks: size requirement, structure, and agreement with the static knowledge base."""

import glob
import os
import re
import unittest

from tests import support  # noqa: F401  (sets sys.path)
from prompts import LIBRARY_KNOWLEDGE_BASE

CORPUS = os.path.join(support.BACKEND_DIR, "corpus")
BOOKS = sorted(glob.glob(os.path.join(CORPUS, "books", "*.md")))
POLICIES = sorted(glob.glob(os.path.join(CORPUS, "policies", "*.md")))


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class CorpusShape(unittest.TestCase):
    def test_size_is_within_assignment_range(self):
        n = len(BOOKS) + len(POLICIES)
        self.assertTrue(50 <= n <= 100, f"assignment requires 50-100 documents, found {n}")

    def test_every_doc_has_title_and_substance(self):
        for f in BOOKS + POLICIES:
            text = read(f)
            self.assertTrue(text.lstrip().startswith("# "), f"{f} has no '# Title' heading")
            self.assertGreaterEqual(len(text.split()), 40, f"{f} is too thin to be useful")

    def test_titles_are_unique(self):
        titles = [read(f).splitlines()[0] for f in BOOKS + POLICIES]
        self.assertEqual(len(titles), len(set(titles)))


class BookRecords(unittest.TestCase):
    def test_copy_counts_are_coherent(self):
        for f in BOOKS:
            t = read(f)
            m = re.search(r"\*\*Copies:\*\* (\d+) total, (\d+) currently available", t)
            self.assertIsNotNone(m, f)
            total, avail = int(m.group(1)), int(m.group(2))
            self.assertLessEqual(avail, total, f)
            self.assertIn(f"{avail} of {total} copies available", t, f)
            holds = int(re.search(r"\*\*Holds in queue:\*\* (\d+)", t).group(1))
            if avail > 0:
                self.assertEqual(holds, 0, f"{f}: holds queue while copies are available")

    def test_loan_periods_match_policy(self):
        allowed = {
            "2 weeks (Standard membership) / 3 weeks (Premium membership)",
            "1 week (Standard membership) / 2 weeks (Premium membership) - new-release short loan",
        }
        for f in BOOKS:
            m = re.search(r"\*\*Standard loan period:\*\* (.*)", read(f))
            self.assertIn(m.group(1).strip(), allowed, f)

    def test_pickup_window_is_consistent(self):
        for f in BOOKS:
            self.assertIn("3 days for Standard members and 5 days for Premium members", read(f), f)


class AgreesWithStaticKnowledgeBase(unittest.TestCase):
    """The static prompt facts and the retrieved documents must never contradict each other."""

    def test_catalogue_entries_in_prompt_match_documents(self):
        entries = re.findall(r'"([^"]+)" - .+? - .+? - (\d+) copies, (\d+) available', LIBRARY_KNOWLEDGE_BASE)
        self.assertGreaterEqual(len(entries), 6)
        by_title = {read(f).splitlines()[0][2:].strip(): read(f) for f in BOOKS}
        for title, total, avail in entries:
            self.assertIn(title, by_title, f"static KB lists '{title}' but there is no document for it")
            self.assertIn(f"{total} total, {avail} currently available", by_title[title], title)

    def test_hours(self):
        hours = read(os.path.join(CORPUS, "policies", "library_hours.md"))
        self.assertIn("Monday to Friday: 9:00 AM - 8:00 PM", hours)
        self.assertIn("Saturday: 10:00 AM - 5:00 PM", hours)
        self.assertIn("Sunday: Closed", hours)
        self.assertIn("Mon-Fri 9:00-20:00, Sat 10:00-17:00, Sun closed", LIBRARY_KNOWLEDGE_BASE)

    def test_fines_and_lost_item_fee(self):
        fines = read(os.path.join(CORPUS, "policies", "overdue_fines_and_lost_items.md"))
        self.assertIn("$0.25 per day", fines)
        self.assertIn("$10.00", fines)
        self.assertIn("$5.00 processing fee", fines)
        self.assertIn("$0.25/day per item, capped at $10/item", LIBRARY_KNOWLEDGE_BASE)
        self.assertIn("replacement cost + $5 processing fee", LIBRARY_KNOWLEDGE_BASE)

    def test_membership_tiers(self):
        tiers = read(os.path.join(CORPUS, "policies", "membership_tiers.md"))
        for needle in ("up to 5 items", "14-day loan period", "up to 15 items", "21-day loan period"):
            self.assertIn(needle, tiers)
        self.assertIn("5 items at a time, 14-day loan period", LIBRARY_KNOWLEDGE_BASE)
        self.assertIn("15 items at a time, 21-day loan period", LIBRARY_KNOWLEDGE_BASE)

    def test_renewals_and_pickup(self):
        borrowing = read(os.path.join(CORPUS, "policies", "borrowing_and_renewals.md"))
        self.assertIn("up to 2 times", borrowing)
        self.assertIn("2 days\n  before the due date", borrowing)
        self.assertIn("renewed twice", LIBRARY_KNOWLEDGE_BASE)
        self.assertIn("up to 2 days before the due date", LIBRARY_KNOWLEDGE_BASE)
        holds = read(os.path.join(CORPUS, "policies", "holds_and_reservations.md"))
        self.assertIn("3 days for Standard members", holds)
        self.assertIn("5 days for Premium members", holds)
        self.assertIn("3 days", LIBRARY_KNOWLEDGE_BASE)
        self.assertIn("5 days for Premium", LIBRARY_KNOWLEDGE_BASE)


if __name__ == "__main__":
    unittest.main()
