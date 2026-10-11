"""Run once from the backend folder:  python fix_books.py
Removes the repeated Standard/Premium wording from the 50 book entries."""
import glob

n = 0
for f in glob.glob("corpus/books/*.md"):
    t = open(f, encoding="utf-8").read()
    t2 = t.replace("**Standard loan period:** 2 weeks (Standard membership) / 3 weeks (Premium membership)",
                   "**Loan period:** standard loan (see the Borrowing and Renewals policy)")
    t2 = t2.replace("**Standard loan period:** 1 week (Standard membership) / 2 weeks (Premium membership) - new-release short loan",
                    "**Loan period:** new-release short loan, shorter than the standard loan (see the Borrowing and Renewals policy)")
    t2 = t2.replace("the pickup window after a held copy becomes available is 3 days for Standard members and 5 days for Premium members.",
                    "the pickup window after a held copy becomes available is set by the Holds and Reservations policy.")
    if t2 != t:
        n += 1
        open(f, "w", encoding="utf-8").write(t2)
print("edited", n, "book files (expected 50; 0 means already done)")
