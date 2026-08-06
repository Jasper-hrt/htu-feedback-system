# TODO - Fix 500 Internal Server Error

## Root Cause
The 500 "Internal Server Error" occurs on the deployed **PostgreSQL (Render)**
environment because the `feedback` table may be missing newer columns that the
submission INSERT requires, and the SentiWordNet/NLTK pipeline can be slow or fail.
The primary cause of the *startup* crash is the **eager NLTK import** in
`sentiment/sentiwordnet_engine.py`, which fails when NLTK corpora
(wordnet/sentiwordnet/punkt/omw-1.4) are missing on a fresh deploy.

## Steps
- [x] Harden `_ensure_schema_aligned()` to be robust on PostgreSQL (proper transaction handling, rollback on failure).
- [x] Add a global error handler in app.py that logs the full traceback and returns a user-friendly error page.
- [x] Make the SentiWordNet engine resilient (lazy-load + graceful fallback when NLTK data is unavailable).
- [x] Harden `HybridSentimentEngine` so a single engine failure never crashes feedback submission.
- [x] Improve NLTK data handling so the app avoids re-downloading resources on every request (which causes slow responses/timeouts).
- [x] Test locally with SQLite to confirm no regression.
- [x] Verify the fixes handle the deployed PostgreSQL schema.
