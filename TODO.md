# TODO - Fix 500 Internal Server Error

## Root Cause
The 500 "Internal Server Error" occurs on the deployed **PostgreSQL (Render)**
environment because the `feedback` table may be missing newer columns that the
submission INSERT requires, and the SentiWordNet/NLTK pipeline can be slow or fail.

## Steps
- [ ] Harden `_ensure_schema_aligned()` to be robust on PostgreSQL (proper transaction handling, rollback on failure).
- [ ] Add a global error handler in app.py that logs the full traceback and returns a user-friendly error page.
- [ ] Make the SentiWordNet engine resilient (caching + graceful fallback when NLTK data is unavailable).
- [ ] Improve NLTK data handling so the app avoids re-downloading resources on every request (which causes slow responses/timeouts).
- [ ] Test locally with SQLite to confirm no regression.
- [ ] Verify the fixes handle the deployed PostgreSQL schema.
