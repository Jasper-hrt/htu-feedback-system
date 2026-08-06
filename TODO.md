# TODO - Fix Feedback Submission 500 Error After Deploy

## Root Cause
The app runs on **PostgreSQL** when deployed to Render, but the schema-repair logic
(`_repair_feedback_schema_if_needed`) is SQLite-only and is skipped on PostgreSQL.
`db.create_all()` only creates missing **tables**, NOT missing **columns** on existing
tables. So the deployed `feedback` table is missing columns that the submission
INSERT needs (`confidence_score`, `dominant_emotion`, `emotion_intensities`,
`recommended_keywords`, `short_term_solution`, etc.), causing a 500 error.

## Steps
- [ ] Add a dialect-agnostic `_ensure_feedback_columns()` in `app.py` that adds
      missing `feedback` columns on BOTH SQLite and PostgreSQL.
- [ ] Call `_ensure_feedback_columns()` in the startup block for all environments.
- [ ] Add `sentiwordnet` to the runtime NLTK safety-net list for robustness.
- [ ] Test locally with SQLite to confirm no regression.
- [ ] Verify the fix handles the deployed PostgreSQL schema.
