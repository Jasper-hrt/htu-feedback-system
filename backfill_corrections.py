"""backfill_corrections.py

One-time migration: populate `feedback_correction_memory` from every existing
`SentimentCorrection` row so the new active-learning memory starts with full
historical coverage from day one.

Idempotent: re-running it skips feedback rows that already have a memory row.

Usage:
    python backfill_corrections.py

Requires a Flask app context to talk to the database. Imports the same
`app` and `db` that `app.py` uses, so it picks up your real DATABASE_URL.
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        from app import app, db  # noqa: F401  (imports register models + config)
    except Exception as exc:
        print(f"Failed to import app: {exc}", file=sys.stderr)
        return 1

    with app.app_context():
        from database import Feedback, SentimentCorrection, FeedbackCorrectionMemory
        from sentiment.similarity_cache import memory_record_correction

        corrections = (
            db.session.query(SentimentCorrection)
            .order_by(SentimentCorrection.created_at.asc())
            .all()
        )
        if not corrections:
            print("No SentimentCorrection rows found. Nothing to backfill.")
            return 0

        existing_ids = {
            row.feedback_id
            for row in db.session.query(FeedbackCorrectionMemory.feedback_id).all()
        }

        inserted = 0
        skipped = 0
        for c in corrections:
            if c.feedback_id in existing_ids:
                skipped += 1
                continue
            feedback = db.session.get(Feedback, c.feedback_id)
            if not feedback:
                continue
            try:
                memory_record_correction(
                    feedback, db.session,
                    admin_name=c.admin_name or "backfill",
                )
                inserted += 1
            except Exception as exc:
                print(f"  feedback_id={c.feedback_id}: {exc}", file=sys.stderr)
                continue

        db.session.commit()

        total = db.session.query(FeedbackCorrectionMemory).count()
        print(
            f"Backfill complete. inserted={inserted} skipped={skipped} "
            f"total_memory_rows={total}"
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
