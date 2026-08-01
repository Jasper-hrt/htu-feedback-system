"""
reset_db.py - Clear all stored data from the HTU SRC Feedback System database
while preserving the admin account (src_users table).

Usage: python reset_db.py
"""

import sqlite3
import os

DB_PATH = os.path.join('instance', 'feedback.db')

def reset_database():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}. Nothing to reset.")
        return

    print(f"Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Found tables: {', '.join(tables)}")

    # Tables to clear (all user-generated / session data)
    tables_to_clear = [
        'students',           # Student registrations
        'feedback',           # Feedback submissions
        'feedback_votes',     # Feedback votes
        'forum_topics',       # Forum topics
        'forum_replies',      # Forum replies
        'forum_topic_votes',  # Topic votes
        'forum_topic_tags',   # Topic tags
        'chat_rooms',         # Chat rooms
        'chat_messages',      # Chat messages
        'chat_room_members',  # Chat room memberships
        'chat_room_sentiment',# Chat sentiment stats
        'announcements',      # Announcements
        'password_reset_tokens',  # Password reset tokens
        'system_logs',        # System logs
    ]

    # Tables to PRESERVE (admin accounts)
    tables_to_preserve = [
        'src_users',          # SRC Admin accounts
    ]

    # Verify all tables exist before proceeding
    missing_tables = [t for t in tables_to_clear if t not in tables]
    if missing_tables:
        print(f"Warning: These tables don't exist and will be skipped: {', '.join(missing_tables)}")
        tables_to_clear = [t for t in tables_to_clear if t in tables]

    if not tables_to_clear:
        print("No tables to clear. Database may already be empty.")
        conn.close()
        return

    # Confirm with user
    print("\n=== RESET SUMMARY ===")
    print(f"Tables to CLEAR ({len(tables_to_clear)}): {', '.join(tables_to_clear)}")
    print(f"Tables to PRESERVE ({len(tables_to_preserve)}): {', '.join(tables_to_preserve)}")
    print("======================")

    # Count rows before clearing
    total_before = 0
    for table in tables_to_clear:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        total_before += count
        if count > 0:
            print(f"  {table}: {count} rows")

    print(f"\nTotal records to delete: {total_before}")

    if total_before == 0:
        print("Database is already clean. Nothing to do.")
        conn.close()
        return

    # Proceed with clearing
    try:
        # Disable foreign key checks temporarily
        cursor.execute("PRAGMA foreign_keys = OFF")

        for table in tables_to_clear:
            cursor.execute(f"DELETE FROM {table}")
            print(f"  ✓ Cleared {table}")

        # Verify admin still exists
        cursor.execute("SELECT COUNT(*) FROM src_users")
        admin_count = cursor.fetchone()[0]
        if admin_count == 0:
            print("\n⚠ Admin accounts were not found! Recreating default admin...")
            from werkzeug.security import generate_password_hash
            cursor.execute(
                "INSERT INTO src_users (username, password_hash, full_name, role) VALUES (?, ?, ?, ?)",
                ('admin', generate_password_hash('admin123'), 'SRC Administrator', 'President')
            )
            print("  ✓ Default admin recreated (admin / admin123)")

        # Verify re-enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")

        conn.commit()
        print(f"\n✅ Database reset complete! {total_before} records deleted.")
        print(f"   Admin account preserved: admin / admin123")

    except Exception as e:
        conn.rollback()
        print(f"\n❌ Error during reset: {e}")
        raise
    finally:
        conn.close()

if __name__ == '__main__':
    reset_database()

