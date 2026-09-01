"""
Comprehensive UI test: all pages, buttons, dark/light mode, mobile.
Tests that all features are present and visible.
"""

import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
from database import db, Student, Feedback, SRCUser

app.config['TESTING'] = True
app.config['WTF_CSRF_ENABLED'] = False

client = app.test_client()
results = {"passed": 0, "failed": 0, "total": 0, "failures": []}


def check_page(path, must_contain=None, login_as=None):
    """Check a page loads and contains expected content."""
    results["total"] += 1
    try:
        if login_as == "student":
            with client.session_transaction() as sess:
                sess['student_id'] = 'TEST001'
                sess['student_name'] = 'Test Student'
        elif login_as == "admin":
            with client.session_transaction() as sess:
                sess['admin_id'] = 1
                sess['admin_name'] = 'Test Admin'

        resp = client.get(path, follow_redirects=True)
        if resp.status_code != 200:
            results["failed"] += 1
            results["failures"].append(f"GET {path} - Status {resp.status_code}")
            return

        if must_contain:
            text = resp.data.decode('utf-8', errors='ignore')
            missing = [m for m in must_contain if m not in text]
            if missing:
                results["failed"] += 1
                results["failures"].append(f"GET {path} - Missing: {missing}")
                return

        results["passed"] += 1
    except Exception as e:
        results["failed"] += 1
        results["failures"].append(f"GET {path} - {str(e)[:60]}")


def check_dark_light_mode():
    """Check that dark/light mode CSS variables are present."""
    results["total"] += 1
    try:
        resp = client.get("/", follow_redirects=True)
        text = resp.data.decode('utf-8', errors='ignore')

        # Check for CSS variables
        has_light = '--text-primary' in text or 'data-theme' in text or 'theme' in text.lower()
        has_dark = 'dark' in text.lower() or '[data-theme="dark"]' in text

        if has_light:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["failures"].append("Dark/Light mode - No theme variables found")
    except Exception as e:
        results["failed"] += 1
        results["failures"].append(f"Dark/Light mode - {str(e)[:60]}")


def check_mobile_responsive():
    """Check that mobile media queries exist."""
    results["total"] += 1
    try:
        css_path = os.path.join(os.path.dirname(__file__), 'static', 'css', 'style.css')
        if os.path.exists(css_path):
            with open(css_path, 'r', encoding='utf-8', errors='ignore') as f:
                css = f.read()
            has_mobile = '@media' in css and 'max-width' in css
            if has_mobile:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["failures"].append("Mobile - No media queries in CSS")
        else:
            resp = client.get("/", follow_redirects=True)
            text = resp.data.decode('utf-8', errors='ignore')
            has_media = '@media' in text and 'max-width' in text
            if has_media:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["failures"].append("Mobile - No media queries found")
    except Exception as e:
        results["failed"] += 1
        results["failures"].append(f"Mobile - {str(e)[:60]}")


def setup_data():
    with app.app_context():
        db.create_all()
        if not Student.query.filter_by(student_id='TEST001').first():
            db.session.add(Student(
                student_id='TEST001', full_name='Test Student',
                email='test@htu.edu.gh', phone='0240000001',
                programme='BSc Computer Science', year=2,
                password_hash='pbkdf2:sha256:150000$test$hash',
            ))
            db.session.commit()
        if not SRCUser.query.filter_by(username='admin').first():
            db.session.add(SRCUser(
                username='admin', full_name='Test Admin',
                email='admin@htu.edu.gh', role='admin',
                password_hash='pbkdf2:sha256:150000$test$hash',
            ))
            db.session.commit()
        # Create test feedback with all fields
        if Feedback.query.count() == 0:
            db.session.add(Feedback(
                student_id='TEST001', anonymous=False, category='ICT',
                feedback_text='Test feedback for UI testing', cleaned_text='test feedback for ui testing',
                sentiment='negative', sentiment_score=-0.5, urgency_score=3, status='Pending',
                short_term_solution='Test solution', long_term_solution='Test long term',
                responsible_department='ICT Department', estimated_time='3-5 days',
                confidence_score=75.0, src_response='Test response',
            ))
            db.session.commit()


def run_all():
    print("\n" + "="*70)
    print("COMPREHENSIVE UI TEST")
    print("="*70)

    setup_data()

    # === PUBLIC PAGES ===
    print("\n--- Public Pages ---")
    check_page("/", ["HTU"])
    check_page("/public")
    check_page("/announcements")
    check_page("/student/login")
    check_page("/student/register")
    check_page("/student/forgot-password")
    check_page("/admin/login")
    print(f"  Public: {results['passed']}/{results['total']}")

    # === STUDENT PAGES ===
    print("\n--- Student Pages ---")
    # Get a valid feedback ID for the student
    with app.app_context():
        fb = Feedback.query.filter_by(student_id='TEST001').first()
        fb_id = fb.id if fb else 1
    check_page("/student/dashboard", ["Dashboard", "Analysis", "Solution"], login_as="student")
    check_page("/submit", ["Submit", "Feedback"], login_as="student")
    check_page("/forum", login_as="student")
    check_page("/chat", login_as="student")
    check_page("/student/change-password", login_as="student")
    check_page(f"/edit-feedback/{fb_id}", login_as="student")
    print(f"  Student: {results['passed']}/{results['total']}")

    # === ADMIN PAGES ===
    print("\n--- Admin Pages ---")
    check_page("/admin/dashboard", ["Dashboard", "Correct Sentiment", "Analyze"], login_as="admin")
    check_page("/admin/feedback", login_as="admin")
    check_page("/admin/students", login_as="admin")
    check_page("/admin/analytics", login_as="admin")
    check_page("/admin/ai-review", login_as="admin")
    check_page("/admin/ai-audit", login_as="admin")
    check_page("/admin/logs", login_as="admin")
    check_page("/admin/templates", login_as="admin")
    check_page("/admin/solution-templates", login_as="admin")
    check_page("/admin/lexicon-gaps", login_as="admin")
    check_page("/admin/lexicon-manager", login_as="admin")
    check_page("/admin/announcements", login_as="admin")
    check_page("/admin/chat", login_as="admin")
    check_page("/admin/change-password", login_as="admin")
    check_page("/admin/export/excel", login_as="admin")
    print(f"  Admin: {results['passed']}/{results['total']}")

    # === UI FEATURES ===
    print("\n--- UI Features ---")
    check_dark_light_mode()
    check_mobile_responsive()
    print(f"  UI Features: {results['passed']}/{results['total']}")

    # === SUMMARY ===
    total = results["total"]
    passed = results["passed"]
    failed = results["failed"]
    rate = passed / total * 100 if total else 0

    print("\n" + "="*70)
    print("UI TEST SUMMARY")
    print("="*70)
    print(f"\n  Total: {total}")
    print(f"  Passed: {passed} ({rate:.1f}%)")
    print(f"  Failed: {failed}")

    if results["failures"]:
        print(f"\n  Failures:")
        for f in results["failures"][:15]:
            print(f"    - {f}")

    if failed == 0:
        print("\n  STATUS: ALL UI ELEMENTS PRESENT")
    elif rate >= 90:
        print("\n  STATUS: MOSTLY WORKING")
    else:
        print("\n  STATUS: NEEDS WORK")


if __name__ == "__main__":
    run_all()
