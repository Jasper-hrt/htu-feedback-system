"""
Test all admin and student pages, tabs, buttons, and features.
Uses Flask test client to verify all routes return 200 and contain expected content.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
from database import db, Student, Feedback
from datetime import datetime


# ==================== TEST CONFIGURATION ====================

app.config['TESTING'] = True
app.config['WTF_CSRF_ENABLED'] = False
app.config['RATELIMIT_ENABLED'] = False

# Create test client
client = app.test_client()

# Test results
results = {"passed": 0, "failed": 0, "total": 0, "failures": []}


def get_csrf_token():
    """Get CSRF token from session."""
    with client.session_transaction() as sess:
        return sess.get('_csrf_token', '')


def test_route(method, path, expected_status=None, expected_content=None, data=None):
    """Test a single route."""
    results["total"] += 1
    try:
        if method == "GET":
            resp = client.get(path)
            if expected_status is None:
                expected_status = 200
        elif method == "POST":
            # Add CSRF token to POST data
            csrf_token = get_csrf_token()
            if data is None:
                data = {}
            data['csrf_token'] = csrf_token
            resp = client.post(path, data=data, follow_redirects=True)
            if expected_status is None:
                expected_status = 200
        else:
            results["failed"] += 1
            results["failures"].append(f"{method} {path} - Unsupported method")
            return

        if resp.status_code != expected_status:
            results["failed"] += 1
            results["failures"].append(f"{method} {path} - Expected {expected_status}, got {resp.status_code}")
            return

        if expected_content:
            text = resp.data.decode('utf-8', errors='ignore')
            items = expected_content if isinstance(expected_content, list) else [expected_content]
            missing = [c for c in items if c not in text]
            if missing:
                results["failed"] += 1
                results["failures"].append(f"{method} {path} - Missing: {missing}")
                return

        results["passed"] += 1
    except Exception as e:
        results["failed"] += 1
        results["failures"].append(f"{method} {path} - {str(e)[:80]}")


def login_student():
    with client.session_transaction() as sess:
        sess['student_id'] = 'TEST001'
        sess['student_name'] = 'Test Student'
        sess['student_email'] = 'test@htu.edu.gh'


def login_admin():
    with client.session_transaction() as sess:
        sess['admin_id'] = 1
        sess['admin_name'] = 'Test Admin'


def setup_data():
    with app.app_context():
        db.create_all()
        if not Student.query.filter_by(student_id="TEST001").first():
            db.session.add(Student(
                student_id="TEST001", full_name="Test Student",
                email="test@htu.edu.gh", phone="0240000001",
                programme="BSc Computer Science", year=2,
                password_hash="pbkdf2:sha256:150000$test$hash",
            ))
            db.session.commit()
        if Feedback.query.filter_by(student_id="TEST001").count() == 0:
            for i, tf in enumerate([
                {"text": "The wifi in the library is very slow", "cat": "ICT", "sent": "negative", "urg": 3},
                {"text": "The lecturer explains topics very well", "cat": "Academics", "sent": "positive", "urg": 1},
                {"text": "The canteen food is cold", "cat": "Catering", "sent": "negative", "urg": 3},
                {"text": "The hostel water supply is unreliable", "cat": "Accommodation", "sent": "negative", "urg": 4},
                {"text": "The security team responded quickly", "cat": "Safety", "sent": "positive", "urg": 2},
            ]):
                db.session.add(Feedback(
                    student_id="TEST001", anonymous=False, category=tf["cat"],
                    feedback_text=tf["text"], cleaned_text=tf["text"].lower(),
                    sentiment=tf["sent"], sentiment_score=0.5 if tf["sent"]=="positive" else -0.5,
                    urgency_score=tf["urg"], status="Pending",
                    short_term_solution="Test solution", long_term_solution="Test long term",
                    responsible_department="Test Dept", estimated_time="3-5 days",
                    confidence_score=75.0,
                ))
            db.session.commit()


# ==================== TESTS ====================

def test_all():
    print("\n" + "="*70)
    print("FULL SYSTEM PAGE & FEATURE TEST")
    print("="*70)

    t0 = time.time()
    setup_data()

    # --- PUBLIC PAGES ---
    print("\n--- Public Pages ---")
    test_route("GET", "/", expected_content=["HTU"])
    test_route("GET", "/public")
    test_route("GET", "/announcements")
    test_route("GET", "/student/login")
    test_route("GET", "/student/register")
    test_route("GET", "/student/forgot-password")
    test_route("GET", "/admin/login")
    print(f"  Public: {results['passed']}/{results['total']}")

    # --- STUDENT PAGES ---
    print("\n--- Student Pages ---")
    login_student()
    test_route("GET", "/student/dashboard", expected_content=["Dashboard"])
    test_route("GET", "/submit", expected_content=["Submit"])
    test_route("GET", "/student/change-password")
    test_route("GET", "/chat")
    test_route("GET", "/forum")
    print(f"  Student pages cumulative: {results['passed']}/{results['total']}")

    # --- STUDENT FEATURES ---
    print("\n--- Student Features ---")
    # Test submit for one category
    test_route("POST", "/submit", data={
        "category": "Wi-Fi Issues", "location": "Library",
        "feedback_text": "Test feedback for Wi-Fi Issues with descriptive text about an issue that needs attention.",
        "anonymous": "on", "user_urgency": "3",
    })
    # Get a feedback ID that belongs to TEST001 for vote/delete/edit
    with app.app_context():
        fb = Feedback.query.filter_by(student_id="TEST001").order_by(Feedback.id.desc()).first()
        test_fb_id = fb.id if fb else 1
    # GET edit page for own feedback (before delete)
    test_route("GET", f"/edit-feedback/{test_fb_id}")
    # POST edit
    test_route("POST", f"/edit-feedback/{test_fb_id}", data={
        "category": "Wi-Fi Issues",
        "feedback_text": "Updated feedback text for testing with more details",
        "user_urgency": "3",
    })
    # Vote (POST) - can't vote on own feedback (expected 400)
    test_route("POST", f"/vote/{test_fb_id}", expected_status=400)
    # Delete (POST) - use own feedback
    test_route("POST", f"/delete-feedback/{test_fb_id}")
    print(f"  Student features cumulative: {results['passed']}/{results['total']}")

    # --- ADMIN PAGES ---
    print("\n--- Admin Pages ---")
    login_admin()
    test_route("GET", "/admin/dashboard", expected_content=["Dashboard"])
    test_route("GET", "/admin/feedback", expected_content=["Feedback"])
    test_route("GET", "/admin/students", expected_content=["Students"])
    test_route("GET", "/admin/analytics", expected_content=["Analytics"])
    test_route("GET", "/admin/ai-review")
    test_route("GET", "/admin/ai-audit")
    test_route("GET", "/admin/logs")
    test_route("GET", "/admin/templates")
    test_route("GET", "/admin/solution-templates")
    test_route("GET", "/admin/lexicon-gaps")
    test_route("GET", "/admin/lexicon-manager")
    test_route("GET", "/admin/announcements")
    test_route("GET", "/admin/chat")
    test_route("GET", "/admin/change-password")
    test_route("GET", "/admin/2fa/setup")
    test_route("GET", "/admin/export/excel")
    print(f"  Admin pages cumulative: {results['passed']}/{results['total']}")

    # --- ADMIN FEATURES ---
    print("\n--- Admin Features ---")
    # Create a dedicated test feedback for admin actions
    with app.app_context():
        admin_test = Feedback(
            student_id="TEST001", anonymous=False, category="Wi-Fi Issues",
            feedback_text="Admin test feedback for actions", cleaned_text="admin test feedback for actions",
            sentiment="negative", sentiment_score=-0.5, urgency_score=3, status="Pending",
            short_term_solution="Test", long_term_solution="Test", responsible_department="Test",
            estimated_time="3 days", confidence_score=75.0,
        )
        db.session.add(admin_test)
        db.session.commit()
        admin_fb_id = admin_test.id
    test_route("POST", f"/admin/update/{admin_fb_id}", data={"status": "In Progress"})
    test_route("POST", f"/admin/delete-feedback/{admin_fb_id}")
    # Create another feedback for non-destructive tests
    with app.app_context():
        admin_test2 = Feedback(
            student_id="TEST001", anonymous=False, category="Maintenance",
            feedback_text="Admin test feedback for non-destructive actions", cleaned_text="admin test",
            sentiment="neutral", sentiment_score=0.0, urgency_score=2, status="Pending",
            short_term_solution="Test", long_term_solution="Test", responsible_department="Test",
            estimated_time="3 days", confidence_score=75.0,
        )
        db.session.add(admin_test2)
        db.session.commit()
        admin_fb_id2 = admin_test2.id
    test_route("POST", f"/admin/feedback/{admin_fb_id2}/regenerate-recommendation")
    test_route("POST", f"/admin/ai-review/{admin_fb_id2}", data={"action": "approve"})
    test_route("POST", f"/admin/ai-review/{admin_fb_id2}/add-to-lexicon")
    test_route("POST", "/admin/solution-templates", data={
        "category": "Wi-Fi Issues", "keywords": "wifi,internet",
        "short_term": "Check settings", "long_term": "Upgrade infrastructure",
    })
    test_route("POST", "/admin/announcements", data={"title": "Test", "content": "Test announcement"})
    print(f"  Admin features cumulative: {results['passed']}/{results['total']}")

    # --- API ENDPOINTS ---
    print("\n--- API Endpoints ---")
    test_route("GET", "/api/notifications")
    test_route("GET", "/api/notifications/unread-count")
    test_route("GET", "/api/admin/notifications")
    test_route("GET", "/api/admin/notifications/unread-count")
    test_route("GET", "/api/admin/trending")
    test_route("GET", "/api/admin/export/analytics")
    test_route("GET", "/api/admin/export/logs")
    test_route("GET", "/api/admin/export/students")
    print(f"  API endpoints cumulative: {results['passed']}/{results['total']}")

    # --- SUMMARY ---
    elapsed = time.time() - t0
    total = results["total"]
    passed = results["passed"]
    failed = results["failed"]
    rate = passed / total * 100 if total else 0

    print("\n" + "="*70)
    print("PAGE & FEATURE TEST SUMMARY")
    print("="*70)
    print(f"\n  Total: {total}")
    print(f"  Passed: {passed} ({rate:.1f}%)")
    print(f"  Failed: {failed}")
    print(f"  Time: {elapsed:.1f}s")

    if results["failures"]:
        print(f"\n  Failures ({len(results['failures'])}):")
        for f in results["failures"][:15]:
            print(f"    - {f}")

    if failed == 0:
        print("\n  STATUS: ALL PAGES AND FEATURES WORKING")
    elif rate >= 90:
        print("\n  STATUS: MOSTLY WORKING - Minor issues")
    elif rate >= 75:
        print("\n  STATUS: ACCEPTABLE - Some issues to fix")
    else:
        print("\n  STATUS: NEEDS WORK")


if __name__ == "__main__":
    test_all()
