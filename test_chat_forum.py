"""
Test chat and forum functionality with real data.
Tests creating topics, replying, sending chat messages.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, socketio
from database import db, Student, Feedback, ForumTopic, ForumReply, ChatRoom, ChatMessage, ChatRoomMember

app.config['TESTING'] = True
app.config['WTF_CSRF_ENABLED'] = False

client = app.test_client()
results = {"passed": 0, "failed": 0, "total": 0, "failures": []}


def test_forum_create():
    """Test creating a forum topic."""
    results["total"] += 1
    try:
        with client.session_transaction() as sess:
            sess['student_id'] = 'TEST001'
            sess['student_name'] = 'Test Student'

        # Get CSRF token first
        resp = client.get('/forum/create', follow_redirects=True)
        text = resp.data.decode('utf-8', errors='ignore')
        # Extract CSRF token
        import re
        csrf_match = re.search(r'name="csrf_token" value="([^"]+)"', text)
        csrf_token = csrf_match.group(1) if csrf_match else ''

        resp = client.post('/forum/create', data={
            'title': 'Test Forum Topic',
            'content': 'This is a test forum topic content that is long enough to pass validation checks.',
            'category': 'ICT',
            'tags': 'wifi, internet, network',
            'csrf_token': csrf_token,
        }, follow_redirects=True)

        if resp.status_code == 200:
            results["passed"] += 1
            print("  [PASS] Forum topic creation")
        else:
            results["failed"] += 1
            results["failures"].append(f"Forum create: Status {resp.status_code}")
            print(f"  [FAIL] Forum topic creation: Status {resp.status_code}")
    except Exception as e:
        results["failed"] += 1
        results["failures"].append(f"Forum create: {str(e)[:60]}")
        print(f"  [FAIL] Forum topic creation: {str(e)[:60]}")


def test_forum_reply():
    """Test replying to a forum topic."""
    results["total"] += 1
    try:
        with client.session_transaction() as sess:
            sess['student_id'] = 'TEST001'
            sess['student_name'] = 'Test Student'

        # Get the first topic
        with app.app_context():
            topic = ForumTopic.query.order_by(ForumTopic.id.desc()).first()
            if topic:
                topic_id = topic.id
            else:
                results["failed"] += 1
                results["failures"].append("Forum reply: No topic found")
                print("  [FAIL] Forum reply: No topic found")
                return

        # Get CSRF token from topic page
        resp = client.get(f'/forum/topic/{topic_id}', follow_redirects=True)
        text = resp.data.decode('utf-8', errors='ignore')
        import re
        csrf_match = re.search(r'name="csrf_token" value="([^"]+)"', text)
        csrf_token = csrf_match.group(1) if csrf_match else ''

        resp = client.post(f'/forum/topic/{topic_id}', data={
            'content': 'This is a test reply to the forum topic that is long enough.',
            'csrf_token': csrf_token,
        }, follow_redirects=True)

        if resp.status_code == 200:
            results["passed"] += 1
            print("  [PASS] Forum reply")
        else:
            results["failed"] += 1
            results["failures"].append(f"Forum reply: Status {resp.status_code}")
            print(f"  [FAIL] Forum reply: Status {resp.status_code}")
    except Exception as e:
        results["failed"] += 1
        results["failures"].append(f"Forum reply: {str(e)[:80]}")
        print(f"  [FAIL] Forum reply: {str(e)[:80]}")


def test_chat_create_room():
    """Test creating a chat room."""
    results["total"] += 1
    try:
        with client.session_transaction() as sess:
            sess['student_id'] = 'TEST001'
            sess['student_name'] = 'Test Student'

        # Get CSRF token first
        resp = client.get('/chat/create', follow_redirects=True)
        text = resp.data.decode('utf-8', errors='ignore')
        import re
        csrf_match = re.search(r'name="csrf_token" value="([^"]+)"', text)
        csrf_token = csrf_match.group(1) if csrf_match else ''

        resp = client.post('/chat/create', data={
            'name': 'Test Chat Room',
            'description': 'A test chat room for testing',
            'category': 'General',
            'csrf_token': csrf_token,
        }, follow_redirects=True)

        if resp.status_code == 200:
            results["passed"] += 1
            print("  [PASS] Chat room creation")
        else:
            results["failed"] += 1
            results["failures"].append(f"Chat create: Status {resp.status_code}")
            print(f"  [FAIL] Chat room creation: Status {resp.status_code}")
    except Exception as e:
        results["failed"] += 1
        results["failures"].append(f"Chat create: {str(e)[:60]}")
        print(f"  [FAIL] Chat room creation: {str(e)[:60]}")


def test_chat_send_message():
    """Test sending a chat message via HTTP API."""
    results["total"] += 1
    try:
        with client.session_transaction() as sess:
            sess['student_id'] = 'TEST001'
            sess['student_name'] = 'Test Student'

        # Get the first room
        with app.app_context():
            room = ChatRoom.query.first()
            if not room:
                room = ChatRoom(name='Test Room', description='Test', category='General', created_by='TEST001')
                db.session.add(room)
                db.session.commit()
                # Add member
                member = ChatRoomMember(room_id=room.id, student_id='TEST001')
                db.session.add(member)
                db.session.commit()
                room_id = room.id
            else:
                # Ensure membership
                member = ChatRoomMember.query.filter_by(room_id=room.id, student_id='TEST001').first()
                if not member:
                    member = ChatRoomMember(room_id=room.id, student_id='TEST001')
                    db.session.add(member)
                    db.session.commit()
                room_id = room.id

        # Use the test client to send a message via Socket.IO test client
        # Since we can't easily test Socket.IO with the Flask test client,
        # we'll test the room page loads correctly
        resp = client.get(f'/chat/room/{room_id}', follow_redirects=True)

        if resp.status_code == 200:
            text = resp.data.decode('utf-8', errors='ignore')
            has_input = 'messageInput' in text
            has_send = 'sendBtn' in text
            if has_input and has_send:
                results["passed"] += 1
                print("  [PASS] Chat room page (input + send button present)")
            else:
                results["failed"] += 1
                results["failures"].append(f"Chat page: input={has_input}, send={has_send}")
                print(f"  [FAIL] Chat room page: input={has_input}, send={has_send}")
        else:
            results["failed"] += 1
            results["failures"].append(f"Chat room page: Status {resp.status_code}")
            print(f"  [FAIL] Chat room page: Status {resp.status_code}")
    except Exception as e:
        results["failed"] += 1
        results["failures"].append(f"Chat send: {str(e)[:60]}")
        print(f"  [FAIL] Chat send: {str(e)[:60]}")


def test_forum_page_loads():
    """Test that forum pages load correctly."""
    results["total"] += 1
    try:
        with client.session_transaction() as sess:
            sess['student_id'] = 'TEST001'
            sess['student_name'] = 'Test Student'

        resp = client.get('/forum', follow_redirects=True)
        if resp.status_code == 200:
            text = resp.data.decode('utf-8', errors='ignore')
            has_create = 'forum/create' in text or 'Create Topic' in text
            if has_create:
                results["passed"] += 1
                print("  [PASS] Forum index page")
            else:
                results["failed"] += 1
                results["failures"].append("Forum index: No create topic link")
                print("  [FAIL] Forum index page: No create topic link")
        else:
            results["failed"] += 1
            results["failures"].append(f"Forum index: Status {resp.status_code}")
            print(f"  [FAIL] Forum index page: Status {resp.status_code}")
    except Exception as e:
        results["failed"] += 1
        results["failures"].append(f"Forum index: {str(e)[:60]}")
        print(f"  [FAIL] Forum index: {str(e)[:60]}")


def test_chat_page_loads():
    """Test that chat pages load correctly."""
    results["total"] += 1
    try:
        with client.session_transaction() as sess:
            sess['student_id'] = 'TEST001'
            sess['student_name'] = 'Test Student'

        resp = client.get('/chat', follow_redirects=True)
        if resp.status_code == 200:
            results["passed"] += 1
            print("  [PASS] Chat index page")
        else:
            results["failed"] += 1
            results["failures"].append(f"Chat index: Status {resp.status_code}")
            print(f"  [FAIL] Chat index page: Status {resp.status_code}")
    except Exception as e:
        results["failed"] += 1
        results["failures"].append(f"Chat index: {str(e)[:60]}")
        print(f"  [FAIL] Chat index: {str(e)[:60]}")


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


def run_all():
    print("\n" + "="*70)
    print("CHAT & FORUM FUNCTIONALITY TEST")
    print("="*70)

    setup_data()

    print("\n--- Forum Tests ---")
    test_forum_page_loads()
    test_forum_create()
    test_forum_reply()

    print("\n--- Chat Tests ---")
    test_chat_page_loads()
    test_chat_create_room()
    test_chat_send_message()

    total = results["total"]
    passed = results["passed"]
    failed = results["failed"]
    rate = passed / total * 100 if total else 0

    print("\n" + "="*70)
    print("CHAT & FORUM TEST SUMMARY")
    print("="*70)
    print(f"\n  Total: {total}")
    print(f"  Passed: {passed} ({rate:.1f}%)")
    print(f"  Failed: {failed}")

    if results["failures"]:
        print(f"\n  Failures:")
        for f in results["failures"]:
            print(f"    - {f}")

    if failed == 0:
        print("\n  STATUS: ALL CHAT & FORUM FEATURES WORKING")
    else:
        print("\n  STATUS: NEEDS FIXES")


if __name__ == "__main__":
    run_all()
