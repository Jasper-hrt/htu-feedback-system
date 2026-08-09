import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import os
from database import db, Student, Feedback, SRCUser, SystemLog, Announcement, FeedbackVote, PasswordResetToken
from database import ForumTopic, ForumReply, ForumTopicVote, ForumTopicTag
from database import ChatRoom, ChatMessage, ChatRoomMember, ChatRoomSentiment
from database import is_valid_htu_email, extract_student_id_from_email
from sentiment_analyzer import process_feedback, analyze_chat_message, analyze_topic, get_room_sentiment_summary, get_forum_sentiment_summary, censor_text, get_sentiment_explanation, get_urgency_explanation
from solution_recommender import recommend_solutions
from logger import log_student_action, log_admin_action, log_feedback_action, log_system_action

from werkzeug.security import generate_password_hash, check_password_hash
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from datetime import datetime, timedelta
import secrets
from io import BytesIO
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from markupsafe import Markup

# ==================== EMOTION EMOJI MAPPING ====================

EMOTION_EMOJI_MAP = {
    'anger': '😠', 'fear': '😨', 'sadness': '😢', 'joy': '😊',
    'trust': '🤝', 'disgust': '🤢', 'surprise': '😲',
    'anticipation': '🤔', 'optimism': '🌟', 'love': '❤️',
    'gratitude': '🙏', 'confusion': '😕', 'neutral': '😐',
}

EMOTION_EMOJI_MAP_COMPOUND = {
    'frustrated_resignation': '😤', 'threatened_defensive': '😰',
    'distressed_anxious': '😟', 'admiring_appreciative': '🥰',
    'delighted_surprised': '🤩', 'revolted_indignant': '🤬',
    'disheartened': '😞', 'alarmed': '😱',
    'agitated': '😤', 'anxious': '😰', 'somber': '😔',
    'cheerful': '😊', 'confident': '💪', 'repulsed': '🤢',
    'astonished': '😲', 'expectant': '🤔', 'hopeful': '🌟',
    'affectionate': '🥰', 'thankful': '🙏', 'perplexed': '😕',
}

def get_emotion_emoji(emotion_name, compound_mood=None):
    """Get emoji for an emotion, falling back to compound mood emoji."""
    if emotion_name in EMOTION_EMOJI_MAP:
        return EMOTION_EMOJI_MAP[emotion_name]
    if compound_mood and compound_mood in EMOTION_EMOJI_MAP_COMPOUND:
        return EMOTION_EMOJI_MAP_COMPOUND[compound_mood]
    return '😐'


def _parse_json_field(value, default):
    """Safely parse a JSON-encoded DB column, returning default on failure."""
    if not value:
        return default
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return default



app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or secrets.token_urlsafe(32)
if not os.environ.get('SECRET_KEY'):
    app.logger.warning('Using an auto-generated SECRET_KEY. Set the SECRET_KEY environment variable in production to preserve sessions and protect cookies.')
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)

# Cache-busting version for static assets (CSS/JS). Changes on every process
# start (i.e. every deploy), forcing browsers/mobile devices to fetch fresh
# files instead of serving a stale cached copy.
import time as _time
ASSET_VERSION = str(int(_time.time()))

@app.context_processor
def inject_asset_version():
    return {'asset_version': ASSET_VERSION}


def generate_csrf_token():
    token = session.get('_csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['_csrf_token'] = token
    return token


@app.context_processor
def inject_csrf_helpers():
    return {
        'csrf_token': generate_csrf_token,
        'csrf_field': lambda: Markup(f'<input type="hidden" name="csrf_token" value="{generate_csrf_token()}">')
    }


def _csrf_exempt_path():
    return request.path.startswith('/socket.io') or request.endpoint == 'static'


@app.before_request
def ensure_csrf_token():
    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_urlsafe(32)


@app.before_request
def validate_csrf_token():
    if request.method not in ('POST', 'PUT', 'PATCH', 'DELETE'):
        return

    if _csrf_exempt_path():
        return

    token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token')
    if not token or token != session.get('_csrf_token'):
        if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'error': 'Invalid or missing CSRF token'}), 400
        return render_template('error.html', error='Invalid CSRF token. Please refresh the page and try again.'), 400

# On Render, DATABASE_URL / SQLALCHEMY_DATABASE_URI point to PostgreSQL.
# Locally (no env var), fall back to the existing SQLite file.
_DATABASE_URL = os.environ.get('DATABASE_URL') or os.environ.get('SQLALCHEMY_DATABASE_URI') or 'sqlite:///feedback.db'
if _DATABASE_URL.startswith('postgres://'):
    _DATABASE_URL = _DATABASE_URL.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = _DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
IS_SQLITE = _DATABASE_URL.startswith('sqlite')


socketio = SocketIO(app, cors_allowed_origins="*")
limiter = Limiter(get_remote_address, app=app, default_limits=["200 per day", "50 per hour"])

db.init_app(app)

# ==================== RESPONSE TEMPLATES ====================

RESPONSE_TEMPLATES = {
    'Wi-Fi Issues': "Thank you for reporting the Wi-Fi issue. Our ICT team has been notified and will investigate. Please expect an update within 48 hours.",
    'Maintenance': "Thank you for your report. Maintenance has been scheduled and will be addressed within 3 working days.",
    'Staff Conduct': "Thank you for bringing this to our attention. This matter has been escalated to the appropriate department for review.",
    'Safety Concern': "⚠️ This is a priority issue. Security has been notified and additional measures will be implemented.",
    'Accommodation': "Thank you for your feedback about accommodation. This has been forwarded to the Hall management.",
    'Catering': "Thank you for your feedback about catering. We will address this with the food services provider.",
    'General Acknowledgment': "Thank you for your feedback. We have received it and will review it with the relevant committee.",
    'Resolved Confirmation': "✅ We are pleased to inform you that this issue has been resolved. Thank you for your patience."
}

# ==================== SECURITY HEADERS ====================

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# ==================== GLOBAL ERROR HANDLER ====================

@app.errorhandler(500)
def internal_server_error(e):
    """Log the full traceback and return a user-friendly error page.

    Without this handler, Flask returns a bare 'Internal Server Error' page
    and the real cause is hidden. This handler prints the full traceback to
    the server logs (helpful on Render) and shows a clean message to the user.
    """
    import traceback
    traceback.print_exc()
    log_system_action('Error', '500', f'Internal Server Error: {e}', 'ERROR')
    try:
        return render_template('error.html', error='Sorry, something went wrong on our end. Please try again in a moment.'), 500
    except Exception:
        return 'Internal Server Error', 500

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error='The page you are looking for was not found.'), 404

# ==================== SESSION TIMEOUT ====================

@app.before_request
def make_session_permanent():
    session.permanent = True

@app.before_request
def check_session_timeout():
    if 'student_id' in session:
        last_activity = session.get('last_activity')
        if last_activity:
            last = datetime.fromisoformat(last_activity)
            if datetime.utcnow() - last > timedelta(hours=2):
                session.clear()
                return redirect(url_for('student_login'))
        session['last_activity'] = datetime.utcnow().isoformat()

# ==================== CREATE TABLES / SCHEMA REPAIR ====================

def _repair_chat_messages_schema_if_needed():
    """Repairs legacy SQLite schema mismatch for chat_messages.

    Old schema (found in some DBs):
      - topic_id (instead of room_id)
      - message_text (instead of message)
      - anonymous (extra)

    Current app schema expects:
      - room_id, message, cleaned_message, is_flagged
    """
    import sqlite3
    from pathlib import Path

    db_path = Path(app.root_path) / 'instance' / 'feedback.db'
    # Fallback to repo-relative path used by your config: sqlite:///feedback.db
    if not db_path.exists():
        db_path = Path('instance') / 'feedback.db'

    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(chat_messages)")
    cols = [r[1] for r in cur.fetchall()]

    # Already correct
    if 'room_id' in cols and 'message' in cols:
        conn.close()
        return

    # Only attempt repair when we recognize the legacy columns
    if 'topic_id' not in cols or 'message_text' not in cols:
        conn.close()
        return

    # Create new table with expected schema
    cur.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages_new (
            id INTEGER PRIMARY KEY,
            room_id INTEGER NOT NULL,
            student_id VARCHAR(20) NOT NULL,
            message TEXT NOT NULL,
            cleaned_message TEXT,
            sentiment VARCHAR(10),
            sentiment_score FLOAT,
            urgency_score INTEGER DEFAULT 1,
            is_flagged BOOLEAN DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Migrate legacy data
    cur.execute('''
        INSERT OR REPLACE INTO chat_messages_new (
            id, room_id, student_id, message, cleaned_message, sentiment, sentiment_score, urgency_score, is_flagged, created_at
        )
        SELECT
            id,
            topic_id AS room_id,
            student_id,
            message_text AS message,
            NULL AS cleaned_message,
            sentiment,
            sentiment_score,
            urgency_score,
            0 AS is_flagged,
            created_at
        FROM chat_messages
    ''')

    # Swap tables
    cur.execute('DROP TABLE chat_messages')
    cur.execute('ALTER TABLE chat_messages_new RENAME TO chat_messages')

    conn.commit()
    conn.close()


def _repair_feedback_schema_if_needed():
    """Repair legacy feedback schema missing solution-recommender columns.

    Some existing feedback.db files may have older 'feedback' tables without:
      - recommended_keywords
      - short_term_solution
      - long_term_solution
      - responsible_department
      - estimated_time

    We add these columns when absent to prevent runtime 500s on /public.
    """
    import sqlite3
    from pathlib import Path

    # Match the DB location used by SQLALCHEMY_DATABASE_URI (sqlite:///feedback.db)
    db_path = Path(app.root_path) / 'instance' / 'feedback.db'
    if not db_path.exists():
        db_path = Path('instance') / 'feedback.db'
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(feedback)")
    cols = [r[1] for r in cur.fetchall()]
    cur.close()

    # Only attempt repair if feedback table exists (or is being created).
    # If it doesn't exist yet, db.create_all() will handle it.
    if not cols:
        conn.close()
        return

    desired_columns = {
        'recommended_keywords': 'TEXT',
        'short_term_solution': 'TEXT',
        'long_term_solution': 'TEXT',
        'responsible_department': 'TEXT',
        'estimated_time': 'TEXT',
        'confidence_score': 'FLOAT',
        'dominant_emotion': 'VARCHAR(50)',
        'compound_mood': 'VARCHAR(50)',
        'emotion_intensities': 'TEXT',
        'secondary_emotions': 'TEXT',
    }

    added_any = False
    for col, col_type in desired_columns.items():
        if col not in cols:
            conn.execute(f"ALTER TABLE feedback ADD COLUMN {col} {col_type}")
            added_any = True

    conn.commit()
    conn.close()


# ==================== NLTK DATA SAFETY NET ====================

def _ensure_schema_aligned():
    """Ensure every model table has all columns required by its SQLAlchemy model.

    db.create_all() only creates MISSING tables; it does NOT add new columns to
    EXISTING tables. On the deployed PostgreSQL database (Render), tables may
    have been originally created by an older release and be missing newer columns
    (e.g. the solution-recommender and confidence/emotion columns on `feedback`).
    Without them, INSERT/UPDATE statements fail with a 500 Internal Server Error.

This function is dialect-agnostic (works on both SQLite and PostgreSQL):
      - Uses SQLAlchemy's inspector to read existing columns per table.
      - Adds any missing columns via ALTER TABLE ... ADD COLUMN.

    IMPORTANT: PostgreSQL does NOT allow ALTER TABLE ... ADD COLUMN with a
    NOT NULL constraint unless a DEFAULT is provided. All our model columns
    that are nullable=False already have a client-side Python default, but the
    compiled DDL may still include NOT NULL. To be safe for the deployment
    migration, we add columns as NULLABLE (omitting NOT NULL) here. The
    application always supplies values on INSERT, so NULLs are never observed
    in practice, and this avoids PostgreSQL migration failures.
    """
    import traceback
    from sqlalchemy import inspect as sa_inspect, text as sa_text

    insp = sa_inspect(db.engine)
    existing_tables = set(insp.get_table_names())

    added_any = False
    try:
        for table in db.metadata.sorted_tables:
            table_name = table.name
            if table_name not in existing_tables:
                continue  # db.create_all() will create it.

            existing_cols = {col['name'] for col in insp.get_columns(table_name)}

            for col in table.columns:
                if col.name in existing_cols:
                    continue

                # Build the base portable SQL type string for the ALTER statement.
                col_type = col.type
                try:
                    compiled_type = col_type.compile(dialect=db.engine.dialect)
                except Exception:
                    compiled_type = str(col_type)

                # Strip any trailing NOT NULL / UNIQUE from the compiled type so
                # PostgreSQL does not reject adding a column without a default.
                cleaned_type = compiled_type.split(" NOT NULL")[0].strip()
                cleaned_type = cleaned_type.split(" UNIQUE")[0].strip()

                sql = f'ALTER TABLE {table_name} ADD COLUMN {col.name} {cleaned_type}'
                try:
                    db.session.execute(sa_text(sql))
                    added_any = True
                    print(f"[Schema] Added column '{table_name}.{col.name}' ({cleaned_type})")
                except Exception as e:
                    # Column may have been added concurrently / already exists.
                    print(f"[Schema] Could not add column '{table_name}.{col.name}': {e}")
                    db.session.rollback()
        if added_any:
            db.session.commit()
            print("[Schema] Added missing columns to the database.")
    except Exception as e:
        db.session.rollback()
        print(f"[Schema] ERROR during schema alignment: {e}")
        traceback.print_exc()


def _ensure_nltk_data():
    """Ensure required NLTK corpora are available at runtime.

    If the build step (download_nltk_data.py) was skipped or failed,
    this provides a fallback download at first app startup.

    To avoid blocking every worker start with a slow/blocking network
    download (which on Render can cause worker timeouts / 500s), we:
      - Search *all* NLTK data paths (not just the first), because the
        actual corpus may live in a non-default path (e.g. %APPDATA%/nltk_data).
      - Apply a short socket timeout so offline/slow startups fail fast
        instead of hanging the whole app for minutes.
      - Only attempt a download for resources that are genuinely missing,
        and never retry a resource that we already tried (cached in a
        module-level set). Each resource is checked once and cached so this
        function is cheap on subsequent calls.
    """
    import nltk

    nltk_resources = [
        'wordnet', 'punkt', 'averaged_perceptron_tagger', 'omw-1.4',
        'sentiwordnet',
    ]

    # Cache the availability result so we don't repeatedly attempt network
    # I/O on every startup / every request (which caused 13s+ delays and
    # gunicorn worker timeouts on Render).
    if getattr(_ensure_nltk_data, '_checked', False):
        return
    _ensure_nltk_data._checked = True

    # Fail-fast: apply a short network timeout for any NLTK download attempt.
    import socket
    _previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(5)

    # NLTK places resources under different subdirectories depending on type:
    #   - corpora:  wordnet, sentiwordnet, omw-1.4, ...
    #   - tokenizers: punkt
    #   - taggers:  averaged_perceptron_tagger
    _NLTK_SUBDIRS = ['corpora', 'tokenizers', 'taggers']

    def _resource_available(resource):
        """Check whether a resource exists in ANY configured NLTK data path.

        NLTK groups resources by type (corpora/tokenizers/taggers), so we
        search each possible subdirectory for both unzipped and zipped forms.
        """
        from pathlib import Path
        for base in nltk.data.path:
            for sub in _NLTK_SUBDIRS:
                candidate = Path(base) / sub / resource
                if candidate.exists():
                    return True
                # Support zipped resources (resource.zip)
                if candidate.with_suffix('.zip').exists():
                    return True
        return False

    for resource in nltk_resources:
        try:
            if _resource_available(resource):
                continue
        except Exception:
            # Fall back to nltk's own lookup; if it fails we'll try download.
            try:
                nltk.data.find(f'corpora/{resource}')
                continue
            except LookupError:
                pass

        # Resource is genuinely missing. Attempt a single download (with the
        # short timeout). Never raise — a missing corpus should degrade the
        # sentiment pipeline gracefully, not crash the app.
        try:
            nltk.download(resource, quiet=True)
            print(f"[NLTK] Downloaded missing resource: {resource}")
        except Exception as e:
            print(f"[NLTK] WARNING: Could not download {resource}: {e}")

    # Restore the previous default socket timeout.
    if _previous_timeout is not None:
        socket.setdefaulttimeout(_previous_timeout)


with app.app_context():
    # Run NLTK data downloads in a background daemon thread so they never
    # block (or crash) web-worker startup. The sentiment pipeline already
    # degrades gracefully when NLTK corpora are unavailable, so a slow or
    # unreachable NLTK server must not prevent the app from booting.
    import threading as _threading
    _nltk_thread = _threading.Thread(target=_ensure_nltk_data, daemon=True)
    _nltk_thread.start()

    # SQLite-only legacy schema repairs (skip on PostgreSQL/Render).
    if IS_SQLITE:
        _repair_chat_messages_schema_if_needed()
        _repair_feedback_schema_if_needed()
    db.create_all()
    # Dialect-agnostic: ensure every model table has all columns the model
    # expects. Critical on PostgreSQL (Render) where db.create_all() does NOT
    # add missing columns to an existing table.
    _ensure_schema_aligned()
    log_system_action('Database', 'Initialization', 'Database tables created')
    
    if not SRCUser.query.filter_by(username='admin').first():
        admin = SRCUser(
            username='admin',
            password_hash=generate_password_hash('admin123'),
            full_name='SRC Administrator',
            role='President'
        )
        db.session.add(admin)
        db.session.commit()
        log_system_action('Admin', 'Creation', 'Default admin account created')


def add_db_log(log_type, level, user_type, user_id, action, details):
    try:
        log = SystemLog(
            log_type=log_type, level=level, user_type=user_type,
            user_id=str(user_id)[:50], action=action[:100], details=details[:500],
            ip_address=request.remote_addr if request else '0.0.0.0'
        )
        db.session.add(log)
        db.session.commit()
    except Exception as e:
        print(f"Failed to add DB log: {e}")

# ==================== AUTHENTICATION MIDDLEWARE ====================

def student_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'student_id' not in session:
            return redirect(url_for('student_login'))
        return f(*args, **kwargs)
    return decorated_function

def src_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_id' not in session:
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

# ==================== STUDENT AUTHENTICATION ====================

@app.route('/student/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")

def student_login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password')
        
        if not is_valid_htu_email(email):
            return render_template('student_login.html', error='Invalid email format. Use: studentid@htu.edu.gh')
        
        student_id = extract_student_id_from_email(email)
        student = Student.query.filter_by(student_id=student_id).first()
        
        if student and check_password_hash(student.password_hash, password):
            session['student_id'] = student.student_id
            session['student_name'] = student.full_name
            session['student_email'] = student.email
            student.last_login = datetime.utcnow()
            db.session.commit()
            
            log_student_action(student.student_id, 'Login Success', 'Logged in')
            add_db_log('auth', 'INFO', 'student', student.student_id, 'Login Success', '')
            return redirect(url_for('student_dashboard'))
        else:
            log_student_action(student_id or 'unknown', 'Login Failed', 'Invalid password', 'WARNING')
            return render_template('student_login.html', error='Invalid email or password')
    
    return render_template('student_login.html')

@app.route('/student/register', methods=['GET', 'POST'])
@limiter.limit("3 per hour")
def student_register():
    if request.method == 'POST':
        student_id = request.form.get('student_id', '').strip()
        email = request.form.get('email', '').strip().lower()
        full_name = request.form.get('full_name', '').strip()
        department = request.form.get('department', '')
        year_of_study = request.form.get('year_of_study', 1, type=int)
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        expected_email = f"{student_id}@htu.edu.gh"
        if email != expected_email:
            return render_template('student_register.html', error=f'Email must be: {expected_email}')
        
        if Student.query.filter_by(student_id=student_id).first():
            return render_template('student_register.html', error='Student ID already registered')
        
        if Student.query.filter_by(email=email).first():
            return render_template('student_register.html', error='Email already registered')
        
        if password != confirm_password:
            return render_template('student_register.html', error='Passwords do not match')
        
        if len(password) < 6:
            return render_template('student_register.html', error='Password must be at least 6 characters')
        
        student = Student(
            student_id=student_id, email=email, password_hash=generate_password_hash(password),
            full_name=full_name, department=department, year_of_study=year_of_study
        )
        db.session.add(student)
        db.session.commit()
        
        log_student_action(student_id, 'Registration Success', f'Student {full_name} registered')
        
        session['student_id'] = student.student_id
        session['student_name'] = student.full_name
        session['student_email'] = student.email
        
        return redirect(url_for('student_dashboard'))
    
    return render_template('student_register.html')

@app.route('/student/forgot-password', methods=['GET', 'POST'])
@limiter.limit("3 per hour")
def forgot_password():
    if request.method == 'POST':
        student_id = request.form.get('student_id', '').strip()
        student = Student.query.filter_by(student_id=student_id).first()
        
        if student:
            token = secrets.token_urlsafe(32)
            expires = datetime.utcnow() + timedelta(hours=1)
            PasswordResetToken.query.filter_by(student_id=student.student_id).delete()
            reset = PasswordResetToken(student_id=student.student_id, token=token, expires_at=expires)
            db.session.add(reset)
            db.session.commit()
            
            log_student_action(student.student_id, 'Password Reset Request', 'Reset token generated')
            return render_template('forgot_password.html', token=token, student_id=student.student_id, message='Your reset token has been generated.')
        
        return render_template('forgot_password.html', error='Student ID not found')
    
    return render_template('forgot_password.html')

@app.route('/student/reset-password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        token = request.form.get('token')
        password = request.form.get('password')
        confirm = request.form.get('confirm_password')
        
        reset = PasswordResetToken.query.filter_by(student_id=student_id, token=token).first()
        
        if not reset or reset.expires_at < datetime.utcnow():
            return render_template('reset_password.html', error='Invalid or expired reset token')
        
        if password != confirm:
            return render_template('reset_password.html', error='Passwords do not match', token=token, student_id=student_id)
        
        if len(password) < 6:
            return render_template('reset_password.html', error='Password must be at least 6 characters', token=token, student_id=student_id)
        
        student = Student.query.filter_by(student_id=student_id).first()
        student.password_hash = generate_password_hash(password)
        db.session.delete(reset)
        db.session.commit()
        
        log_student_action(student.student_id, 'Password Reset', 'Password changed successfully')
        return redirect(url_for('student_login'))
    
    return render_template('reset_password.html')

@app.route('/student/logout', methods=['POST'])
def student_logout():
    student_id = session.get('student_id')
    log_student_action(student_id, 'Logout', 'User logged out')
    session.clear()
    return redirect(url_for('index'))

# ==================== STUDENT DASHBOARD & FEEDBACK ====================

@app.route('/student/dashboard')
@student_required
def student_dashboard():
    student = Student.query.filter_by(student_id=session['student_id']).first()
    my_feedback = Feedback.query.filter_by(student_id=session['student_id']).order_by(Feedback.created_at.desc()).all()

    # Build per-feedback explanations for the UI (no DB schema change).
    # We use cleaned_text when available; fallback to feedback_text.
    for f in my_feedback:
        base_text = (f.cleaned_text or '').strip() or (f.feedback_text or '').strip()
        f.sentiment_explanation = get_sentiment_explanation(base_text).get('sentiment_explanation', [])
        f.urgency_explanation = get_urgency_explanation(base_text, f.sentiment)
        # Attach emotion & confidence display helpers
        f.confidence_display = f.confidence_score
        f.emotion_emoji = get_emotion_emoji(f.dominant_emotion, f.compound_mood)
        f.emotion_intensities_list = _parse_json_field(f.emotion_intensities, {})
        f.secondary_emotions_list = _parse_json_field(f.secondary_emotions, [])

    stats = {

        'total': len(my_feedback),
        'resolved': sum(1 for f in my_feedback if f.status == 'Resolved'),
        'in_progress': sum(1 for f in my_feedback if f.status == 'In Progress'),
        'pending': sum(1 for f in my_feedback if f.status == 'Pending')
    }

    return render_template('student_dashboard.html', feedbacks=my_feedback, stats=stats, student=student)


@app.route('/submit', methods=['GET', 'POST'])
@student_required
@limiter.limit("5 per minute")




def submit_feedback():

    if request.method == 'POST':
        text = request.form.get('feedback_text')
        category = request.form.get('category')
        location = request.form.get('location')
        anonymous = request.form.get('anonymous') == 'on'
        user_urgency = int(request.form.get('user_urgency', 3))
        
        analysis = process_feedback(text, category)
        final_urgency = max(user_urgency, analysis['urgency_score'])
        
        detected_category = analysis['detected_category'] if category == 'Other' else category

        # Enhanced recommendation with sentiment, emotion, and confidence data
        emotion_data = analysis.get('emotion', None)
        sentiment_label = analysis.get('sentiment', None)
        sentiment_val = analysis.get('sentiment_score', None)

        rec = recommend_solutions(
            text=text,
            category=detected_category,
            urgency_score=final_urgency,
            sentiment=sentiment_label,
            sentiment_score=sentiment_val,
            emotion=emotion_data,
        )

        # Persist confidence score and emotion data from hybrid analysis
        emotion_data = analysis.get('emotion', {})
        confidence_val = analysis.get('confidence', None)

        feedback = Feedback(
            student_id=session['student_id'],
            anonymous=anonymous,
            category=detected_category,
            location=location,
            feedback_text=text,
            cleaned_text=analysis['cleaned_text'],
            sentiment=analysis['sentiment'],
            sentiment_score=analysis['sentiment_score'],
            urgency_score=final_urgency,
            has_profanity=analysis['has_profanity'],
            recommended_keywords=','.join(rec.matched_keywords) if rec.matched_keywords else None,
            short_term_solution=rec.short_term_solution,
            long_term_solution=rec.long_term_solution,
            responsible_department=rec.responsible_department,
            estimated_time=rec.estimated_time,
            # New confidence & emotion fields
            confidence_score=confidence_val,
            dominant_emotion=emotion_data.get('dominant_emotion') if emotion_data else None,
            compound_mood=emotion_data.get('compound_mood') if emotion_data else None,
            emotion_intensities=json.dumps(emotion_data.get('emotion_intensities')) if emotion_data and emotion_data.get('emotion_intensities') else None,
            secondary_emotions=json.dumps(emotion_data.get('secondary_emotions')) if emotion_data and emotion_data.get('secondary_emotions') else None,
        )

        db.session.add(feedback)
        db.session.commit()
        
        if analysis['has_profanity']:
            log_feedback_action(feedback.id, session['student_id'], 'Profanity Detected', 
                               'Feedback contained inappropriate language', 'WARNING')
        
        log_feedback_action(feedback.id, session['student_id'], 'Submit', 
                           f'Category: {feedback.category}, Urgency: {final_urgency}')
        
        return render_template('submit_success.html', feedback_id=feedback.id)
    
    return render_template('submit_feedback.html')

@app.route('/edit-feedback/<int:feedback_id>', methods=['GET', 'POST'])
@student_required
def edit_feedback(feedback_id):
    feedback = Feedback.query.get_or_404(feedback_id)
    
    if feedback.student_id != session['student_id']:
        return render_template('error.html', error="Unauthorized"), 403
    
    time_limit = feedback.created_at + timedelta(hours=1)
    if datetime.utcnow() > time_limit:
        return render_template('error.html', error="Feedback can only be edited within 1 hour of submission")
    
    if request.method == 'POST':
        new_text = request.form.get('feedback_text')
        feedback.feedback_text = new_text
        feedback.category = request.form.get('category')
        feedback.location = request.form.get('location')
        feedback.urgency_score = int(request.form.get('user_urgency', 3))
        
        analysis = process_feedback(new_text, feedback.category)
        emotion_data = analysis.get('emotion', {})
        feedback.cleaned_text = analysis['cleaned_text']
        feedback.sentiment = analysis['sentiment']
        feedback.sentiment_score = analysis['sentiment_score']
        feedback.urgency_score = max(feedback.urgency_score, analysis['urgency_score'])
        feedback.has_profanity = analysis['has_profanity']
        feedback.confidence_score = analysis.get('confidence', None)
        feedback.dominant_emotion = emotion_data.get('dominant_emotion') if emotion_data else None
        feedback.compound_mood = emotion_data.get('compound_mood') if emotion_data else None
        feedback.emotion_intensities = json.dumps(emotion_data.get('emotion_intensities')) if emotion_data and emotion_data.get('emotion_intensities') else None
        feedback.secondary_emotions = json.dumps(emotion_data.get('secondary_emotions')) if emotion_data and emotion_data.get('secondary_emotions') else None
        
        db.session.commit()
        log_feedback_action(feedback.id, session['student_id'], 'Edit', 'Feedback edited')
        return redirect(url_for('student_dashboard'))
    
    return render_template('edit_feedback.html', feedback=feedback)

@app.route('/delete-feedback/<int:feedback_id>', methods=['POST'])
@student_required
def delete_feedback(feedback_id):
    feedback = Feedback.query.get_or_404(feedback_id)
    
    if feedback.student_id != session['student_id']:
        return render_template('error.html', error="Unauthorized"), 403
    
    if feedback.status != 'Pending' and datetime.utcnow() > feedback.created_at + timedelta(hours=1):
        return render_template('error.html', error="Feedback can only be deleted if pending or within 1 hour")
    
    db.session.delete(feedback)
    db.session.commit()
    log_feedback_action(feedback_id, session['student_id'], 'Delete', 'Feedback deleted')
    return redirect(url_for('student_dashboard'))

@app.route('/vote/<int:feedback_id>', methods=['POST'])
@student_required
def vote_feedback(feedback_id):
    feedback = Feedback.query.get_or_404(feedback_id)
    
    if feedback.student_id == session['student_id']:
        return jsonify({'error': 'Cannot vote on your own feedback'}), 400
    
    existing = FeedbackVote.query.filter_by(feedback_id=feedback_id, student_id=session['student_id']).first()
    
    if existing:
        db.session.delete(existing)
        action = 'removed'
    else:
        vote = FeedbackVote(feedback_id=feedback_id, student_id=session['student_id'])
        db.session.add(vote)
        action = 'added'
    
    db.session.commit()
    return jsonify({'success': True, 'action': action, 'vote_count': feedback.vote_count})

# ==================== FORUM ROUTES ====================

@app.route('/forum')
@student_required
def forum_index():
    category = request.args.get('category', '')
    sort = request.args.get('sort', 'latest')
    sentiment_filter = request.args.get('sentiment', '')
    
    query = ForumTopic.query
    if category:
        query = query.filter_by(category=category)
    if sentiment_filter:
        query = query.filter_by(sentiment=sentiment_filter)
    
    if sort == 'latest':
        topics = query.order_by(ForumTopic.created_at.desc()).all()
    elif sort == 'hottest':
        topics = query.all()
        topics.sort(key=lambda t: (len(t.replies) + t.urgency_score * 2), reverse=True)
    elif sort == 'most_voted':
        topics = query.all()
        topics.sort(key=lambda t: t.vote_count, reverse=True)
    else:
        topics = query.all()
        topics.sort(key=lambda t: t.vote_count, reverse=True)

    
    pinned_topics = [t for t in topics if t.is_pinned]
    regular_topics = [t for t in topics if not t.is_pinned]
    
    forum_summary = get_forum_sentiment_summary(ForumTopic.query.all())
    
    category_counts = {}
    for t in ForumTopic.query.all():
        category_counts[t.category] = category_counts.get(t.category, 0) + 1
    
    return render_template('forum_index.html', pinned_topics=pinned_topics, topics=regular_topics,
                         forum_summary=forum_summary, category_counts=category_counts,
                         current_category=category, current_sort=sort, current_sentiment=sentiment_filter)

@app.route('/forum/topic/<int:topic_id>', methods=['GET', 'POST'])
@student_required
def view_topic(topic_id):
    topic = ForumTopic.query.get_or_404(topic_id)
    student_id = session['student_id']
    
    topic.view_count += 1
    db.session.commit()
    
    user_vote = ForumTopicVote.query.filter_by(topic_id=topic_id, student_id=student_id).first()
    
    if request.method == 'POST':
        if topic.is_locked:
            return render_template('forum_topic.html', topic=topic, replies=topic.replies, 
                                 user_vote=user_vote, error='This topic is locked.')
        
        content = request.form.get('content')
        if content:
            analysis = analyze_chat_message(content)
            reply = ForumReply(
                topic_id=topic_id, student_id=student_id, content=content,
                cleaned_content=analysis['cleaned_message'], sentiment=analysis['sentiment'],
                sentiment_score=analysis['sentiment_score'], urgency_score=analysis['urgency_score'],
                is_flagged=analysis['is_flagged']
            )
            db.session.add(reply)
            topic.updated_at = datetime.utcnow()
            
            topic_analysis = analyze_topic(topic.content, topic.replies + [reply])
            topic.sentiment = topic_analysis['sentiment']
            topic.sentiment_score = topic_analysis['sentiment_score']
            topic.urgency_score = topic_analysis['urgency_score']
            
            db.session.commit()
            log_student_action(student_id, 'Forum Reply', f'Replied to topic: {topic.title}')
            return redirect(url_for('view_topic', topic_id=topic_id))
    
    replies = ForumReply.query.filter_by(topic_id=topic_id).order_by(ForumReply.created_at.asc()).all()
    return render_template('forum_topic.html', topic=topic, replies=replies, user_vote=user_vote, student_id=student_id)

@app.route('/forum/create', methods=['GET', 'POST'])
@student_required
def create_topic():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        category = request.form.get('category')
        tags = request.form.get('tags', '').split(',')
        
        analysis = analyze_topic(content, [])
        
        topic = ForumTopic(
            title=title, content=content, category=category, student_id=session['student_id'],
            sentiment=analysis['sentiment'], sentiment_score=analysis['sentiment_score'],
            urgency_score=analysis['urgency_score']
        )
        db.session.add(topic)
        db.session.commit()
        
        for tag in tags:
            tag = tag.strip().lower()
            if tag:
                topic_tag = ForumTopicTag(topic_id=topic.id, tag=tag)
                db.session.add(topic_tag)
        
        db.session.commit()
        log_student_action(session['student_id'], 'Create Topic', f'Created topic: {title}')
        return redirect(url_for('view_topic', topic_id=topic.id))
    
    return render_template('create_topic.html')

@app.route('/forum/topic/<int:topic_id>/vote', methods=['POST'])
@student_required
def vote_topic(topic_id):
    vote_type = request.json.get('vote_type')
    if vote_type not in ('up', 'down'):
        return jsonify({'error': 'Invalid vote type'}), 400

    student_id = session['student_id']
    
    existing = ForumTopicVote.query.filter_by(topic_id=topic_id, student_id=student_id).first()
    
    if existing:
        if existing.vote_type == vote_type:
            db.session.delete(existing)
            action = 'removed'
        else:
            existing.vote_type = vote_type
            action = 'changed'
    else:
        vote = ForumTopicVote(topic_id=topic_id, student_id=student_id, vote_type=vote_type)
        db.session.add(vote)
        action = 'added'
    
    db.session.commit()
    
    upvotes = ForumTopicVote.query.filter_by(topic_id=topic_id, vote_type='up').count()
    downvotes = ForumTopicVote.query.filter_by(topic_id=topic_id, vote_type='down').count()
    
    return jsonify({'success': True, 'action': action, 'upvotes': upvotes, 'downvotes': downvotes, 'net_votes': upvotes - downvotes})

@app.route('/forum/reply/<int:reply_id>/edit', methods=['POST'])
@student_required
def edit_reply(reply_id):
    reply = ForumReply.query.get_or_404(reply_id)
    
    if reply.student_id != session['student_id']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    time_diff = datetime.utcnow() - reply.created_at
    if time_diff.total_seconds() > 300:
        return jsonify({'error': 'Replies can only be edited within 5 minutes'}), 400
    
    new_content = request.json.get('content')
    if new_content:
        reply.content = new_content
        reply.is_edited = True
        reply.edited_at = datetime.utcnow()
        
        analysis = analyze_chat_message(new_content)
        reply.cleaned_content = analysis['cleaned_message']
        reply.sentiment = analysis['sentiment']
        reply.sentiment_score = analysis['sentiment_score']
        reply.urgency_score = analysis['urgency_score']
        reply.is_flagged = analysis['is_flagged']
        
        topic = ForumTopic.query.get(reply.topic_id)
        topic_analysis = analyze_topic(topic.content, topic.replies)
        topic.sentiment = topic_analysis['sentiment']
        topic.sentiment_score = topic_analysis['sentiment_score']
        topic.urgency_score = topic_analysis['urgency_score']
        
        db.session.commit()
        return jsonify({'success': True})
    
    return jsonify({'error': 'No content provided'}), 400

@app.route('/forum/topic/<int:topic_id>/pin', methods=['POST'])
@src_required
def pin_topic(topic_id):
    topic = ForumTopic.query.get_or_404(topic_id)
    topic.is_pinned = not topic.is_pinned
    db.session.commit()
    log_admin_action(session['admin_name'], 'Pin Topic', f'Pinned: {topic.title}')
    return jsonify({'success': True, 'is_pinned': topic.is_pinned})

@app.route('/forum/topic/<int:topic_id>/lock', methods=['POST'])
@src_required
def lock_topic(topic_id):
    topic = ForumTopic.query.get_or_404(topic_id)
    topic.is_locked = not topic.is_locked
    db.session.commit()
    log_admin_action(session['admin_name'], 'Lock Topic', f'Locked: {topic.title}')
    return jsonify({'success': True, 'is_locked': topic.is_locked})

@app.route('/forum/topic/<int:topic_id>/delete', methods=['POST'])
@src_required
def delete_topic(topic_id):
    topic = ForumTopic.query.get_or_404(topic_id)
    ForumReply.query.filter_by(topic_id=topic_id).delete()
    ForumTopicVote.query.filter_by(topic_id=topic_id).delete()
    ForumTopicTag.query.filter_by(topic_id=topic_id).delete()
    db.session.delete(topic)
    db.session.commit()
    log_admin_action(session['admin_name'], 'Delete Topic', f'Deleted topic ID: {topic_id}')
    return jsonify({'success': True})

@app.route('/forum/search')
@student_required
def search_topics():
    query = request.args.get('q', '')
    if not query:
        return redirect(url_for('forum_index'))
    
    topics = ForumTopic.query.filter(
        ForumTopic.title.contains(query) | ForumTopic.content.contains(query)
    ).order_by(ForumTopic.created_at.desc()).all()
    
    return render_template('forum_search.html', topics=topics, query=query)

@app.route('/forum/my-topics')
@student_required
def my_topics():
    topics = ForumTopic.query.filter_by(student_id=session['student_id']).order_by(ForumTopic.created_at.desc()).all()
    return render_template('forum_my_topics.html', topics=topics)

# ==================== CHAT ROUTES ====================

@app.route('/chat')
@student_required
def chat_index():
    student_id = session['student_id']
    all_rooms = ChatRoom.query.filter_by(is_active=True).order_by(ChatRoom.last_activity.desc()).all()
    my_room_ids = [m.room_id for m in ChatRoomMember.query.filter_by(student_id=student_id).all()]
    
    rooms_with_info = []
    for room in all_rooms:
        sentiment_summary = get_room_sentiment_summary(room.messages)
        rooms_with_info.append({
            'room': room,
            'is_member': room.id in my_room_ids,
            'sentiment': sentiment_summary,
            'member_count': len(room.members),
            'message_count': len(room.messages)
        })
    
    return render_template('chat_index.html', rooms=rooms_with_info)

@app.route('/chat/room/<int:room_id>')
@student_required
def chat_room(room_id):
    room = ChatRoom.query.get_or_404(room_id)
    student_id = session['student_id']
    
    existing = ChatRoomMember.query.filter_by(room_id=room_id, student_id=student_id).first()
    if not existing:
        member = ChatRoomMember(room_id=room_id, student_id=student_id)
        db.session.add(member)
        db.session.commit()
    
    messages = ChatMessage.query.filter_by(room_id=room_id).order_by(ChatMessage.created_at.asc()).limit(100).all()
    sentiment_summary = get_room_sentiment_summary(room.messages)
    
    return render_template('chat_room.html', room=room, messages=messages, sentiment_summary=sentiment_summary, student_id=student_id)

@app.route('/chat/create', methods=['GET', 'POST'])
@student_required
def create_chat_room():
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        category = request.form.get('category')
        
        room = ChatRoom(name=name, description=description, category=category, created_by=session['student_id'])
        db.session.add(room)
        db.session.commit()
        
        member = ChatRoomMember(room_id=room.id, student_id=session['student_id'])
        db.session.add(member)
        db.session.commit()
        
        log_student_action(session['student_id'], 'Create Chat Room', f'Created room: {name}')
        return redirect(url_for('chat_room', room_id=room.id))
    
    return render_template('create_chat_room.html')

# ==================== SOCKET.IO EVENTS ====================

@socketio.on('join')
def handle_join(data):
    room_id = data['room_id']
    username = data.get('username', session.get('student_name', 'Student'))
    join_room(str(room_id))
    emit('user_joined', {'username': username, 'timestamp': datetime.now().strftime('%H:%M')}, room=str(room_id))

@socketio.on('leave')
def handle_leave(data):
    room_id = data['room_id']
    username = data.get('username', session.get('student_name', 'Student'))
    leave_room(str(room_id))
    emit('user_left', {'username': username, 'timestamp': datetime.now().strftime('%H:%M')}, room=str(room_id))

@socketio.on('send_message')
def handle_send_message(data):
    room_id = data['room_id']
    message_text = data['message']
    student_id = session.get('student_id')
    
    if not student_id:
        emit('error', {'message': 'Not authenticated'})
        return

    member = ChatRoomMember.query.filter_by(room_id=room_id, student_id=student_id).first()
    if not member:
        emit('error', {'message': 'You must be a member of this room to send messages'})
        return
    
    analysis = analyze_chat_message(message_text)
    
    chat_message = ChatMessage(
        room_id=room_id, student_id=student_id, message=message_text,
        cleaned_message=analysis['cleaned_message'], sentiment=analysis['sentiment'],
        sentiment_score=analysis['sentiment_score'], urgency_score=analysis['urgency_score'],
        is_flagged=analysis['is_flagged']
    )
    db.session.add(chat_message)
    
    room = ChatRoom.query.get(room_id)
    if room:
        room.last_activity = datetime.utcnow()
    
    db.session.commit()
    
    student = Student.query.get(student_id)
    student_name = student.full_name.split()[0] if student else "Student"

    payload = {
        'id': chat_message.id,
        'room_id': room_id,
        'room_name': room.name if room else None,
        'message': message_text,
        'username': student_name,
        'sentiment': analysis['sentiment'],
        'sentiment_score': analysis['sentiment_score'],
        'urgency_score': analysis['urgency_score'],
        'is_flagged': analysis['is_flagged'],
        'timestamp': chat_message.created_at.strftime('%Y-%m-%d %H:%M')
    }

    # Broadcast the new message to the room participants.
    emit('new_message', payload, room=str(room_id))

    # Also stream it live to the admin chat messages page.
    emit('admin_new_message', payload, room='admins')


    # Update daily sentiment stats
    today = datetime.utcnow().date()
    start_of_day = datetime(today.year, today.month, today.day)
    end_of_day = start_of_day + timedelta(days=1)
    
    messages_today = ChatMessage.query.filter(
        ChatMessage.room_id == room_id,
        ChatMessage.created_at >= start_of_day,
        ChatMessage.created_at < end_of_day
    ).all()
    
    positive = sum(1 for m in messages_today if m.sentiment == 'Positive')
    negative = sum(1 for m in messages_today if m.sentiment == 'Negative')
    neutral = len(messages_today) - positive - negative
    avg_urgency = sum(m.urgency_score for m in messages_today) / len(messages_today) if messages_today else 0
    
    stats = ChatRoomSentiment.query.filter_by(room_id=room_id, date=today).first()
    if stats:
        stats.positive_count = positive
        stats.negative_count = negative
        stats.neutral_count = neutral
        stats.avg_urgency = avg_urgency
        stats.total_messages = len(messages_today)
    else:
        stats = ChatRoomSentiment(
            room_id=room_id, date=today, positive_count=positive,
            negative_count=negative, neutral_count=neutral,
            avg_urgency=avg_urgency, total_messages=len(messages_today)
        )
        db.session.add(stats)
    db.session.commit()

# ==================== ANNOUNCEMENTS ====================


@app.route('/announcements')
def announcements():
    now = datetime.utcnow()
    active_announcements = Announcement.query.filter(
        Announcement.is_active == True,
        (Announcement.expires_at.is_(None) | (Announcement.expires_at > now))
    ).order_by(Announcement.created_at.desc()).all()
    return render_template('announcements.html', announcements=active_announcements)

@app.route('/admin/announcements', methods=['GET', 'POST'])
@src_required
def admin_announcements():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        expires_days = request.form.get('expires_days', type=int)
        
        announcement = Announcement(
            title=title, content=content, author=session['admin_name'],
            expires_at=datetime.utcnow() + timedelta(days=expires_days) if expires_days else None
        )
        db.session.add(announcement)
        db.session.commit()
        
        log_admin_action(session['admin_name'], 'Create Announcement', f'Title: {title}')
        return redirect(url_for('admin_announcements'))
    
    announcements = Announcement.query.order_by(Announcement.created_at.desc()).all()
    return render_template('admin_announcements.html', announcements=announcements, datetime=datetime)

@app.route('/admin/announcements/delete/<int:id>', methods=['POST'])
@src_required
def delete_announcement(id):
    announcement = Announcement.query.get_or_404(id)
    db.session.delete(announcement)
    db.session.commit()
    return redirect(url_for('admin_announcements'))

# ==================== SRC ADMIN ROUTES ====================

@app.route('/admin/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = SRCUser.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session['admin_id'] = user.id
            session['admin_name'] = user.full_name
            session['admin_role'] = user.role
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            log_admin_action(user.full_name, 'Login Success', 'Logged in')
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('admin_login.html', error='Invalid credentials')
    
    return render_template('admin_login.html')

@app.route('/admin/dashboard')
@src_required
def admin_dashboard():
    page = request.args.get('page', 1, type=int)

    per_page = request.args.get('per_page', 20, type=int)
    category_filter = request.args.get('category', '')
    status_filter = request.args.get('status', '')
    urgency_filter = request.args.get('urgency', '')
    search_query = request.args.get('search', '')
    
    query = Feedback.query
    if category_filter:
        query = query.filter_by(category=category_filter)
    if status_filter:
        query = query.filter_by(status=status_filter)
    if urgency_filter:
        query = query.filter(Feedback.urgency_score >= int(urgency_filter))
    if search_query:
        query = query.filter(Feedback.feedback_text.contains(search_query))
    
    paginated = query.order_by(Feedback.urgency_score.desc(), Feedback.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    
    total = Feedback.query.count()
    urgent = Feedback.query.filter(Feedback.urgency_score >= 4).count()
    resolved = Feedback.query.filter_by(status='Resolved').count()
    positive = Feedback.query.filter_by(sentiment='Positive').count()
    negative = Feedback.query.filter_by(sentiment='Negative').count()
    neutral = Feedback.query.filter_by(sentiment='Neutral').count()
    total_students = Student.query.count()
    profanity_count = Feedback.query.filter_by(has_profanity=True).count()

    
    trend_data = []
    for i in range(6):
        month = datetime.utcnow().replace(day=1) - timedelta(days=30*i)
        month_feedbacks = Feedback.query.filter(
            Feedback.created_at >= month,
            Feedback.created_at < month.replace(day=28) + timedelta(days=4)
        ).all()
        trend_data.append({
            'month': month.strftime('%b'),
            'positive': sum(1 for f in month_feedbacks if f.sentiment == 'Positive'),
            'negative': sum(1 for f in month_feedbacks if f.sentiment == 'Negative'),
            'neutral': sum(1 for f in month_feedbacks if f.sentiment == 'Neutral')
        })
    
    categories = {}
    for f in Feedback.query.all():
        categories[f.category] = categories.get(f.category, 0) + 1
    
    stats = {
        'total': total, 'urgent': urgent, 'resolved': resolved,
        'positive': positive, 'negative': negative, 'neutral': neutral,
        'resolution_rate': round((resolved / total * 100) if total > 0 else 0, 1),
        'total_students': total_students, 'profanity_count': profanity_count
    }
    
    # Precompute explanations for Analyze modal.
    # We use cleaned_text when available, fallback to feedback_text.
    explanations_by_id = {}
    recommendations_by_id = {}

    for f in paginated.items:
        base_text = (f.cleaned_text or '').strip() or (f.feedback_text or '').strip()
        sentiment_expl = get_sentiment_explanation(base_text).get('sentiment_explanation', [])
        urgency_expl = get_urgency_explanation(base_text, f.sentiment)

        explanations_by_id[f.id] = {
            'sentiment_explanation': sentiment_expl,
            'urgency_explanation': urgency_expl,
            'confidence_score': f.confidence_score,
            'dominant_emotion': f.dominant_emotion,
            'compound_mood': f.compound_mood,
            'emotion_emoji': get_emotion_emoji(f.dominant_emotion, f.compound_mood),
            'emotion_intensities': _parse_json_field(f.emotion_intensities, {}),
            'secondary_emotions': _parse_json_field(f.secondary_emotions, []),
        }

        recommendations_by_id[f.id] = {
            'recommended_keywords': f.recommended_keywords,
            'short_term_solution': f.short_term_solution,
            'long_term_solution': f.long_term_solution,
            'responsible_department': f.responsible_department,
            'estimated_time': f.estimated_time,
        }

    return render_template('admin_dashboard.html', feedbacks=paginated.items, pagination=paginated,
                         stats=stats, categories=categories, trend_data=trend_data,
                         admin_name=session.get('admin_name'), current_category=category_filter,
                         current_status=status_filter, current_urgency=urgency_filter, current_search=search_query,
                         explanations_by_id=explanations_by_id,
                         recommendations_by_id=recommendations_by_id)



@app.route('/admin/update/<int:feedback_id>', methods=['POST'])
@src_required
def update_feedback(feedback_id):
    feedback = Feedback.query.get_or_404(feedback_id)
    old_status = feedback.status
    new_status = request.form.get('status', feedback.status)
    
    feedback.status = new_status
    feedback.assigned_to = request.form.get('assigned_to', feedback.assigned_to)
    feedback.src_response = request.form.get('src_response', feedback.src_response)
    
    if feedback.status == 'Resolved' and not feedback.resolved_at:
        feedback.resolved_at = datetime.utcnow()
    
    db.session.commit()
    
    log_admin_action(session['admin_name'], 'Feedback Update', f'Feedback ID: {feedback_id}, Status: {old_status} -> {new_status}')
    add_db_log('feedback', 'INFO', 'admin', session['admin_name'], 'Feedback Updated', f'Feedback ID: {feedback_id}, Status: {old_status} -> {new_status}')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/templates', methods=['GET', 'POST'])
@src_required
def admin_templates():
    if request.method == 'POST':
        template_key = request.form.get('template_key')
        template_value = request.form.get('template_value')
        if template_key and template_value:
            RESPONSE_TEMPLATES[template_key] = template_value
            log_admin_action(session['admin_name'], 'Template Update', f'Updated: {template_key}')
    
    return render_template('admin_templates.html', templates=RESPONSE_TEMPLATES)

@app.route('/admin/delete-feedback/<int:feedback_id>', methods=['POST'])
@src_required
def admin_delete_feedback(feedback_id):
    feedback = Feedback.query.get_or_404(feedback_id)
    db.session.delete(feedback)
    db.session.commit()
    log_admin_action(session['admin_name'], 'Delete Feedback', f'Deleted feedback ID: {feedback_id}')
    add_db_log('feedback', 'INFO', 'admin', session['admin_name'], 'Feedback Deleted', f'Feedback ID: {feedback_id}')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/students')
@src_required
def admin_students():
    students = Student.query.order_by(Student.created_at.desc()).all()
    return render_template('admin_students.html', students=students)

@app.route('/admin/logs')
@src_required
def admin_logs():
    log_type = request.args.get('log_type', '')
    level = request.args.get('level', '')
    days = request.args.get('days', 7, type=int)
    page = request.args.get('page', 1, type=int)
    
    query = SystemLog.query
    if log_type:
        query = query.filter_by(log_type=log_type)
    if level:
        query = query.filter_by(level=level)
    if days:
        query = query.filter(SystemLog.timestamp >= datetime.utcnow() - timedelta(days=days))
    
    paginated = query.order_by(SystemLog.timestamp.desc()).paginate(page=page, per_page=25, error_out=False)
    logs = paginated.items
    
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Stats
    total_logs = SystemLog.query.count()
    today_logs = SystemLog.query.filter(SystemLog.timestamp >= today_start).count()
    warning_count = SystemLog.query.filter_by(level='WARNING').count()
    error_count = SystemLog.query.filter_by(level='ERROR').count()
    
    stats = {
        'total': total_logs,
        'today': today_logs,
        'warnings': warning_count,
        'errors': error_count
    }
    
    # Severity distribution for chart
    severity_distribution = {
        'INFO': SystemLog.query.filter_by(level='INFO').count(),
        'WARNING': SystemLog.query.filter_by(level='WARNING').count(),
        'ERROR': SystemLog.query.filter_by(level='ERROR').count()
    }
    
    # Log type distribution
    type_distribution = {}
    for t in ['auth', 'feedback', 'admin', 'system']:
        type_distribution[t] = SystemLog.query.filter_by(log_type=t).count()
    
    # Activity trend (last 7 days)
    log_trend = []
    for i in range(6, -1, -1):
        day = now - timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        day_count = SystemLog.query.filter(
            SystemLog.timestamp >= day_start,
            SystemLog.timestamp < day_end
        ).count()
        log_trend.append({
            'date': day_start.strftime('%a'),
            'count': day_count
        })
    
    return render_template('admin_logs.html', logs=logs, pagination=paginated, stats=stats,
                         current_type=log_type, current_level=level, current_days=days,
                         severity_distribution=severity_distribution,
                         type_distribution=type_distribution,
                         log_trend=log_trend)

@app.route('/admin/export/excel')
@src_required
def export_excel():
    feedbacks = Feedback.query.order_by(Feedback.created_at.desc()).all()
    
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Feedback Data"
    
    headers = ['ID', 'Date', 'Category', 'Original Feedback', 'Cleaned Text', 'Sentiment', 'Urgency', 'Status', 'Assigned To', 'SRC Response', 'Has Profanity']
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1e40af", end_color="1e40af", fill_type="solid")
    
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center')
    
    for row, f in enumerate(feedbacks, 2):
        ws.cell(row=row, column=1, value=f.id)
        ws.cell(row=row, column=2, value=f.created_at.strftime('%Y-%m-%d %H:%M'))
        ws.cell(row=row, column=3, value=f.category)
        ws.cell(row=row, column=4, value=f.feedback_text[:500])
        ws.cell(row=row, column=5, value=f.cleaned_text[:500] if f.cleaned_text else '')
        ws.cell(row=row, column=6, value=f.sentiment)
        ws.cell(row=row, column=7, value=f"{f.urgency_score}/5")
        ws.cell(row=row, column=8, value=f.status)
        ws.cell(row=row, column=9, value=f.assigned_to or '')
        ws.cell(row=row, column=10, value=f.src_response or '')
        ws.cell(row=row, column=11, value='Yes' if f.has_profanity else 'No')
    
    output = BytesIO()
    wb.save(output)
    output.seek(0)
    
    log_admin_action(session['admin_name'], 'Export Excel', f'Exported {len(feedbacks)} feedback records')
    
    return Response(
        output.getvalue(),
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={'Content-Disposition': f'attachment; filename=feedback_export_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.xlsx'}
    )

@app.route('/admin/analytics')
@src_required
def admin_analytics():
    """Advanced analytics dashboard with predictive insights"""
    all_feedback = Feedback.query.all()
    now = datetime.utcnow()
    
    # ==================== Predictive Analytics ====================
    from predictive_analytics import predict_events_combined

    # Get chat messages from the last 7 days for predictions
    seven_days_ago = now - timedelta(days=7)
    recent_chat_messages = ChatMessage.query.filter(ChatMessage.created_at >= seven_days_ago).all()
    
    # Get unresolved feedback older than 48 hours
    unresolved_feedback = Feedback.query.filter(
        Feedback.status != 'Resolved',
        Feedback.created_at <= now - timedelta(hours=48)
    ).all()
    
    prediction_result = predict_events_combined(
        room_messages=recent_chat_messages,
        feedback_items=unresolved_feedback,
        now=now
    )
    
    # ==================== Category Stats ====================
    category_stats = {}
    for f in all_feedback:
        if f.category not in category_stats:
            category_stats[f.category] = {
                'total': 0, 'positive': 0, 'negative': 0, 'neutral': 0,
                'avg_urgency': 0, 'resolution_rate': 0, 'resolved': 0
            }
        stats = category_stats[f.category]
        stats['total'] += 1
        stats[f.sentiment.lower()] += 1
        if f.status == 'Resolved':
            stats['resolved'] += 1
        stats['avg_urgency'] += f.urgency_score
    
    for cat in category_stats:
        s = category_stats[cat]
        if s['total'] > 0:
            s['avg_urgency'] = round(s['avg_urgency'] / s['total'], 1)
            s['resolution_rate'] = round((s['resolved'] / s['total']) * 100, 1)
            s['positive_pct'] = round((s['positive'] / s['total']) * 100, 1)
            s['negative_pct'] = round((s['negative'] / s['total']) * 100, 1)
    
    # ==================== Location Analysis ====================
    location_complaints = {}
    for f in all_feedback:
        if f.location:
            loc = f.location
            location_complaints[loc] = location_complaints.get(loc, 0) + 1
    top_locations = sorted(location_complaints.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # ==================== Time Analysis ====================
    hour_distribution = [0] * 24
    day_distribution = [0] * 7
    for f in all_feedback:
        hour_distribution[f.created_at.hour] += 1
        day_distribution[f.created_at.weekday()] += 1
    
    # ==================== Sentiment Trend (30 days) ====================
    sentiment_trend = []
    for i in range(30, -1, -1):
        day = now - timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        day_feedback = [f for f in all_feedback if day_start <= f.created_at < day_end]
        sentiment_trend.append({
            'date': day_start.strftime('%b %d'),
            'positive': sum(1 for f in day_feedback if f.sentiment == 'Positive'),
            'negative': sum(1 for f in day_feedback if f.sentiment == 'Negative'),
            'neutral': sum(1 for f in day_feedback if f.sentiment == 'Neutral'),
            'total': len(day_feedback)
        })
    
    # ==================== Department Performance ====================
    dept_stats = {}
    for f in all_feedback:
        dept = f.responsible_department or 'Unassigned'
        if dept not in dept_stats:
            dept_stats[dept] = {'total': 0, 'resolved': 0, 'avg_time': 0, 'total_time': 0}
        dept_stats[dept]['total'] += 1
        if f.status == 'Resolved' and f.resolved_at:
            dept_stats[dept]['resolved'] += 1
            time_taken = (f.resolved_at - f.created_at).total_seconds() / 3600
            dept_stats[dept]['total_time'] += time_taken
    
    dept_performance = []
    for dept, s in dept_stats.items():
        dept_performance.append({
            'name': dept,
            'total': s['total'],
            'resolved': s['resolved'],
            'resolution_rate': round((s['resolved'] / s['total']) * 100, 1) if s['total'] > 0 else 0,
            'avg_time_hours': round(s['total_time'] / s['resolved'], 1) if s['resolved'] > 0 else 0
        })
    dept_performance.sort(key=lambda x: x['total'], reverse=True)
    
    # ==================== Overall Metrics ====================
    total = len(all_feedback)
    resolved_count = sum(1 for f in all_feedback if f.status == 'Resolved')
    pending_count = sum(1 for f in all_feedback if f.status == 'Pending')
    in_progress_count = sum(1 for f in all_feedback if f.status == 'In Progress')
    urgent_count = sum(1 for f in all_feedback if f.urgency_score >= 4)
    
    # Resolution time (average in hours)
    resolved_feedback = [f for f in all_feedback if f.status == 'Resolved' and f.resolved_at]
    avg_resolution_time = 0
    if resolved_feedback:
        times = [(f.resolved_at - f.created_at).total_seconds() / 3600 for f in resolved_feedback]
        avg_resolution_time = round(sum(times) / len(times), 1)
    
    return render_template('admin_analytics.html',
                         category_stats=category_stats,
                         top_locations=top_locations,
                         hour_distribution=hour_distribution,
                         day_distribution=day_distribution,
                         sentiment_trend=sentiment_trend,
                         dept_performance=dept_performance,
                         predictions=prediction_result.get('predictions', []),
                         early_warning_level=prediction_result.get('early_warning_level', 'N/A'),
                         max_confidence=prediction_result.get('max_confidence', 0),
                         total=total,
                         resolved_count=resolved_count,
                         pending_count=pending_count,
                         in_progress_count=in_progress_count,
                         urgent_count=urgent_count,
                         avg_resolution_time=avg_resolution_time)

@app.route('/admin/chat/rooms')
@src_required
def admin_chat_rooms():
    rooms = ChatRoom.query.order_by(ChatRoom.created_at.desc()).all()
    room_stats = []
    today = datetime.utcnow().date()
    today_start = datetime(today.year, today.month, today.day)
    total_today_messages = 0
    total_today_flagged = 0
    
    for room in rooms:
        sentiment = get_room_sentiment_summary(room.messages)
        room_today_msgs = ChatMessage.query.filter(
            ChatMessage.room_id == room.id,
            ChatMessage.created_at >= today_start
        ).count()
        room_today_flagged = ChatMessage.query.filter(
            ChatMessage.room_id == room.id,
            ChatMessage.is_flagged == True,
            ChatMessage.created_at >= today_start
        ).count()
        total_today_messages += room_today_msgs
        total_today_flagged += room_today_flagged
        
        # Build sentiment distribution for the mini bar
        pos = sum(1 for m in room.messages if m.sentiment == 'Positive')
        neg = sum(1 for m in room.messages if m.sentiment == 'Negative')
        neu = len(room.messages) - pos - neg
        total = len(room.messages) or 1
        
        room_stats.append({
            'room': room,
            'sentiment': sentiment,
            'member_count': ChatRoomMember.query.filter_by(room_id=room.id).count(),
            'message_count': len(room.messages),
            'today_messages': room_today_msgs,
            'today_flagged': room_today_flagged,
            'pos_pct': round(pos / total * 100),
            'neg_pct': round(neg / total * 100),
            'neu_pct': round(neu / total * 100)
        })
    
    total_rooms = len(rooms)
    total_messages_all = sum(rs['message_count'] for rs in room_stats)
    # Fix: use list comprehension to count flagged messages (avoid .count() with kwargs which doesn't exist on lists)
    total_flagged_all = 0
    for rs in room_stats:
        for msg in rs['room'].messages:
            if msg.is_flagged:
                total_flagged_all += 1
    
    return render_template('admin_chat_rooms.html', rooms=room_stats,
                         total_rooms=total_rooms, total_messages_all=total_messages_all,
                         total_today_messages=total_today_messages,
                         total_today_flagged=total_today_flagged,
                         total_flagged_all=total_flagged_all)

@app.route('/admin/chat/rooms/<int:room_id>/delete', methods=['POST'])
@src_required
def admin_delete_chat_room(room_id):
    room = ChatRoom.query.get_or_404(room_id)
    ChatMessage.query.filter_by(room_id=room_id).delete()
    ChatRoomMember.query.filter_by(room_id=room_id).delete()
    ChatRoomSentiment.query.filter_by(room_id=room_id).delete()
    db.session.delete(room)
    db.session.commit()
    log_admin_action(session['admin_name'], 'Delete Chat Room', f'Deleted room ID: {room_id}')
    return redirect(url_for('admin_chat_rooms'))

@app.route('/admin/chat/messages')
@src_required
def admin_chat_messages():
    flagged_only = request.args.get('flagged', 'false') == 'true'
    query = ChatMessage.query
    if flagged_only:
        query = query.filter_by(is_flagged=True)
    # Eager-load student and room relationships to avoid N+1 queries
    messages = query.options(
        db.joinedload(ChatMessage.student),
        db.joinedload(ChatMessage.room)
    ).order_by(ChatMessage.created_at.desc()).limit(200).all()
    # Convert to dicts for JSON serialization in the template
    messages = [m.to_dict() for m in messages]
    return render_template('admin_chat_messages.html', messages=messages, flagged_only=flagged_only)

@app.route('/admin/logout', methods=['POST'])
@src_required
def admin_logout():
    admin_name = session.get('admin_name')
    log_admin_action(admin_name, 'Logout', 'Admin logged out')
    session.clear()
    return redirect(url_for('index'))

# ==================== PUBLIC ROUTES ====================

@app.route('/public')
def public_board():
    all_feedback = Feedback.query.all()

    sentiment_counts = {
        'positive': sum(1 for f in all_feedback if f.sentiment == 'Positive'),
        'negative': sum(1 for f in all_feedback if f.sentiment == 'Negative'),
        'neutral': sum(1 for f in all_feedback if f.sentiment == 'Neutral')
    }

    category_sentiment = {}
    for f in all_feedback:
        if f.category not in category_sentiment:
            category_sentiment[f.category] = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}
        category_sentiment[f.category][f.sentiment.lower()] += 1
        category_sentiment[f.category]['total'] += 1

    top_complaints = Feedback.query.filter(Feedback.sentiment == 'Negative').order_by(Feedback.urgency_score.desc()).limit(5).all()
    forum_summary = get_forum_sentiment_summary(ForumTopic.query.all())

    # ==================== SRC Performance ====================
    # Engagement increase = compare last 7 days vs previous 7 days.
    # Engagement definition: (Feedback submissions + Forum replies).
    now = datetime.utcnow()
    current_start = (now - timedelta(days=7))
    previous_start = (now - timedelta(days=14))

    current_feedback_count = Feedback.query.filter(Feedback.created_at >= current_start).count()
    previous_feedback_count = Feedback.query.filter(Feedback.created_at >= previous_start, Feedback.created_at < current_start).count()

    current_replies_count = ForumReply.query.filter(ForumReply.created_at >= current_start).count()
    previous_replies_count = ForumReply.query.filter(ForumReply.created_at >= previous_start, ForumReply.created_at < current_start).count()

    current_engagement = current_feedback_count + current_replies_count
    previous_engagement = previous_feedback_count + previous_replies_count

    if previous_engagement > 0:
        engagement_increase_pct = ((current_engagement - previous_engagement) / previous_engagement) * 100
    else:
        # If there was no engagement in the previous period, treat any engagement as "new".
        engagement_increase_pct = 100.0 if current_engagement > 0 else 0.0

    engagement_increase_pct = round(engagement_increase_pct, 1)
    if engagement_increase_pct > 0:
        engagement_arrow = '↑'
    elif engagement_increase_pct < 0:
        engagement_arrow = '↓'
    else:
        engagement_arrow = '→'

    positive_rate = (sentiment_counts['positive'] / len(all_feedback) * 100) if len(all_feedback) > 0 else 0

    return render_template(
        'public_board.html',
        sentiment_counts=sentiment_counts,
        category_sentiment=category_sentiment,
        top_complaints=top_complaints,
        total_feedback=len(all_feedback),
        forum_summary=forum_summary,
        src_positive_rate=round(positive_rate, 0),
        engagement_increase_pct=abs(engagement_increase_pct),
        engagement_arrow=engagement_arrow
    )

@app.route('/')
def index():
    # Live stats for landing page
    all_feedback = Feedback.query.all()
    total_feedback = len(all_feedback)
    total_students = Student.query.count()
    resolved_count = sum(1 for f in all_feedback if f.status == 'Resolved')
    positive_count = sum(1 for f in all_feedback if f.sentiment == 'Positive')
    positive_rate = round((positive_count / total_feedback * 100) if total_feedback > 0 else 0, 0)

    # Recent announcements
    now = datetime.utcnow()
    recent_announcements = Announcement.query.filter(
        Announcement.is_active == True,
        (Announcement.expires_at.is_(None) | (Announcement.expires_at > now))
    ).order_by(Announcement.created_at.desc()).limit(3).all()

    # Engagement stats
    week_ago = now - timedelta(days=7)
    recent_feedback = sum(1 for f in all_feedback if f.created_at >= week_ago)
    recent_replies = ForumReply.query.filter(ForumReply.created_at >= week_ago).count()

    return render_template('index.html',
        total_feedback=total_feedback,
        total_students=total_students,
        resolved_count=resolved_count,
        positive_rate=positive_rate,
        recent_announcements=recent_announcements,
        recent_engagement=recent_feedback + recent_replies
    )

@app.route('/api/set-theme', methods=['POST'])
@student_required
def set_theme():
    theme = request.json.get('theme')
    if theme not in ('light', 'dark'):
        return jsonify({'error': 'Invalid theme selection'}), 400

    student = Student.query.filter_by(student_id=session['student_id']).first()
    if student:
        student.theme_preference = theme
        db.session.commit()
    return jsonify({'success': True})

if __name__ == '__main__':
    log_system_action('System', 'Startup', 'HTU SRC Feedback System started with all features')
    debug_mode = os.environ.get('FLASK_DEBUG', '0').lower() in ('1', 'true', 'yes')
    socketio.run(app, debug=debug_mode)
