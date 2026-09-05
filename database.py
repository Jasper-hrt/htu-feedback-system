from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import re

db = SQLAlchemy()

# ==================== STUDENT MODEL ====================

class Student(db.Model):
    __tablename__ = 'students'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), unique=True, nullable=False, index=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    full_name = db.Column(db.String(100), index=True)
    department = db.Column(db.String(100), index=True)
    year_of_study = db.Column(db.Integer)
    is_active = db.Column(db.Boolean, default=True, index=True)
    theme_preference = db.Column(db.String(10), default='light')
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    last_login = db.Column(db.DateTime, index=True)
    
    __table_args__ = (
        db.Index('idx_student_dept_year', 'department', 'year_of_study'),
        db.Index('idx_student_active_created', 'is_active', 'created_at'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'student_id': self.student_id,
            'email': self.email,
            'full_name': self.full_name,
            'department': self.department,
            'year_of_study': self.year_of_study
        }

# ==================== FEEDBACK MODEL ====================

class Feedback(db.Model):
    __tablename__ = 'feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), db.ForeignKey('students.student_id'), nullable=False)
    anonymous = db.Column(db.Boolean, default=True)
    category = db.Column(db.String(50), nullable=False)
    location = db.Column(db.String(200))
    feedback_text = db.Column(db.Text, nullable=False)
    cleaned_text = db.Column(db.Text)
    sentiment = db.Column(db.String(10))
    sentiment_score = db.Column(db.Float)
    urgency_score = db.Column(db.Integer, default=1)
    status = db.Column(db.String(20), default='Pending')
    assigned_to = db.Column(db.String(100))
    src_response = db.Column(db.Text)
    
    # === Solution Recommendation System (nullable to avoid breaking existing DB rows) ===
    recommended_keywords = db.Column(db.Text)  # comma-separated keywords matched
    short_term_solution = db.Column(db.Text)
    long_term_solution = db.Column(db.Text)
    responsible_department = db.Column(db.String(200))
    estimated_time = db.Column(db.String(100))
    recommendation_confidence = db.Column(db.Float)       # 0-1 match confidence for the recommendation
    secondary_categories = db.Column(db.Text)             # JSON list of other plausible categories
    used_template_id = db.Column(db.Integer, db.ForeignKey('solution_templates.id'), nullable=True)

    # === Enhanced Recommendation System (separate student/admin layers) ===
    student_recommendation_summary = db.Column(db.Text)
    student_recommendation_action = db.Column(db.Text)
    student_recommendation_contact = db.Column(db.String(200))
    student_recommendation_timeline = db.Column(db.String(100))
    admin_action_investigation = db.Column(db.Text)  # JSON list
    admin_action_corrective = db.Column(db.Text)  # JSON list
    admin_action_preventive = db.Column(db.Text)  # JSON list
    admin_action_department = db.Column(db.String(200))
    admin_action_priority = db.Column(db.String(20))
    admin_action_escalation = db.Column(db.String(200))
    recommendation_sentiment_type = db.Column(db.String(20))
    recommendation_urgency_level = db.Column(db.String(20))
    recommendation_fallback_used = db.Column(db.Boolean, default=False)
    recommendation_fallback_message = db.Column(db.Text)
    recommendation_multi_issue = db.Column(db.Boolean, default=False)

    # === Confidence & Emotion Analysis ===
    confidence_score = db.Column(db.Float)                # 0-100 agreement confidence
    dominant_emotion = db.Column(db.String(50))           # e.g. 'anger', 'joy', 'neutral'
    compound_mood = db.Column(db.String(50))              # e.g. 'frustrated_resignation', 'cheerful'
    emotion_intensities = db.Column(db.Text)              # JSON dict {emotion: intensity 0-1}
    secondary_emotions = db.Column(db.Text)               # JSON list of secondary emotions

    # === Active-learning lineage ===
    label_source = db.Column(db.String(20), default='model', index=True)
    # 'model' | 'admin_corrected' | 'memory_override'
    # Set to 'memory_override' when the label was overridden by a similarity-cache hit
    # against a prior admin correction. Set to 'admin_corrected' when the admin directly
    # corrected this row. Used by the UI to render an explanatory badge.

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    resolved_at = db.Column(db.DateTime)
    has_profanity = db.Column(db.Boolean, default=False)

    __table_args__ = (
        db.Index('idx_feedback_status_urgency', 'status', 'urgency_score'),
        db.Index('idx_feedback_category_sentiment', 'category', 'sentiment'),
        db.Index('idx_feedback_student_created', 'student_id', 'created_at'),
        db.Index('idx_feedback_created_category', 'created_at', 'category'),
    )
    
    @property
    def vote_count(self):
        return FeedbackVote.query.filter_by(feedback_id=self.id).count()
    
    def to_dict(self, include_student=False):
        data = {
            'id': self.id,
            'category': self.category,
            'location': self.location,
            'feedback_text': self.feedback_text,
            'sentiment': self.sentiment,
            'sentiment_score': self.sentiment_score,
            'urgency_score': self.urgency_score,
            'status': self.status,
            'assigned_to': self.assigned_to,
            'src_response': self.src_response,
            'short_term_solution': self.short_term_solution,
            'long_term_solution': self.long_term_solution,
            'responsible_department': self.responsible_department,
            'estimated_time': self.estimated_time,
            'recommended_keywords': self.recommended_keywords,
            'recommendation_confidence': self.recommendation_confidence,
            'secondary_categories': self.secondary_categories,
            'confidence_score': self.confidence_score,
            'dominant_emotion': self.dominant_emotion,
            'compound_mood': self.compound_mood,
            'emotion_intensities': self.emotion_intensities,
            'secondary_emotions': self.secondary_emotions,
            'student_recommendation_summary': self.student_recommendation_summary,
            'student_recommendation_action': self.student_recommendation_action,
            'student_recommendation_contact': self.student_recommendation_contact,
            'student_recommendation_timeline': self.student_recommendation_timeline,
            'admin_action_investigation': self.admin_action_investigation,
            'admin_action_corrective': self.admin_action_corrective,
            'admin_action_preventive': self.admin_action_preventive,
            'admin_action_department': self.admin_action_department,
            'admin_action_priority': self.admin_action_priority,
            'admin_action_escalation': self.admin_action_escalation,
            'recommendation_sentiment_type': self.recommendation_sentiment_type,
            'recommendation_urgency_level': self.recommendation_urgency_level,
            'recommendation_fallback_used': self.recommendation_fallback_used,
            'recommendation_fallback_message': self.recommendation_fallback_message,
            'recommendation_multi_issue': self.recommendation_multi_issue,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M'),
            'vote_count': self.vote_count
        }
        if include_student and not self.anonymous:
            data['student_id'] = self.student_id
        return data

class FeedbackVote(db.Model):
    __tablename__ = 'feedback_votes'
    
    id = db.Column(db.Integer, primary_key=True)
    feedback_id = db.Column(db.Integer, db.ForeignKey('feedback.id'), nullable=False)
    student_id = db.Column(db.String(20), db.ForeignKey('students.student_id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (db.UniqueConstraint('feedback_id', 'student_id', name='unique_vote'),)

# ==================== FORUM MODELS ====================

class ForumTopic(db.Model):
    __tablename__ = 'forum_topics'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    student_id = db.Column(db.String(20), db.ForeignKey('students.student_id'), nullable=False)
    is_pinned = db.Column(db.Boolean, default=False)
    is_locked = db.Column(db.Boolean, default=False)
    view_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    sentiment = db.Column(db.String(10))
    sentiment_score = db.Column(db.Float)
    urgency_score = db.Column(db.Integer, default=1)
    
    student = db.relationship('Student', backref='topics')
    replies = db.relationship('ForumReply', backref='topic', lazy=True, cascade='all, delete-orphan')
    tags = db.relationship('ForumTopicTag', backref='topic', lazy=True, cascade='all, delete-orphan')
    
    @property
    def upvotes(self):
        return ForumTopicVote.query.filter_by(topic_id=self.id, vote_type='up').count()

    @property
    def downvotes(self):
        return ForumTopicVote.query.filter_by(topic_id=self.id, vote_type='down').count()

    @property
    def vote_count(self):
        return self.upvotes - self.downvotes
    
    @property
    def is_hot(self):
        return len(self.replies) > 10 or self.urgency_score >= 4

class ForumReply(db.Model):
    __tablename__ = 'forum_replies'
    
    id = db.Column(db.Integer, primary_key=True)
    topic_id = db.Column(db.Integer, db.ForeignKey('forum_topics.id'), nullable=False)
    student_id = db.Column(db.String(20), db.ForeignKey('students.student_id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    cleaned_content = db.Column(db.Text)
    sentiment = db.Column(db.String(10))
    sentiment_score = db.Column(db.Float)
    urgency_score = db.Column(db.Integer, default=1)
    is_flagged = db.Column(db.Boolean, default=False)
    is_edited = db.Column(db.Boolean, default=False)
    reply_to_id = db.Column(db.Integer, db.ForeignKey('forum_replies.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    edited_at = db.Column(db.DateTime)
    
    student = db.relationship('Student', backref='replies')
    reply_to = db.relationship('ForumReply', remote_side=[id], backref='replies_to')

    @property
    def upvotes(self):
        return ForumReplyVote.query.filter_by(reply_id=self.id, vote_type='up').count()

    @property
    def downvotes(self):
        return ForumReplyVote.query.filter_by(reply_id=self.id, vote_type='down').count()

    @property
    def vote_count(self):
        return self.upvotes - self.downvotes

class ForumReplyVote(db.Model):
    __tablename__ = 'forum_reply_votes'

    id = db.Column(db.Integer, primary_key=True)
    reply_id = db.Column(db.Integer, db.ForeignKey('forum_replies.id'), nullable=False)
    student_id = db.Column(db.String(20), db.ForeignKey('students.student_id'), nullable=False)
    vote_type = db.Column(db.String(10))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (db.UniqueConstraint('reply_id', 'student_id', name='unique_reply_vote'),)

class ForumTopicVote(db.Model):
    __tablename__ = 'forum_topic_votes'
    
    id = db.Column(db.Integer, primary_key=True)
    topic_id = db.Column(db.Integer, db.ForeignKey('forum_topics.id'), nullable=False)
    student_id = db.Column(db.String(20), db.ForeignKey('students.student_id'), nullable=False)
    vote_type = db.Column(db.String(10))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (db.UniqueConstraint('topic_id', 'student_id', name='unique_topic_vote'),)

class ForumTopicTag(db.Model):
    __tablename__ = 'forum_topic_tags'
    
    id = db.Column(db.Integer, primary_key=True)
    topic_id = db.Column(db.Integer, db.ForeignKey('forum_topics.id'), nullable=False)
    tag = db.Column(db.String(50), nullable=False)
    
    __table_args__ = (db.UniqueConstraint('topic_id', 'tag', name='unique_topic_tag'),)

# ==================== CHAT MODELS ====================

class ChatRoom(db.Model):
    __tablename__ = 'chat_rooms'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(500))
    category = db.Column(db.String(50))
    is_active = db.Column(db.Boolean, default=True)
    created_by = db.Column(db.String(20), db.ForeignKey('students.student_id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow)
    
    messages = db.relationship('ChatMessage', backref='room', lazy=True, cascade='all, delete-orphan')
    members = db.relationship('ChatRoomMember', backref='room', lazy=True, cascade='all, delete-orphan')

class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    room_id = db.Column(db.Integer, db.ForeignKey('chat_rooms.id'), nullable=False)
    student_id = db.Column(db.String(20), db.ForeignKey('students.student_id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    cleaned_message = db.Column(db.Text)
    sentiment = db.Column(db.String(10))
    sentiment_score = db.Column(db.Float)
    urgency_score = db.Column(db.Integer, default=1)
    is_flagged = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # === Reply-to-message support ===
    reply_to_id = db.Column(db.Integer, db.ForeignKey('chat_messages.id'), nullable=True)

    # === Voice note support ===
    message_type = db.Column(db.String(10), default='text')   # 'text' or 'voice'
    voice_data = db.Column(db.Text)                            # base64-encoded audio data URL
    voice_duration = db.Column(db.Integer)                      # duration in whole seconds

    student = db.relationship('Student', backref='chat_messages')
    reply_to = db.relationship('ChatMessage', remote_side=[id], backref='thread_replies')

    def to_dict(self):
        student_name = self.student.full_name if self.student else 'Unknown Student'
        room_name = self.room.name if self.room else 'Unknown Room'

        reply_to_data = None
        if self.reply_to:
            rt_student_name = self.reply_to.student.full_name if self.reply_to.student else 'Unknown Student'
            reply_to_data = {
                'id': self.reply_to.id,
                'username': rt_student_name.split()[0] if rt_student_name else 'Student',
                'preview': ('🎤 Voice note' if self.reply_to.message_type == 'voice'
                            else (self.reply_to.message[:80] if self.reply_to.message else ''))
            }

        return {
            'id': self.id,
            'room_id': self.room_id,
            'room_name': room_name,
            'student_id': self.student_id,
            'username': student_name.split()[0] if student_name else 'Student',
            'full_name': student_name,
            'message': self.message,
            'cleaned_message': self.cleaned_message,
            'sentiment': self.sentiment,
            'sentiment_score': self.sentiment_score,
            'urgency_score': self.urgency_score,
            'is_flagged': bool(self.is_flagged),
            'message_type': self.message_type or 'text',
            'voice_data': self.voice_data,
            'voice_duration': self.voice_duration,
            'reply_to': reply_to_data,
            'timestamp': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else '',
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else '',
            'student': {
                'full_name': student_name
            } if self.student else None,
            'room': {
                'name': room_name
            } if self.room else None
        }

class ChatRoomMember(db.Model):
    __tablename__ = 'chat_room_members'
    
    id = db.Column(db.Integer, primary_key=True)
    room_id = db.Column(db.Integer, db.ForeignKey('chat_rooms.id'), nullable=False)
    student_id = db.Column(db.String(20), db.ForeignKey('students.student_id'), nullable=False)
    joined_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_read = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (db.UniqueConstraint('room_id', 'student_id', name='unique_member'),)

class ChatRoomSentiment(db.Model):
    __tablename__ = 'chat_room_sentiment'
    
    id = db.Column(db.Integer, primary_key=True)
    room_id = db.Column(db.Integer, db.ForeignKey('chat_rooms.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    positive_count = db.Column(db.Integer, default=0)
    negative_count = db.Column(db.Integer, default=0)
    neutral_count = db.Column(db.Integer, default=0)
    avg_urgency = db.Column(db.Float, default=0)
    total_messages = db.Column(db.Integer, default=0)
    
    __table_args__ = (db.UniqueConstraint('room_id', 'date', name='unique_room_date'),)

# ==================== ANNOUNCEMENT MODEL ====================

class Announcement(db.Model):
    __tablename__ = 'announcements'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    author = db.Column(db.String(100))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)

# ==================== AUTH MODELS ====================

class PasswordResetToken(db.Model):
    __tablename__ = 'password_reset_tokens'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), nullable=False)
    token = db.Column(db.String(100), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)

class SystemLog(db.Model):
    __tablename__ = 'system_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    log_type = db.Column(db.String(20))
    level = db.Column(db.String(10))
    user_type = db.Column(db.String(20))
    user_id = db.Column(db.String(50))
    action = db.Column(db.String(100))
    details = db.Column(db.Text)
    ip_address = db.Column(db.String(45))

class SRCUser(db.Model):
    __tablename__ = 'src_users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(50))
    email = db.Column(db.String(120))
    last_login = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    totp_secret = db.Column(db.String(32), nullable=True)
    is_2fa_enabled = db.Column(db.Boolean, default=False)

# ==================== HYBRID SENTIMENT ENGINE MODELS ====================

class AIReviewLog(db.Model):
    """Audit trail for administrator AI review/correction actions."""
    __tablename__ = 'ai_review_logs'

    id = db.Column(db.Integer, primary_key=True)
    feedback_id = db.Column(db.Integer, db.ForeignKey('feedback.id'), nullable=False, index=True)
    admin_name = db.Column(db.String(100), nullable=False)
    action = db.Column(db.String(50), nullable=False)
    old_sentiment = db.Column(db.String(20))
    new_sentiment = db.Column(db.String(20))
    old_category = db.Column(db.String(50))
    new_category = db.Column(db.String(50))
    old_urgency = db.Column(db.Integer)
    new_urgency = db.Column(db.Integer)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            'id': self.id, 'feedback_id': self.feedback_id,
            'admin_name': self.admin_name, 'action': self.action,
            'old_sentiment': self.old_sentiment, 'new_sentiment': self.new_sentiment,
            'old_category': self.old_category, 'new_category': self.new_category,
            'old_urgency': self.old_urgency, 'new_urgency': self.new_urgency,
            'notes': self.notes,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else ''
        }


class CustomLexicon(db.Model):
    """HTU-specific terms and Ghanaian slang with sentiment scores.
    
    Before VADER analyzes a sentence, any words found here override VADER's default score.
    """
    __tablename__ = 'custom_lexicon'
    
    id = db.Column(db.Integer, primary_key=True)
    word = db.Column(db.String(100), unique=True, nullable=False, index=True)
    sentiment_score = db.Column(db.Float, nullable=False)  # -1.0 to 1.0
    category = db.Column(db.String(50), default='general')  # e.g., 'slang', 'htu_specific', 'academic'
    added_by = db.Column(db.String(100))
    is_active = db.Column(db.Boolean, default=True)
    is_bigram = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'word': self.word,
            'sentiment_score': self.sentiment_score,
            'category': self.category,
            'added_by': self.added_by,
            'is_active': self.is_active,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else '',
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M') if self.updated_at else ''
        }


class UnknownWord(db.Model):
    """Words not found in CustomLexicon, VADER, AFINN, or SentiWordNet.
    
    FastText suggests a score, but admin approval is required before adding to CustomLexicon.
    """
    __tablename__ = 'unknown_words'
    
    id = db.Column(db.Integer, primary_key=True)
    word = db.Column(db.String(100), nullable=False, index=True)
    context = db.Column(db.Text)  # The sentence/context where the word was found
    suggested_score = db.Column(db.Float)  # FastText suggestion
    polarity = db.Column(db.String(20))    # 'positive' | 'negative' | None
    proposed_score = db.Column(db.Float)   # delta-weighted score from active learning
    is_bigram = db.Column(db.Boolean, default=False)
    source = db.Column(db.String(50), default='fasttext')  # 'fasttext', 'manual', 'correction'
    is_reviewed = db.Column(db.Boolean, default=False)
    is_approved = db.Column(db.Boolean, default=False)
    reviewed_by = db.Column(db.String(100))
    reviewed_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'word': self.word,
            'context': self.context,
            'suggested_score': self.suggested_score,
            'source': self.source,
            'is_reviewed': self.is_reviewed,
            'is_approved': self.is_approved,
            'reviewed_by': self.reviewed_by,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else ''
        }


class SentimentFeedbackScore(db.Model):
    """Stores all intermediate sentiment scores and the final decision.
    
    Linked to Feedback, ForumTopic, or ChatMessage via entity_type + entity_id.
    """
    __tablename__ = 'sentiment_feedback_scores'
    
    id = db.Column(db.Integer, primary_key=True)
    entity_type = db.Column(db.String(20), nullable=False)  # 'feedback', 'forum_topic', 'forum_reply', 'chat'
    entity_id = db.Column(db.Integer, nullable=False)
    
    # VADER (primary)
    vader_compound = db.Column(db.Float)
    vader_label = db.Column(db.String(10))
    vader_pos = db.Column(db.Float)
    vader_neu = db.Column(db.Float)
    vader_neg = db.Column(db.Float)
    
    # Custom Lexicon override
    custom_lexicon_words_used = db.Column(db.Text)  # JSON list of custom words matched
    custom_lexicon_adjusted_score = db.Column(db.Float)
    
    # TextBlob
    textblob_polarity = db.Column(db.Float)
    textblob_subjectivity = db.Column(db.Float)
    
    # AFINN
    afinn_score = db.Column(db.Float)
    
    # SentiWordNet
    sentiwordnet_pos_score = db.Column(db.Float)
    sentiwordnet_neg_score = db.Column(db.Float)
    sentiwordnet_obj_score = db.Column(db.Float)
    
    # NRC Emotion Lexicon
    nrc_joy = db.Column(db.Float, default=0.0)
    nrc_anger = db.Column(db.Float, default=0.0)
    nrc_fear = db.Column(db.Float, default=0.0)
    nrc_sadness = db.Column(db.Float, default=0.0)
    nrc_trust = db.Column(db.Float, default=0.0)
    nrc_surprise = db.Column(db.Float, default=0.0)
    nrc_disgust = db.Column(db.Float, default=0.0)
    nrc_anticipation = db.Column(db.Float, default=0.0)
    
    # BERT/RoBERTa (fallback)
    bert_used = db.Column(db.Boolean, default=False)
    bert_score = db.Column(db.Float)
    bert_label = db.Column(db.String(10))
    
    # Decision Engine Output
    final_sentiment = db.Column(db.String(10))
    final_score = db.Column(db.Float)
    confidence_score = db.Column(db.Float)
    
    # Unknown words detected
    unknown_words_found = db.Column(db.Text)  # JSON list
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class DecisionEngineWeight(db.Model):
    """Configurable weights for the Decision Engine.
    
    Stored as key-value pairs for easy management.
    """
    __tablename__ = 'decision_engine_weights'
    
    id = db.Column(db.Integer, primary_key=True)
    source_name = db.Column(db.String(50), unique=True, nullable=False)  # 'vader', 'textblob', 'afinn', 'sentiwordnet', 'bert'
    weight = db.Column(db.Float, nullable=False, default=1.0)
    is_active = db.Column(db.Boolean, default=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    @classmethod
    def get_default_weights(cls):
        """Return default weights if none are configured."""
        return {
            'vader': 1.0,
            'textblob': 0.6,
            'afinn': 0.7,
            'sentiwordnet': 0.5,
            'bert': 0.8
        }


# ==================== SOLUTION RECOMMENDATION MODELS ====================

class SolutionTemplate(db.Model):
    """Configurable solution recommendation templates.
    
    Admins can create/edit/delete templates via the admin panel.
    If a template exists for a category in the DB, it takes precedence
    over the hardcoded templates in solution_recommender.py.
    """
    __tablename__ = 'solution_templates'

    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(50), nullable=False, index=True)
    keywords = db.Column(db.Text, nullable=False)  # comma-separated
    short_term_solution = db.Column(db.Text, nullable=False)
    long_term_solution = db.Column(db.Text, nullable=False)
    responsible_department = db.Column(db.String(200), nullable=False)
    estimated_time = db.Column(db.String(100), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    usage_count = db.Column(db.Integer, default=0)
    resolution_count = db.Column(db.Integer, default=0)
    created_by = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def resolution_rate(self):
        if self.usage_count > 0:
            return round((self.resolution_count / self.usage_count) * 100, 1)
        return 0.0

    def to_dict(self):
        return {
            'id': self.id,
            'category': self.category,
            'keywords': self.keywords,
            'short_term_solution': self.short_term_solution,
            'long_term_solution': self.long_term_solution,
            'responsible_department': self.responsible_department,
            'estimated_time': self.estimated_time,
            'is_active': self.is_active,
            'usage_count': self.usage_count,
            'resolution_count': self.resolution_count,
            'resolution_rate': self.resolution_rate,
            'created_by': self.created_by,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else '',
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M') if self.updated_at else '',
        }


class SolutionFeedback(db.Model):
    """Tracks whether a solution recommendation was helpful.
    
    Enables a feedback loop: "Was this recommendation helpful?"
    Students/admins can mark solutions as helpful or not.
    Over time, this data improves which templates are used.
    """
    __tablename__ = 'solution_feedback'

    id = db.Column(db.Integer, primary_key=True)
    feedback_id = db.Column(db.Integer, db.ForeignKey('feedback.id'), nullable=False)
    template_category = db.Column(db.String(50), nullable=False)
    was_helpful = db.Column(db.Boolean, nullable=False)
    resolved_after = db.Column(db.Boolean, default=False)  # Was the issue eventually resolved?
    resolution_time_hours = db.Column(db.Float, nullable=True)  # Hours between submission and resolution
    comment = db.Column(db.Text)
    created_by = db.Column(db.String(100))  # admin or student ID
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'feedback_id': self.feedback_id,
            'template_category': self.template_category,
            'was_helpful': self.was_helpful,
            'resolved_after': self.resolved_after,
            'resolution_time_hours': self.resolution_time_hours,
            'comment': self.comment,
            'created_by': self.created_by,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else '',
        }


# ==================== NOTIFICATION MODEL ====================

class Notification(db.Model):
    """Notifications for students and admins."""
    __tablename__ = 'notifications'

    id = db.Column(db.Integer, primary_key=True)
    recipient_type = db.Column(db.String(20), nullable=False, index=True)  # 'student' or 'admin'
    student_id = db.Column(db.String(20), db.ForeignKey('students.student_id'), nullable=True, index=True)
    admin_id = db.Column(db.Integer, db.ForeignKey('src_users.id'), nullable=True, index=True)
    feedback_id = db.Column(db.Integer, db.ForeignKey('feedback.id'), nullable=True, index=True)
    notification_type = db.Column(db.String(50), nullable=False)  # 'new_feedback', 'unattended', 'status_change', 'response'
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            'id': self.id,
            'recipient_type': self.recipient_type,
            'student_id': self.student_id,
            'admin_id': self.admin_id,
            'feedback_id': self.feedback_id,
            'notification_type': self.notification_type,
            'title': self.title,
            'message': self.message,
            'is_read': self.is_read,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else '',
        }


# ==================== SENTIMENT CORRECTION MODEL ====================

class SentimentCorrection(db.Model):
    """Stores admin corrections to sentiment analysis for active learning"""
    __tablename__ = 'sentiment_corrections'

    id = db.Column(db.Integer, primary_key=True)
    feedback_id = db.Column(db.Integer, db.ForeignKey('feedback.id'), nullable=False, index=True)
    original_sentiment = db.Column(db.String(20), nullable=False)
    corrected_sentiment = db.Column(db.String(20), nullable=False)
    admin_name = db.Column(db.String(100), nullable=False)
    confidence_before = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            'id': self.id,
            'feedback_id': self.feedback_id,
            'original_sentiment': self.original_sentiment,
            'corrected_sentiment': self.corrected_sentiment,
            'admin_name': self.admin_name,
            'confidence_before': self.confidence_before,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else '',
        }


class FeedbackCorrectionMemory(db.Model):
    """Stores the cleaned-text vector of every admin-corrected feedback row so
    that future, similar feedback can be auto-relabelled to match the admin's
    decision. This is the project's active-learning 'memory': once an admin has
    corrected a piece of feedback, the system remembers it and applies the same
    label to text that is cosine-similar (>=0.75 TF-IDF cosine) at runtime.

    Rows are inserted by:
      - review_ai_feedback / api/ai-review/correct on every admin correction
      - backfill_corrections.py one-shot migration over historical
        SentimentCorrection rows

    The TF-IDF vector is stored as a JSON-encoded sparse dict
    ({feature_index: weight}) so we can reuse the *same* fitted vectorizer that
    the runtime uses without needing to refit it for every lookup.
    """
    __tablename__ = 'feedback_correction_memory'

    id = db.Column(db.Integer, primary_key=True)
    feedback_id = db.Column(db.Integer, db.ForeignKey('feedback.id'),
                            nullable=False, unique=True, index=True)
    cleaned_text = db.Column(db.Text, nullable=False)
    tfidf_vector = db.Column(db.Text, nullable=False)   # JSON sparse dict
    sentiment = db.Column(db.String(20), nullable=False, index=True)
    category = db.Column(db.String(50), nullable=False, index=True)
    urgency_score = db.Column(db.Integer, nullable=False)
    admin_name = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        db.Index('idx_mem_sentiment_category', 'sentiment', 'category'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'feedback_id': self.feedback_id,
            'cleaned_text': self.cleaned_text,
            'sentiment': self.sentiment,
            'category': self.category,
            'urgency_score': self.urgency_score,
            'admin_name': self.admin_name,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else '',
        }


class PredictionOutcome(db.Model):
    """Tracks whether a predictive analytics warning was accurate.

    This lets the system learn from admin feedback and improve confidence
    calibration over time.
    """
    __tablename__ = 'prediction_outcomes'

    id = db.Column(db.Integer, primary_key=True)
    event = db.Column(db.String(200), nullable=False, index=True)
    source_type = db.Column(db.String(20), nullable=False, index=True)
    predicted_confidence = db.Column(db.Integer, nullable=False)
    outcome = db.Column(db.String(20), nullable=False, index=True)
    admin_notes = db.Column(db.Text)
    created_by = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        db.Index('idx_prediction_event_source', 'event', 'source_type'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'event': self.event,
            'source_type': self.source_type,
            'predicted_confidence': self.predicted_confidence,
            'outcome': self.outcome,
            'admin_notes': self.admin_notes,
            'created_by': self.created_by,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else '',
        }


# ==================== HELPER FUNCTIONS ====================

def is_valid_htu_email(email):
    pattern = r'^03\d{8}@htu\.edu\.gh$'
    return re.match(pattern, email) is not None

def extract_student_id_from_email(email):
    if email and '@' in email:
        return email.split('@')[0]
    return None
