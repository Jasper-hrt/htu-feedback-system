"""
Security & Session Management Module
- Two-Factor Authentication (2FA)
- Session management
- Login history
- Anomaly detection
- Enhanced audit logging
"""

import pyotp
import secrets
from datetime import datetime, timedelta
from functools import wraps
from flask import session, request, redirect, url_for, flash, jsonify
from database import db, SystemLog


class SecurityManager:
    """Manages security features for the application."""

    @staticmethod
    def generate_2fa_secret():
        """Generate a new 2FA secret for a user."""
        return pyotp.random_base32()

    @staticmethod
    def get_totp_uri(secret, username):
        """Get the TOTP URI for QR code generation."""
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=username, issuer_name="HTU SRC System")

    @staticmethod
    def verify_2fa_token(secret, token):
        """Verify a 2FA token."""
        if not secret or not token:
            return False
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)

    @staticmethod
    def record_login(user_id, user_type, ip_address, user_agent, success, details=""):
        """Record a login attempt."""
        try:
            log = SystemLog(
                timestamp=datetime.utcnow(),
                log_type='auth',
                level='INFO' if success else 'WARNING',
                user_type=user_type,
                user_id=str(user_id),
                action='login_success' if success else 'login_failed',
                details=details or f"Login {'succeeded' if success else 'failed'} from {ip_address}",
                ip_address=ip_address
            )
            db.session.add(log)
            db.session.commit()
        except Exception:
            db.session.rollback()

    @staticmethod
    def detect_anomaly(user_id, ip_address, user_agent):
        """Detect suspicious login activity."""
        # Check for multiple failed attempts
        recent_failures = SystemLog.query.filter(
            SystemLog.user_id == str(user_id),
            SystemLog.action == 'login_failed',
            SystemLog.timestamp >= datetime.utcnow() - timedelta(minutes=30)
        ).count()

        if recent_failures >= 5:
            return {
                'is_anomaly': True,
                'risk': 'high',
                'message': f'Multiple failed login attempts detected ({recent_failures} in 30 min)'
            }

        # Check for login from new IP
        known_ips = SystemLog.query.filter(
            SystemLog.user_id == str(user_id),
            SystemLog.action == 'login_success'
        ).with_entities(SystemLog.ip_address).distinct().all()
        known_ips = [ip[0] for ip in known_ips]

        if known_ips and ip_address not in known_ips:
            return {
                'is_anomaly': True,
                'risk': 'medium',
                'message': f'Login from new IP address: {ip_address}'
            }

        return {'is_anomaly': False, 'risk': 'low', 'message': ''}

    @staticmethod
    def get_login_history(user_id, limit=20):
        """Get recent login history for a user."""
        return SystemLog.query.filter(
            SystemLog.user_id == str(user_id),
            SystemLog.action.in_(['login_success', 'login_failed'])
        ).order_by(SystemLog.timestamp.desc()).limit(limit).all()

    @staticmethod
    def get_active_sessions():
        """Get count of active sessions (placeholder for Redis implementation)."""
        return 0


def require_2fa(f):
    """Decorator to require 2FA verification for sensitive operations."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('admin_id') and not session.get('2fa_verified'):
            return redirect(url_for('admin_2fa_verify'))
        return f(*args, **kwargs)
    return decorated_function


def get_client_ip():
    """Get the client's real IP address."""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    return request.remote_addr


def get_user_agent():
    """Get the client's user agent."""
    return request.headers.get('User-Agent', 'Unknown')[:500]
