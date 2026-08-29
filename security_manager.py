"""
Security & Session Management Module
- Session management
- Login history
- Anomaly detection
- Enhanced audit logging
"""

from datetime import datetime, timedelta
from flask import request
from database import db, SystemLog


class SecurityManager:
    """Manages security features for the application."""

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


def get_client_ip():
    """Get the client's real IP address."""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    return request.remote_addr


def get_user_agent():
    """Get the client's user agent."""
    return request.headers.get('User-Agent', 'Unknown')[:500]
