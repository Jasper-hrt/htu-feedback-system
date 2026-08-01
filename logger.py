import logging
import os
from logging.handlers import RotatingFileHandler

os.makedirs('logs', exist_ok=True)

DETAILED_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s'
SIMPLE_FORMAT = '%(asctime)s | %(levelname)-8s | %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def setup_logger(name, log_file, level=logging.INFO, format_type='detailed'):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    file_handler = RotatingFileHandler(f'logs/{log_file}', maxBytes=5*1024*1024, backupCount=5)
    format_string = DETAILED_FORMAT if format_type == 'detailed' else SIMPLE_FORMAT
    file_handler.setFormatter(logging.Formatter(format_string, DATE_FORMAT))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(SIMPLE_FORMAT, DATE_FORMAT))
    logger.addHandler(console_handler)
    
    return logger

system_logger = setup_logger('system', 'system.log')
auth_logger = setup_logger('auth', 'auth.log')
feedback_logger = setup_logger('feedback', 'feedback.log')
admin_logger = setup_logger('admin', 'admin.log')

def log_student_action(student_id, action, details, status='INFO'):
    log_func = {'INFO': auth_logger.info, 'WARNING': auth_logger.warning, 'ERROR': auth_logger.error}.get(status, auth_logger.info)
    log_func(f"Student: {student_id} | Action: {action} | Details: {details}")

def log_admin_action(admin_name, action, details, status='INFO'):
    log_func = {'INFO': admin_logger.info, 'WARNING': admin_logger.warning, 'ERROR': admin_logger.error}.get(status, admin_logger.info)
    log_func(f"Admin: {admin_name} | Action: {action} | Details: {details}")

def log_feedback_action(feedback_id, student_id, action, details, status='INFO'):
    log_func = {'INFO': feedback_logger.info, 'WARNING': feedback_logger.warning, 'ERROR': feedback_logger.error}.get(status, feedback_logger.info)
    log_func(f"Feedback ID: {feedback_id} | Student: {student_id} | Action: {action} | Details: {details}")

def log_system_action(component, action, details, status='INFO'):
    log_func = {'INFO': system_logger.info, 'WARNING': system_logger.warning, 'ERROR': system_logger.error}.get(status, system_logger.info)
    log_func(f"Component: {component} | Action: {action} | Details: {details}")