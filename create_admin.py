# create_admin.py - Run to add SRC admins
import sqlite3
from werkzeug.security import generate_password_hash

def add_admin(username, full_name, role, password):
    conn = sqlite3.connect('instance/feedback.db')
    cursor = conn.cursor()
    hashed = generate_password_hash(password)
    cursor.execute('INSERT OR REPLACE INTO src_users (username, password_hash, full_name, role) VALUES (?, ?, ?, ?)', 
                   (username, hashed, full_name, role))
    conn.commit()
    print(f"Admin {username} added!")
    conn.close()

if __name__ == '__main__':
    add_admin('admin', 'SRC Administrator', 'President', 'admin123')
    print("Default admin created: admin / admin123")