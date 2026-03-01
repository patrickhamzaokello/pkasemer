#!/usr/bin/env python3
"""
Pknwitq user management CLI.

Usage:
  python create_user.py                    — interactive: add a new user
  python create_user.py list               — list all users
  python create_user.py delete <email>     — delete a user by email
"""

import os
import re
import sys
import sqlite3
import getpass

try:
    from werkzeug.security import generate_password_hash
except ImportError:
    sys.exit("Error: werkzeug is not installed. Run: pip install werkzeug")

# Auto-detect path: use /data/users.db in Docker, ./data/users.db locally.
_default = "/data/users.db" if os.path.isdir("/data") else "./data/users.db"
DB_PATH = os.environ.get("USERS_DB_PATH", _default)

EMAIL_RE = re.compile(r"^[\w+.\-]+@[\w.\-]+\.\w{2,}$")


def _get_conn():
    data_dir = os.path.dirname(os.path.abspath(DB_PATH))
    os.makedirs(data_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT NOT NULL,
            email         TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at    TEXT DEFAULT (datetime('now')),
            last_login    TEXT
        )
    """)
    conn.commit()
    return conn


def cmd_add():
    print("=== Add New User ===")

    name = input("Name: ").strip()
    if not name:
        sys.exit("Error: name is required.")

    email = input("Email: ").strip().lower()
    if not EMAIL_RE.match(email):
        sys.exit("Error: invalid email format.")

    while True:
        password = getpass.getpass("Password (min 12 chars): ")
        if len(password) < 12:
            print("  Password must be at least 12 characters. Try again.")
            continue
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("  Passwords do not match. Try again.")
            continue
        break

    password_hash = generate_password_hash(password)

    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, password_hash),
        )
        conn.commit()
        print(f"\nUser '{name}' <{email}> created successfully.")
        print(f"DB: {os.path.abspath(DB_PATH)}")
    except sqlite3.IntegrityError:
        sys.exit(f"Error: a user with email '{email}' already exists.")
    finally:
        conn.close()


def cmd_list():
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, name, email, created_at, last_login FROM users ORDER BY id"
    ).fetchall()
    conn.close()

    if not rows:
        print("No users found.")
        return

    print(f"\n{'ID':<4}  {'Name':<22}  {'Email':<32}  {'Created':<19}  {'Last Login'}")
    print("─" * 92)
    for r in rows:
        created   = (r["created_at"]  or "")[:19]
        last_login = (r["last_login"] or "never")[:19]
        print(f"{r['id']:<4}  {r['name']:<22}  {r['email']:<32}  {created:<19}  {last_login}")


def cmd_delete(email: str):
    conn = _get_conn()
    cur = conn.execute("DELETE FROM users WHERE email = ?", (email.strip().lower(),))
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        sys.exit(f"No user found with email '{email}'.")
    print(f"User '{email}' deleted.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        cmd_add()
    elif args[0] == "list":
        cmd_list()
    elif args[0] == "delete":
        if len(args) < 2:
            sys.exit("Usage: python create_user.py delete <email>")
        cmd_delete(args[1])
    else:
        print(__doc__)
        sys.exit(1)
