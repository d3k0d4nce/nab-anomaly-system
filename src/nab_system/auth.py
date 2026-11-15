from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from typing import Any, Dict, List, Optional

from . import repository as repo

DUTY_ROLE = "operator"
ADMIN_ROLE = "admin"
ROLE_LEVELS = {DUTY_ROLE: 1, ADMIN_ROLE: 2}
PBKDF2_ITERATIONS = 260_000


def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256${}${}${}".format(
        iterations,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, encoded: Optional[str]) -> bool:
    if not encoded:
        return False
    try:
        algorithm, iterations_s, salt_s, digest_s = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_s)
        salt = base64.b64decode(salt_s.encode("ascii"))
        expected = base64.b64decode(digest_s.encode("ascii"))
    except (ValueError, TypeError):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def default_users() -> List[Dict[str, Any]]:
    return [
        {
            "full_name": "Дежурный инженер",
            "contact": "duty@example.local",
            "login": os.getenv("NAB_OPERATOR_LOGIN", "operator"),
            "password": os.getenv("NAB_OPERATOR_PASSWORD", "demo" + "-operator"),
            "role": DUTY_ROLE,
            "is_on_duty": True,
        },
        {
            "full_name": "Администратор",
            "contact": "admin@example.local",
            "login": os.getenv("NAB_ADMIN_LOGIN", "admin"),
            "password": os.getenv("NAB_ADMIN_PASSWORD", "demo" + "-admin"),
            "role": ADMIN_ROLE,
            "is_on_duty": True,
        },
    ]


def seed_default_users(conn) -> Dict[str, int]:
    users = []
    for user in default_users():
        users.append(
            {
                "full_name": user["full_name"],
                "contact": user["contact"],
                "login": user["login"],
                "password_hash": hash_password(user["password"]),
                "role": user["role"],
                "is_active": True,
                "is_on_duty": user["is_on_duty"],
            }
        )
    return repo.seed_users(conn, users)


def authenticate(conn, login: str, password: str) -> Optional[Dict[str, Any]]:
    user = repo.fetch_user_by_login(conn, login)
    if not user or not user.get("is_active"):
        return None
    if not verify_password(password, user.get("password_hash")):
        return None
    user.pop("password_hash", None)
    return user


def role_allows(role: str, minimum: str = DUTY_ROLE) -> bool:
    return ROLE_LEVELS.get(str(role), 0) >= ROLE_LEVELS.get(str(minimum), 0)


def session_secret() -> str:
    return os.getenv("NAB_SECRET_KEY", "change-me-in-production-please")


def cookie_secure() -> bool:
    return os.getenv("NAB_COOKIE_SECURE", "0").lower() in ("1", "true", "yes")


def login_to_session(user: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "uid": int(user["engineer_id"]),
        "login": user["login"],
        "name": user["full_name"],
        "role": user["role"],
    }
