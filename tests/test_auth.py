from nab_system.auth import (
    ADMIN_ROLE,
    DUTY_ROLE,
    default_users,
    hash_password,
    role_allows,
    verify_password,
)


def test_password_hash_roundtrip():
    encoded = hash_password("s3cret!", iterations=1000)
    assert encoded.startswith("pbkdf2_sha256$1000$")
    assert verify_password("s3cret!", encoded)
    assert not verify_password("wrong", encoded)
    assert not verify_password("s3cret!", None)
    assert not verify_password("s3cret!", "garbage")


def test_role_hierarchy_allows_admin_everywhere_but_operator_cannot_observe():
    assert role_allows("operator", "operator")
    assert role_allows("admin", "operator")
    assert role_allows("admin", "admin")
    assert not role_allows("operator", "admin")
    assert not role_allows("viewer", "operator")
    assert not role_allows("unknown", "operator")


def test_default_users_include_operator_and_admin():
    roles = {user["role"] for user in default_users()}
    assert roles == {DUTY_ROLE, ADMIN_ROLE}
