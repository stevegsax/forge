"""API endpoints for the simple project."""

from __future__ import annotations

from myapp.models import Item, User


def get_users() -> list[User]:
    return []


def get_items() -> list[Item]:
    return []


def create_user(name: str, email: str) -> User:
    return User(name=name, email=email)
