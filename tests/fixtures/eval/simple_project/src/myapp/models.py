"""Data models for the simple project."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class User:
    name: str
    email: str


@dataclass
class Item:
    title: str
    owner: User
