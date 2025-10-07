"""User and authentication models."""

from enum import Enum
from typing import Optional

from sqlalchemy import String, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import BaseModel


class UserRole(str, Enum):
    """User role enum."""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    REVIEWER = "reviewer"


class User(BaseModel):
    """User model."""

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(
        String(50),
        default=UserRole.RESEARCHER.value,
        nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    api_key: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)
