"""Pydantic schemas for literature monitoring API."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class LiteratureSourceCreate(BaseModel):
    """Schema for creating a literature source."""

    source_type: str = Field(..., description="Type of source (arxiv or pubmed)")
    category: Optional[str] = Field(None, description="ArXiv categories (comma-separated)")
    query: Optional[str] = Field(None, description="PubMed search query")
    sync_frequency: str = Field("daily", description="Sync frequency (daily, weekly, monthly)")
    status: str = Field("active", description="Initial status (active, paused, disabled)")

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, v: str) -> str:
        """Validate source type."""
        if v not in ["arxiv", "pubmed"]:
            raise ValueError("source_type must be 'arxiv' or 'pubmed'")
        return v

    @field_validator("sync_frequency")
    @classmethod
    def validate_sync_frequency(cls, v: str) -> str:
        """Validate sync frequency."""
        if v not in ["daily", "weekly", "monthly"]:
            raise ValueError("sync_frequency must be 'daily', 'weekly', or 'monthly'")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status."""
        if v not in ["active", "paused", "disabled"]:
            raise ValueError("status must be 'active', 'paused', or 'disabled'")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "source_type": "arxiv",
                "category": "cs.AI,cs.LG,q-bio.NC",
                "sync_frequency": "daily",
                "status": "active",
            }
        }


class LiteratureSourceUpdate(BaseModel):
    """Schema for updating a literature source."""

    category: Optional[str] = Field(None, description="ArXiv categories (comma-separated)")
    query: Optional[str] = Field(None, description="PubMed search query")
    sync_frequency: Optional[str] = Field(None, description="Sync frequency")
    status: Optional[str] = Field(None, description="Status (active, paused, disabled)")

    @field_validator("sync_frequency")
    @classmethod
    def validate_sync_frequency(cls, v: Optional[str]) -> Optional[str]:
        """Validate sync frequency."""
        if v is not None and v not in ["daily", "weekly", "monthly"]:
            raise ValueError("sync_frequency must be 'daily', 'weekly', or 'monthly'")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: Optional[str]) -> Optional[str]:
        """Validate status."""
        if v is not None and v not in ["active", "paused", "disabled"]:
            raise ValueError("status must be 'active', 'paused', or 'disabled'")
        return v


class LiteratureSourceResponse(BaseModel):
    """Schema for literature source response."""

    id: UUID
    source_type: str
    category: Optional[str]
    query: Optional[str]
    last_sync_time: Optional[datetime]
    status: str
    sync_frequency: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MonitoringAlertCreate(BaseModel):
    """Schema for creating a monitoring alert."""

    topic: str = Field(..., description="Alert topic/title", min_length=1, max_length=200)
    keywords: List[str] = Field(..., description="Keywords to monitor", min_length=1)
    frequency: str = Field("daily", description="Alert frequency (daily, weekly, monthly)")
    user_id: Optional[UUID] = Field(None, description="User ID for personalized alerts")

    @field_validator("frequency")
    @classmethod
    def validate_frequency(cls, v: str) -> str:
        """Validate frequency."""
        if v not in ["daily", "weekly", "monthly"]:
            raise ValueError("frequency must be 'daily', 'weekly', or 'monthly'")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "topic": "AI in Neuroscience",
                "keywords": ["machine learning", "brain imaging", "fMRI"],
                "frequency": "weekly",
            }
        }


class MonitoringAlertUpdate(BaseModel):
    """Schema for updating a monitoring alert."""

    topic: Optional[str] = Field(None, description="Alert topic/title")
    keywords: Optional[List[str]] = Field(None, description="Keywords to monitor")
    frequency: Optional[str] = Field(None, description="Alert frequency")
    active: Optional[bool] = Field(None, description="Active status")

    @field_validator("frequency")
    @classmethod
    def validate_frequency(cls, v: Optional[str]) -> Optional[str]:
        """Validate frequency."""
        if v is not None and v not in ["daily", "weekly", "monthly"]:
            raise ValueError("frequency must be 'daily', 'weekly', or 'monthly'")
        return v


class MonitoringAlertResponse(BaseModel):
    """Schema for monitoring alert response."""

    id: UUID
    user_id: Optional[UUID]
    topic: str
    keywords: List[str]
    frequency: str
    last_alert_sent: Optional[datetime]
    active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SyncTriggerResponse(BaseModel):
    """Schema for sync trigger response."""

    task_id: str = Field(..., description="Celery task ID")
    source_id: Optional[str] = Field(None, description="Source ID being synced")
    source_type: Optional[str] = Field(None, description="Source type")
    status: str = Field(..., description="Task status")
    message: Optional[str] = Field(None, description="Additional message")


class SourceStatisticsResponse(BaseModel):
    """Schema for source statistics response."""

    source_id: str
    source_type: str
    status: str
    sync_frequency: str
    last_sync_time: Optional[str]
    time_since_sync_seconds: Optional[float]
    configuration: dict

    class Config:
        json_schema_extra = {
            "example": {
                "source_id": "123e4567-e89b-12d3-a456-426614174000",
                "source_type": "arxiv",
                "status": "active",
                "sync_frequency": "daily",
                "last_sync_time": "2025-10-11T10:00:00",
                "time_since_sync_seconds": 3600.0,
                "configuration": {
                    "categories": ["cs.AI", "cs.LG"],
                    "query": None,
                },
            }
        }
