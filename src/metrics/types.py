# src/metrics/types.py
from sqlalchemy import Column, String, Boolean, Float, Integer, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from src.models.base import Base
from datetime import datetime
from uuid import uuid4


class AgentExecution(Base):
    __tablename__ = "agent_executions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id = Column(String(100), nullable=False, index=True)
    task_type = Column(String(50), nullable=False)
    task_id = Column(String(100), nullable=False)
    success = Column(Boolean, nullable=False)
    confidence = Column(Float)
    execution_time_ms = Column(Float)
    tokens_used = Column(Integer)
    quality_score = Column(Float)
    extra_metadata = Column(JSONB)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class WorkflowMetric(Base):
    __tablename__ = "workflow_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    task_type = Column(String(50), nullable=False)
    agents_used = Column(JSONB, nullable=False)
    quality_score = Column(Float, nullable=False)
    execution_time_ms = Column(Float, nullable=False)
    success = Column(Boolean, nullable=False)
    extra_metadata = Column(JSONB)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
