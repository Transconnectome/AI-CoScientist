"""Literature/knowledge base database models."""

from datetime import date
from typing import List, Optional
from uuid import UUID

from sqlalchemy import String, Text, Integer, Date, ForeignKey, Float, Table, Column
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import BaseModel, Base


# Association tables for many-to-many relationships
paper_authors = Table(
    "paper_authors",
    Base.metadata,
    Column("paper_id", PGUUID(as_uuid=True), ForeignKey("literature.id", ondelete="CASCADE")),
    Column("author_id", PGUUID(as_uuid=True), ForeignKey("authors.id", ondelete="CASCADE")),
    Column("author_order", Integer),
    Column("is_corresponding", Integer, default=0)
)

paper_fields = Table(
    "paper_fields",
    Base.metadata,
    Column("paper_id", PGUUID(as_uuid=True), ForeignKey("literature.id", ondelete="CASCADE")),
    Column("field_id", PGUUID(as_uuid=True), ForeignKey("fields_of_study.id", ondelete="CASCADE")),
    Column("relevance_score", Float, nullable=True)
)

project_papers = Table(
    "project_papers",
    Base.metadata,
    Column("project_id", PGUUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE")),
    Column("paper_id", PGUUID(as_uuid=True), ForeignKey("literature.id", ondelete="CASCADE")),
    Column("relevance_score", Float, nullable=True),
    Column("notes", Text, nullable=True)
)


class Literature(BaseModel):
    """Scientific literature/paper model."""

    __tablename__ = "literature"

    doi: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    publication_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    journal: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    volume: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    issue: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    pages: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    citations_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pdf_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    authors: Mapped[List["Author"]] = relationship(
        "Author",
        secondary=paper_authors,
        back_populates="papers"
    )
    fields: Mapped[List["FieldOfStudy"]] = relationship(
        "FieldOfStudy",
        secondary=paper_fields,
        back_populates="papers"
    )
    citations_citing: Mapped[List["Citation"]] = relationship(
        "Citation",
        foreign_keys="Citation.citing_paper_id",
        back_populates="citing_paper"
    )
    citations_cited: Mapped[List["Citation"]] = relationship(
        "Citation",
        foreign_keys="Citation.cited_paper_id",
        back_populates="cited_paper"
    )


class Author(BaseModel):
    """Author model."""

    __tablename__ = "authors"

    name: Mapped[str] = mapped_column(String(500), nullable=False)
    affiliation: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    orcid: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    h_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    papers: Mapped[List["Literature"]] = relationship(
        "Literature",
        secondary=paper_authors,
        back_populates="authors"
    )


class FieldOfStudy(BaseModel):
    """Field of study/research domain model."""

    __tablename__ = "fields_of_study"

    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    parent_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("fields_of_study.id"),
        nullable=True
    )
    level: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Relationships
    papers: Mapped[List["Literature"]] = relationship(
        "Literature",
        secondary=paper_fields,
        back_populates="fields"
    )
    parent: Mapped[Optional["FieldOfStudy"]] = relationship(
        "FieldOfStudy",
        remote_side="FieldOfStudy.id",
        back_populates="children"
    )
    children: Mapped[List["FieldOfStudy"]] = relationship(
        "FieldOfStudy",
        back_populates="parent"
    )


class Citation(BaseModel):
    """Citation relationship between papers."""

    __tablename__ = "citations"

    citing_paper_id: Mapped[UUID] = mapped_column(
        ForeignKey("literature.id", ondelete="CASCADE"),
        nullable=False
    )
    cited_paper_id: Mapped[UUID] = mapped_column(
        ForeignKey("literature.id", ondelete="CASCADE"),
        nullable=False
    )
    context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    citing_paper: Mapped["Literature"] = relationship(
        "Literature",
        foreign_keys=[citing_paper_id],
        back_populates="citations_citing"
    )
    cited_paper: Mapped["Literature"] = relationship(
        "Literature",
        foreign_keys=[cited_paper_id],
        back_populates="citations_cited"
    )
