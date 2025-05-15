from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(String(255), unique=True, nullable=False)

    vectorstores = relationship(
        "VectorStore", back_populates="user", cascade="all, delete-orphan"
    )


class VectorStore(Base):
    __tablename__ = "vectorstores"

    vectorstore_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    file_name = Column(String(255), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="vectorstores")
    documents = relationship(
        "Document", back_populates="vectorstore", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("user_id", "file_name", name="uix_user_vectorstore_file_name"),
    )


class Document(Base):
    __tablename__ = "documents"

    doc_id = Column(Integer, primary_key=True, index=True)
    vectorstore_id = Column(Integer, ForeignKey("vectorstores.vectorstore_id"))
    content = Column(Text)
    doc_metadata = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    embedding = Column(Vector(1024))

    vectorstore = relationship("VectorStore", back_populates="documents")
