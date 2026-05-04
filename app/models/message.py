import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

from app.routers.auth import Base

class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=uuid.uuid4)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="messages")