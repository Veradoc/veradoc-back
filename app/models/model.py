from sqlalchemy import Column, String

from app.routers.auth import Base

class Model(Base):
    __tablename__ = "models"

    id = Column(String, primary_key=True)
    ollama_name = Column(String)
    huggingface_name = Column(String)
    pipeline_tag = Column(String)