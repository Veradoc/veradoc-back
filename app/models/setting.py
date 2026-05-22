from sqlalchemy import Column, String

from app.routers.auth import Base

class Setting(Base):
    __tablename__ = "settings"

    id = Column(String, primary_key=True)
    key = Column(String)
    value = Column(String)