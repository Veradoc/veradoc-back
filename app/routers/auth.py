import logging
import uuid
from typing import Optional, List

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request, APIRouter, status, HTTPException
from fastapi_users import FastAPIUsers, schemas, BaseUserManager, UUIDIDMixin
from fastapi_users.authentication import AuthenticationBackend, BearerTransport, JWTStrategy
from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTableUUID, SQLAlchemyUserDatabase

from sqlalchemy import select, Column, String
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from httpx_oauth.clients.google import GoogleOAuth2
from google.oauth2 import id_token
from google.auth.transport import requests

from app.utils.const import *

logger = logging.getLogger(__name__)

DATABASE_URL = "sqlite+aiosqlite:///./db/veradoc.db"
SECRET = "SECRET_KEY_CHANGE_THIS_IN_PRODUCTION"

# Configure the Google OAuth Client
google_oauth_client = GoogleOAuth2(
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET
)

# 1. Database Setup
class Base(DeclarativeBase): pass

class User(SQLAlchemyBaseUserTableUUID, Base):
    name = Column(String, nullable=False)

class UserRead(schemas.BaseUser[uuid.UUID]):
    name: str

class UserCreate(schemas.BaseUserCreate):
    name: str
    
class UserUpdate(schemas.BaseUserUpdate):
    name: Optional[str] = None

class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        print(f"User {user.id} has registered.")

engine = create_async_engine(DATABASE_URL)
async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_async_session():
    async with async_session_maker() as session:
        yield session

async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, User)

async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)

@asynccontextmanager
async def lifespan_auth(app: FastAPI):
    # --- STARTUP LOGIC ---
    # This replaces @app.on_event("startup")
    async with engine.begin() as conn:
        # Create tables if they don't exist
        await conn.run_sync(Base.metadata.create_all)
    
    print("Database tables initialized.")
    
    yield  # The app stays here while it's running and serving requests
    
    # --- SHUTDOWN LOGIC ---
    # This replaces @app.on_event("shutdown")
    # You can close engine connections or cleanup here if needed
    print("Shutting down database resources.")

# 2. Auth Transport
bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")

def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

# 3. Schemas (FastAPI Users needs specific ones for Read/Create)
fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])
current_active_user = fastapi_users.current_user(active=True)
current_active_superuser = fastapi_users.current_user(active=True, superuser=True)

router = APIRouter(
    prefix="/api/v1",
    tags=["auth"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)

router.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)

router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate), 
    prefix="/auth",
    tags=["auth"],
    dependencies=[Depends(current_active_superuser)],
)

router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/auth",
    tags=["auth"],
)

router.include_router(
    fastapi_users.get_oauth_router(
        google_oauth_client, 
        auth_backend, 
        "SECRET_KEY_HERE",
        associate_by_email=True, # Important: links Google email to existing user
        is_verified_by_default=True
    ),
    prefix="/auth/google",
    tags=["auth"],
)

@router.post("/auth/register-admin", tags=["auth"])
async def admin_register(
    user_create: UserCreate,
    user_manager: UserManager = Depends(get_user_manager),
    _: User = Depends(current_active_superuser),
):
    return await user_manager.create(user_create, safe=False)

@router.get("/users/by-email/{email}", response_model=UserRead)
async def get_user_by_email(
    email: str,
    user_manager: UserManager = Depends(get_user_manager),
    current_user: User = Depends(current_active_superuser), # Ensure only admins can do this
):
    user = await user_manager.get_by_email(email)

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return user

@router.get("/users", response_model=List[UserRead])
async def list_users(
    # We inject the session directly instead of the user_manager for this query
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(current_active_superuser),
):
    result = await session.execute(select(User))
    
    users = result.scalars().all()
    
    return users

@router.post("/auth/google/token-exchange")
async def google_token_exchange(
    data: dict,
    # Inject the strategy and user manager here
    strategy: JWTStrategy = Depends(get_jwt_strategy),
    user_manager = Depends(get_user_manager)    
):
    token = data.get("token")
    try:
        # Verify the Google token...
        idinfo = id_token.verify_oauth2_token(token, requests.Request(), GOOGLE_CLIENT_ID)
        
        # Get the user from your database
        user = await user_manager.get_by_email(idinfo['email'])
        
        # THIS generates the HS256 token your backend loves
        backend_token = await strategy.write_token(user)
        
        return {"token": backend_token}
        
    except ValueError:
        # Invalid token
        raise HTTPException(status_code=401, detail="Invalid Google Token")