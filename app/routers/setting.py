

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.setting import Setting
from app.routers.auth import current_active_user, get_async_session
from app.routers.chat import set_top_vectors, set_top_rerankers_vectors

TOP_VECTORS_KEYS = "TOP_VECTORS"
TOP_RERANKER_VECTORS_KEYS = "TOP_RERANKER_VECTORS"

router = APIRouter(
    prefix="/api/v1/settings",
    tags=["settings"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)

# Schema definitions
class SettingResponse(BaseModel):
    status: str
    message: str
    key: str
    value: str  # Or Any/dict if you change 'value' to a JSON column later

class SettingValueResponse(BaseModel):
    key: str
    value: str | None = None # If you use JSON columns later, change this type to Any or dic

class SettingUpdatePayload(BaseModel):
    value: str

@router.post("/{key:path}",
    summary="Save a Key",
    description="""
    Creae a new key
    This is a **streaming endpoint** that provides real-time progress logs.
    """,
    response_description="A text stream of the CLI execution logs.",    
    dependencies=[Depends(current_active_user)])
async def save_key(
    key: str,
    payload: SettingUpdatePayload,
    session: AsyncSession = Depends(get_async_session)):
    """
    Pull a model. 
    Uses StreamingResponse to provide real-time logs.
    """
    # 1. Check if the key already exists in the database
    query = select(Setting).where(Setting.key == key)
    result = await session.execute(query)
    existing_setting = result.scalar_one_or_none()
    
    if existing_setting:
        # 2. Key exists -> Update the value
        existing_setting.value = payload.value
        action = "updated"
        message = f"Setting '{key}' successfully updated."
    else:
        key_id = str(uuid.uuid4())
        new_setting = Setting(
            id=key_id, 
            key=key, 
            value=payload.value
        )
        session.add(new_setting)
        action = "created"
        message = f"Setting '{key}' successfully created."

    # set running model for next chat
    if key == TOP_VECTORS_KEYS:
        set_top_vectors(payload.value)

    if key == TOP_RERANKER_VECTORS_KEYS:
        set_top_rerankers_vectors(payload.value)

    try:
        # Commit the transaction to disk
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database transaction failed: {str(e)}"
        )

    # 4. Return a structured response
    return SettingResponse(
        status="success",
        message=message,
        key=key,
        value=payload.value)

@router.get("/{key:path}", 
    summary="Get a key",
    description="""
    Get a key
    This is a **streaming endpoint** that provides real-time progress logs.
    """,
    response_description="A text stream of the CLI execution logs.",             
    dependencies=[Depends(current_active_user)],
    response_model=SettingValueResponse)
async def get_setting_by_key(
    key: str, 
    session: AsyncSession = Depends(get_async_session)
):
    """
    Retrieve the value of a specific configuration setting by its key.
    """
    # 1. Query the database for the matching key row
    query = select(Setting).where(Setting.key == key)
    result = await session.execute(query)
    setting = result.scalar_one_or_none()

    # 2. If the key isn't found, raise a clean 404 HTTP Error
    if not setting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Setting with key '{key}' not found."
        )

    # 3. Return the key and its value matching our Pydantic schema
    return SettingValueResponse(
        key=setting.key,
        value=setting.value
    )

@router.get("", 
    summary="Get all settings",
    description="""
    Get all settings.
    """,
    response_description="A list of settings.",
    response_model=list[SettingValueResponse])
async def get_all_settings(
    session: AsyncSession = Depends(get_async_session)
):
    """
    Retrieve all settings.
    """
    # 1. Query the database for the matching key row
    query = select(Setting)
    result = await session.execute(query)
    settings = result.scalars().all()

    # 3. Return the key and its value matching our Pydantic schema
    return settings 