import json
from typing import Optional
import requests
import uuid
import boto3
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import desc, select, func
from fastapi import APIRouter, Depends, Request, Query
from fastapi.responses import StreamingResponse

from app.utils.const import *
from app.utils.vector_util import search
from app.config import settings
from app.routers.auth import User, current_active_user, get_async_session

from app.models.conversation import Conversation
from app.models.message import Message

s3 = boto3.client(
    "s3",
    endpoint_url=settings.minio_endpoint,
    aws_access_key_id=settings.minio_access_key,
    aws_secret_access_key=settings.minio_secret_key,
    config=boto3.session.Config(s3={'addressing_style': 'path'})
)

router = APIRouter(
    prefix="/api/v1/chats",
    tags=["chats"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)

@router.post("/promt")
async def chat_endpoint(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    current_active_user: User = Depends(current_active_user)
):
    data = await request.json()
    conversation_id = data.get("conversationId")
    user_question = data.get("question")
    active_RAG = data.get("activeRAG")

    # 1. Save User Conversation if is new or gte id
    if not conversation_id:
        # Create a new UUID string
        conversation_id = str(uuid.uuid4()) 
        new_conv = Conversation(
            id=conversation_id, 
            user_id=str(current_active_user.id), 
            title=user_question[:30]
        )
        session.add(new_conv)
        await session.commit()
    else:
        conversation_id = str(conversation_id)

    # 2. Save User Message
    user_msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id, 
        role="user", 
        content=user_question
    )
    session.add(user_msg)
    await session.commit()

    # 3. Fetch History (Last 6 messages for context)
    stmt = (
        select(Message)
        .where(Message.conversation_id == uuid.UUID(conversation_id))
        .order_by(Message.created_at.desc())
        .limit(6)
    )
    result = await session.execute(stmt)
    db_history = result.scalars().all()

    # Reverse to chronological order: [User, AI, User, AI...]
    history_msgs = [{"role": m.role, "content": m.content} for m in reversed(db_history)]

    async def event_generator():
        # 4. chek if the requets must be used knowledge base
        if (active_RAG):
            # 4. Perform your search (RAG Logic)
            res = search(user_question)
            documents = " ".join([d["text"].strip() for d in res.to_list()])

            content = RAG_PROMPT.format(user_question=user_question, documents=documents)

            # Prepare the context dataframe equivalent for the frontend
            # We send this as the FIRST chunk so the UI updates the table immediately
            context_data = res.to_pandas().drop(columns=['source', 'vector']).to_dict(orient="records")

            yield f"data: {json.dumps({'type': 'id', 'conversation_id': conversation_id})}\n\n"
            yield f"data: {json.dumps({'type': 'context', 'data': context_data})}\n\n"            

        else:
            content = user_question
        
            yield f"data: {json.dumps({'type': 'id', 'conversation_id': conversation_id})}\n\n"

        # 5. Call the LLM with streaming
        # Use a context manager (with) to ensure the connection to the LLM is closed        
        full_ai_response = ""

        with requests.post(
            settings.ollama_host + "/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": history_msgs + [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "options": {
                    "temperature": 0,
                    "top_p": 0.90,
                }
            },
            stream=True
        ) as llm_resp:
            for resp in llm_resp.iter_lines():                
                # CRITICAL: Check if the frontend user is still there
                if await request.is_disconnected():
                    print("Client disconnected. Stopping LLM request.")
                    break # This stops the loop and exits the generator

                if resp:
                    json_data = json.loads(resp)
                    print(json_data)
                    content = json_data["message"]["content"]

                    full_ai_response += content

                    # Send text chunks
                    yield f"data: {json.dumps({'type': 'text', 'conversation_id': conversation_id, 'content': content})}\n\n"

        # 6. Save Final AI Response to SQLite
        if full_ai_response:
            ai_msg = Message(
                id=str(uuid.uuid4()), 
                conversation_id=str(conversation_id), 
                role="assistant", 
                content=full_ai_response
            )
            session.add(ai_msg)

            await session.commit()                 

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/history/conversations/recents")
async def get_recent_conversations(
    limit: int = 10,
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(current_active_user)
):
    stmt = (
        select(Conversation)
            .where(Conversation.user_id == str(current_user.id))
            .order_by(Conversation.created_at.desc())
            .limit(limit)
    )
        
    result = await session.execute(stmt)
    
    conversations = result.scalars().all()
    
    return conversations

@router.get("/history/chats/recents")
async def get_user_chats(
    offset: int = Query(default=0, le=100),
    limit: int = Query(default=10, ge=0),
    search: Optional[str] = Query(None, description="Search text inside message content"),
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(current_active_user)
):
    # 1. Start the base query with the JOIN
    stmt = (
        select(Message)
        .join(Conversation)
        .where(Conversation.user_id == str(current_user.id))
    )

    # 2. Add Search Filter if the 'search' string is provided
    if search:
        # SQL equivalent: WHERE messages.content ILIKE '%search_term%'
        stmt = stmt.where(Message.content.ilike(f"%{search}%"))

    # 3. Apply Ordering and Pagination
    stmt = (
        stmt.order_by(Message.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
        
    result = await session.execute(stmt)
    chats = result.scalars().all()
    
    return chats

@router.get("/history/conversations/count")
async def get_conversations_count(
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(current_active_user)
):
    stmt = (
        select(func.count())
            .select_from(Conversation)
            .where(Conversation.user_id == str(current_user.id))
    )
    
    result = await session.execute(stmt)
    total = result.scalar()
    
    return {"total": total}

@router.get("/history/conversations/{conversation_id}")
async def get_chats_by_conversation_id(
    conversation_id: str,
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(current_active_user)
):
    stmt = (
        select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
    )
    
    result = await session.execute(stmt)
    
    messages = result.scalars().all()
        
    return messages

@router.get("/history/conversations/latest/messages")
async def get_latest_conversation_messages(
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(current_active_user)
):
    # 1. Get the ID of the most recent conversation
    latest_conv_stmt = (
        select(Conversation.id)
            .where(Conversation.user_id == str(current_user.id))
            .order_by(desc(Conversation.created_at))
            .limit(1)
    )
    conv_result = await session.execute(latest_conv_stmt)
    latest_conv_id = conv_result.scalar_one_or_none()

    if not latest_conv_id:
        return {"messages": [], "conversation_id": None}

    # 2. Get all messages for that specific ID
    messages_stmt = (
        select(Message)
        .where(Message.conversation_id == latest_conv_id)
        .order_by(Message.id.desc()) # Ascending so chat reads top-to-bottom
    )

    msg_result = await session.execute(messages_stmt)
    messages = msg_result.scalars().all()

    return {
        "conversation_id": latest_conv_id,
        "messages": messages
    }