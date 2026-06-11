import logging

import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi import Depends
from contextlib import asynccontextmanager

from app.routers.auth import lifespan_auth, User, current_active_user
from app.routers.llm import lifespan_llm
from app.routers.ollama import lifespan_ollama
from app.routers import auth, collection, file, chat, llm, ollama, huggingface, docker, setting, websocket
from app.utils.websocket_manager import ws_manager

logger = logging.getLogger(__name__)

def conf_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="VeraDoc Portal",
        description="""
        Management API for the VeraDoc Private RAG Platform.

        * auth: JWT token handling, registration and CRUD operations and permission management for users.
        * collections: CRUD operations for collections.
        * files: CRUD operations for files.
        * models: LLM integration services.
        """,
        version="1.0.0",
        contact={
            "name": "Miguel Salinas Gancedo",
            "url": "http://veradoc.ai",
            "email": "masalinas.gancedo@gmail.com",
        },
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        },
        # This hides the default 'Schemas' section at the bottom if you want it cleaner
        # docs_url="/admin-docs", # You can even change the URL from /docs to something else
        routes=app.routes,
    )

    # This manually corrects the security scheme path if it's stuck
    if "components" in openapi_schema and "securitySchemes" in openapi_schema["components"]:
            for scheme_name, scheme in openapi_schema["components"]["securitySchemes"].items():
                # Check if this is an oauth2 type scheme
                if scheme.get("type") == "oauth2" and "flows" in scheme:
                    for flow_type, flow_data in scheme["flows"].items():
                        if "tokenUrl" in flow_data:
                            # Update the nested tokenUrl
                            flow_data["tokenUrl"] = "/api/v1/auth/jwt/login"
                            print(f"Successfully updated tokenUrl for {scheme_name} -> {flow_type}")

    app.openapi_schema = openapi_schema

    return app.openapi_schema

# 1. Initialize FastAPI App with Auth and LLM Management configuration
@asynccontextmanager
async def main_lifespan(app: FastAPI):
    async with lifespan_auth(app):
        async with lifespan_llm(app):
            async with lifespan_ollama(app):
                yield  # The app runs here

app = FastAPI(lifespan=main_lifespan)

# 2. Add CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],    # Allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],    # Allows Content-Type, Authorization, etc.
    expose_headers=["Content-Disposition", "Content-Length"]
)

# 3. Add OpenAPI configuration
app.openapi = conf_openapi

# 4. Add service routers configuration
app.include_router(auth.router)
app.include_router(collection.router)
app.include_router(file.router)
app.include_router(chat.router)
app.include_router(llm.router)
app.include_router(ollama.router)
app.include_router(huggingface.router)
app.include_router(docker.router)
app.include_router(setting.router)
app.include_router(websocket.router)

@app.get("/health", tags=["system"], summary="Health Check")
async def health_check():
    return {"status": "ok"}

@app.get("/ws", tags=["system"], summary="Websocket Check")
async def websocket_check(
    text: str = Query(None, description="Websocket Text Test"),
    current_active_user: User = Depends(current_active_user)
):    
    await ws_manager.send_to_user(
        current_active_user.id, 
        {
            "event": "ws.test",
            "from_id": str(current_active_user.id),
            "text": text,
            "status": "ready"
        }
    )

    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8808, reload=True)