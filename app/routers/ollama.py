import logging
import uuid
from contextlib import asynccontextmanager

from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ollama import AsyncClient

from app.utils.const import LLM_MODEL, DEF_TOP_VECTORS, DEF_TOP_RERANKER_VECTORS
from app.config import settings
from app.routers.auth import current_active_superuser, current_active_user, async_session_maker, get_async_session
from app.routers.huggingface import get_hf_model_info
from app.routers.chat import set_active_model, set_top_vectors, set_top_rerankers_vectors
from app.routers.setting import SettingUpdatePayload, get_setting_by_key, save_key
from app.models.model import Model

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["ollama"],
)

ACTIVE_LLM_MODEL_KEY = "ACTIVE_MODEL"
TOP_VECTORS_KEY = "TOP_VECTORS"
TOP_RERANKER_VECTORS_KEY = "TOP_RERANKER_VECTORS"

@asynccontextmanager
async def lifespan_ollama(app: FastAPI):    
    async with async_session_maker() as session:
        try:
            # system models configuration
            llm_model_setting = await get_setting_by_key(ACTIVE_LLM_MODEL_KEY, session)
            top_vectors_setting = await get_setting_by_key(TOP_VECTORS_KEY, session)
            top_reranker_vectors_setting = await get_setting_by_key(TOP_RERANKER_VECTORS_KEY, session)

            if llm_model_setting and llm_model_setting.value:                
                set_active_model(llm_model_setting.value)
                print(f"[LIFESPAN] Successfully restored active llm model: {llm_model_setting.value}")
            else:
                set_active_model(LLM_MODEL)
                print("[LIFESPAN] No llm model configuration found. Using default preset.")

            if top_vectors_setting and top_vectors_setting.value:                
                set_top_vectors(top_vectors_setting.value)
                print(f"[LIFESPAN] Successfully recovery top vectors: {top_vectors_setting.value}")
            else:
                set_top_vectors(DEF_TOP_VECTORS)
                print("[LIFESPAN] No model configuration found. Using default preset.")

            if top_reranker_vectors_setting and top_reranker_vectors_setting.value:                
                set_top_rerankers_vectors(top_reranker_vectors_setting.value)
                print(f"[LIFESPAN] Successfully recovery top rerankers vectors: {top_reranker_vectors_setting.value}")
            else:
                set_top_rerankers_vectors(DEF_TOP_RERANKER_VECTORS)
                print("[LIFESPAN] No model configuration found. Using default preset.")

        except Exception as e:
            set_active_model(LLM_MODEL)
            set_top_vectors(DEF_TOP_VECTORS)
            set_top_rerankers_vectors(DEF_TOP_RERANKER_VECTORS)

            print(f"[LIFESPAN ERROR] Failed to fetch settings from DB, set default model configuratins: {e}")

    yield  # Let control return to the main loop            
    
# Initialize Ollama client
client = AsyncClient(host=settings.ollama_host)

async def model_mapper(
        session: AsyncSession,
        huggingface_name: str, 
        pipeline_tag: str, 
        ollama_name: str):
    # 1. Lookup the tag in your local DB
    stmt = select(Model).where(
        Model.huggingface_name == huggingface_name, pipeline_tag == pipeline_tag
    )    
        
    result = await session.execute(stmt)
    
    model = result.scalars().first()

    # 2. Create huggingface ollama map
    if not model:
        model = Model(
            id=str(uuid.uuid4()),
            huggingface_name=huggingface_name, 
            ollama_name=ollama_name, 
            pipeline_tag=pipeline_tag            
        )

        session.add(model)

        await session.commit() 

    return model

@router.get("/models/ollama",
    summary="List all Ollama Models pulled",
    description="""
    Retrieves a list of AI models currently pulled on the local Ollama server. Useful for monitoring resource usage.
    """,
    response_description="List of currently pulled models.",    
    dependencies=[Depends(current_active_user)])
async def get_pulled_models(search: str = None):
    """
    Get Ollama compatible models from HuggenFace.
    The :search filer models (e.g., namespace/model).
    """    
    pulled = await client.list()
    running = await client.ps()
    running_names = {m.model for m in running.models}

    all_models = [
        {
            "name": m.model,
            "size": f"{m.size / 1e9:.2f} GB",
            "quant": m.details.quantization_level,
            "family": m.details.family,
            "params": m.details.parameter_size,
            "running": m.model in running_names
        } for m in pulled.models
    ]

    if search:
        search = search.lower()
        # Filters by name, family, or parameter size
        return [m for m in all_models if search in m['name'].lower() or search in m['family'].lower()]

    return all_models

@router.delete("/models/ollama/delete/{model_name:path}",
    summary="Delete Model",
    description="Permanently removes the model files from the local Ollama storage.",
    dependencies=[Depends(current_active_superuser)] 
)
async def delete_model(model_name: str):
    """
    Deletes a model using the Ollama SDK.
    The :path type allows for model names containing slashes (e.g., namespace/model).
    """    
    try:
        # Check if model exists first (Optional but cleaner)
        local_models = await client.list()
        exists = any(m.model == model_name for m in local_models.models)
        
        if not exists:
            # Sometimes Ollama refers to models with or without ':latest'
            # We check both to be safe
            if not model_name.endswith(':latest'):
                exists = any(m.model == f"{model_name}:latest" for m in local_models.models)
        
        if not exists:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

        # The SDK call to delete
        await client.delete(model=model_name)
        
        return {
            "status": "success", 
            "message": f"Model '{model_name}' has been removed from disk."
        }

    except Exception as e:
        # Catch SDK errors or connection issues
        raise HTTPException(status_code=500, detail=f"Ollama SDK Error: {str(e)}")
    
@router.post("/models/ollama/pull/{model_name:path}/{pipeline_tag}",
    summary="Pull Model",
    description="""
    Initiates the pulling (or downloading) of a specific model into the Ollama engine. 
    This is a **streaming endpoint** that provides real-time progress logs.
    """,
    response_description="A text stream of the CLI execution logs.",    
    dependencies=[Depends(current_active_superuser)])
async def pull_model(
    model_name: str,
    pipeline_tag: str,
    session: AsyncSession = Depends(get_async_session)):
    """
    Pull a model. 
    Uses StreamingResponse to provide real-time logs.
    """
    async def generate_logs():
        try:
            # pull() models published in huggenface. Ensures the model is on disk. stream=True returns an async generator.            
            pull_stream = await client.pull(model=f'hf.co/{model_name}', stream=True)

            async for part in pull_stream:
                # 'part' is a dict containing status, total, and completed bytes
                status = part.get('status', '')
                progress = ""
                
                if 'completed' in part and 'total' in part:
                    p = (part['completed'] / part['total']) * 100
                    progress = f" ({p:.2f}%)"
                
                yield f"{status}{progress}\n"

            # --- THE LOOP HAS FINISHED HERE ---
            yield "Pull complete. Registering and activating model...\n"

            # 2. Get the actual local name from Ollama
            local_models = await client.list()
            # We look for a model that contains our model_name 
            # (Ollama often keeps the 'hf.co/' prefix in the local name)
            registered_name = next(
                (m['model'] for m in local_models['models'] if model_name in m['model']), 
                f"hf.co/{model_name}" # Fallback to the expected name
            )

            # 3. Map huggingface to ollama model
            await model_mapper(session, model_name, pipeline_tag, registered_name)

            yield f"Model registered as: {registered_name}\n"
        except Exception as e:
            yield f"\nError: {str(e)}"

    return StreamingResponse(generate_logs(), media_type="text/plain")

@router.post("/models/ollama/start/{model_name:path}",
    summary="Deploy/Start Model",
    description="""
    Initiates the loading (or downloading) of a specific model into the Ollama engine. 
    This is a **streaming endpoint** that provides real-time progress logs.
    """,
    response_description="A text stream of the CLI execution logs.",    
    dependencies=[Depends(current_active_superuser)])
async def start_model(
    model_name: str,
    session: AsyncSession = Depends(get_async_session)):
    """
    Starts/Pulls a model. 
    Uses StreamingResponse to provide real-time logs.
    """
    async def generate_logs():
        try:
            yield f"Initiating load for: {model_name}\n"
            
            # pull() ensures the model is on disk. 
            # stream=True returns an async generator.
            pull_stream = await client.pull(model=model_name, stream=True)

            async for part in pull_stream:
                # 'part' is a dict containing status, total, and completed bytes
                status = part.get('status', '')
                progress = ""
                
                if 'completed' in part and 'total' in part:
                    p = (part['completed'] / part['total']) * 100
                    progress = f" ({p:.2f}%)"
                
                yield f"{status}{progress}\n"

            # After pulling, stop LLM of embedding model (only one LLM)
            running = await client.ps()
            
            for model in running.models:
                # parse ollama to huggingface model id
                ollama_model_id = model.model 
                ollama_model_id_tokens = ollama_model_id.split("/")
                huggingface_model_id = ollama_model_id_tokens[1] + "/" + ollama_model_id_tokens[2].split(":")[0]

                print(f"Running Model ID: {huggingface_model_id}")

                # check the model info
                model_info = await get_hf_model_info(huggingface_model_id)
                
                # Only stop the LLM model (only one LLM must be running at the same time near the embedded model)
                if model_info["pipeline_tag"] == "text-generation" or model_info["pipeline_tag"] == "image-text-to-text":
                    yield f"Initiating unload for: {ollama_model_id}\n"
                    
                    # Sending keep_alive=0 (integer) tells Ollama to evict the model immediately
                    # We don't need a prompt or real generation here
                    await client.generate(model=ollama_model_id, prompt="", keep_alive=0)
                    
                    yield f"Success: {ollama_model_id} has been removed from memory.\n"
                    yield f"VRAM/RAM resources freed."

                    break                    
            
            # After pulling, we 'warm' the model (load into VRAM)
            yield f"Loading {model_name} into memory...\n"
            await client.generate(model=model_name, prompt="", keep_alive=-1)
            
            # set running model for next chat
            set_active_model(model_name)

            # persist running model for next restart
            await save_key(
                key=ACTIVE_LLM_MODEL_KEY,
                payload=SettingUpdatePayload(value=model_name),
                session=session)

            yield f"\nModel {model_name} is ready."
        except Exception as e:
            yield f"\nError: {str(e)}"

    return StreamingResponse(generate_logs(), media_type="text/plain")

@router.post("/models/ollama/stop/{model_name:path}",
    summary="Unload/Stop Model",
    description="""
    Forces the Ollama server to **unload a model from VRAM**. 
    This sets the `keep_alive` parameter to 0, immediately freeing up hardware resources.
    """,
    response_description="JSON confirmation of the stop command status.",    
    dependencies=[Depends(current_active_superuser)])
async def stop_model(model_name: str):
    """
    Unloads a model from Ollama memory.
    """
    async def generate_logs():
        try:
            yield f"Initiating unload for: {model_name}\n"
            
            # Sending keep_alive=0 (integer) tells Ollama to evict the model immediately
            # We don't need a prompt or real generation here
            await client.generate(model=model_name, prompt="", keep_alive=0)
            
            yield f"Success: {model_name} has been removed from memory.\n"
            yield f"VRAM/RAM resources freed."
        except Exception as e:
            yield f"\nError: {str(e)}"

    return StreamingResponse(generate_logs(), media_type="text/plain")  