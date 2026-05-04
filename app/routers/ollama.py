
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Depends, HTTPException
from ollama import AsyncClient

from app.config import settings
from app.routers.auth import current_active_superuser, current_active_user

router = APIRouter(
    prefix="/api/v1",
    tags=["ollama"],
)

# Initialize Ollama client
client = AsyncClient(host=settings.ollama_host)

@router.get(
    "/models/ollama",
    summary="List all Ollama Models pulled",
    description="""
    Retrieves a list of AI models currently pulled on the local Ollama server. Useful for monitoring resource usage.
    """,
    response_description="List of currently pulled models.",    
    dependencies=[Depends(current_active_user)])
async def get_pulled_ollama_models(search: str = None):
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

@router.delete(
    "/models/ollama/delete/{model_name:path}",
    summary="Delete Ollama Model",
    description="Permanently removes the model files from the local Ollama storage.",
    dependencies=[Depends(current_active_superuser)] 
)
async def delete_ollama_model(model_name: str):
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
    
@router.post("/models/ollama/pull/{model_name:path}",
    summary="Pull Ollama Model",
    description="""
    Initiates the pulling (or downloading) of a specific model into the Ollama engine. 
    This is a **streaming endpoint** that provides real-time progress logs.
    """,
    response_description="A text stream of the CLI execution logs.",    
    dependencies=[Depends(current_active_superuser)]             )
async def pull_model(model_name: str):
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
        except Exception as e:
            yield f"\nError: {str(e)}"

    return StreamingResponse(generate_logs(), media_type="text/plain")

@router.post(
    "/models/ollama/start/{model_name:path}",
    summary="Deploy/Start Ollama Model",
    description="""
    Initiates the loading (or downloading) of a specific model into the Ollama engine. 
    This is a **streaming endpoint** that provides real-time progress logs.
    """,
    response_description="A text stream of the CLI execution logs.",    
    dependencies=[Depends(current_active_superuser)])
async def start_ollama_model(model_name: str):
    """
    Starts/Pulls a model. 
    Uses StreamingResponse to provide real-time logs.
    """
    async def generate_logs():
        #client = AsyncClient(host=settings.ollama_host)
        
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

            # After pulling, we 'warm' the model (load into VRAM)
            yield f"Loading {model_name} into memory...\n"
            await client.generate(model=model_name, prompt="", keep_alive=-1)
            
            yield f"\nModel {model_name} is ready."
        except Exception as e:
            yield f"\nError: {str(e)}"

    return StreamingResponse(generate_logs(), media_type="text/plain")

@router.post(
    "/models/ollama/stop/{model_name:path}",
    summary="Unload/Stop Ollama Model",
    description="""
    Forces the Ollama server to **unload a model from VRAM**. 
    This sets the `keep_alive` parameter to 0, immediately freeing up hardware resources.
    """,
    response_description="JSON confirmation of the stop command status.",    
    dependencies=[Depends(current_active_superuser)])
async def stop_ollama_model(model_name: str):
    """
    Unloads a model from Ollama memory.
    """
    async def generate_logs():
        #client = AsyncClient(host=settings.ollama_host)

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