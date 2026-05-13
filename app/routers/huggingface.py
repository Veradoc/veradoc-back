
from typing import Literal, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, Query

from huggingface_hub import HfApi, Organization, User
from huggingface_hub.errors import HfHubHTTPError
from pydantic import BaseModel

from app.routers.auth import current_active_superuser

router = APIRouter(
    prefix="/api/v1",
    tags=["huggingface"],
)

class ModelSummary(BaseModel):
    id: str
    author: Optional[Union[User, Organization]] = None
    author_type: Optional[Literal["organization", "user"]] = None
    sha: Optional[str]
    created_at: Optional[str]
    last_modified: Optional[str]
    private: bool
    disabled: Optional[bool]
    downloads: Optional[int]
    likes: Optional[int]
    pipeline_tag: Optional[str]
    tags: list[str]
 
 
class PaginatedModels(BaseModel):
    models: list[ModelSummary]
    page: int
    page_size: int
    total_fetched: int
    has_more: bool
    
hf_api = HfApi()

def get_author(author: str) -> str:
    HF_BASE_URL = "https://huggingface.co"

    """Returns (avatar_url, type) where type is 'org' or 'user'"""
    try:
        org = hf_api.get_organization_overview(author)
        
        return org, "organization"
    except HfHubHTTPError:
        pass
    
    try:
        user = hf_api.get_user_overview(author)
        user.avatar_url = HF_BASE_URL + user.avatar_url
        
        return user, "user"
    except HfHubHTTPError:
        return "", None
        
# --- HuggingFace API ---
@router.get(
    "/huggingface",
    summary="Search HuggingFace Hub models",
    description="""
    Queries the HuggingFace Model Hub for available models. 
    Results are sorted by **popularity (downloads)** to help admins identify 
    trending models for potential deployment.
    """,
    response_description="A list of model metadata objects from HuggingFace.",  
    dependencies=[Depends(current_active_superuser)])
async def list_huggingface_models(
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),    
    page_size: int = Query(default=10, ge=5, le=100, description="Items per page"),
    sort: str = "downloads", 
    filter: str = "image-text-to-text",
    search: Optional[str] = Query(default=None, description="Search query"),
):
    try:
        limit = page * page_size + page_size

        models = hf_api.list_models(
            search=search, # search text inside model name
            apps="ollama", # compatible models with Ollama
            filter="gguf", # compatible format with Ollama
            pipeline_tag=filter, # filter only multimodal to text models: feature-extraction
            sort=sort, # sort by downloads, likes or creation date
            limit=limit # pagination limit                  
        )

        # must filter by compatible ollama models
        compatible_models = models

        all_models = list(compatible_models)
        start = (page - 1) * page_size
        end = start + page_size
        page_models = all_models[start:end]
        has_more = len(all_models) > end
        
        results = []
        for m in page_models:
            try: 
                # get author from model model id to get icons
                authorName = m.id.split("/")[0]
                author, author_type = get_author(authorName)

                results.append(ModelSummary(
                    id=m.id,
                    author=author,
                    author_type=author_type,
                    sha=getattr(m, "sha", None),
                    created_at=str(m.created_at) if getattr(m, "created_at", None) else None,
                    last_modified=str(m.last_modified) if getattr(m, "last_modified", None) else None,
                    private=m.private or False,
                    disabled=getattr(m, "disabled", None),
                    downloads=getattr(m, "downloads", None),
                    likes=getattr(m, "likes", None),
                    pipeline_tag=getattr(m, "pipeline_tag", None),
                    tags=getattr(m, "tags", []) or [],
                ))
            except Exception as e:
                pass

        return PaginatedModels(
            models=results,
            page=page,
            page_size=page_size,
            total_fetched=len(all_models),
            has_more=has_more,
        )
    except HfHubHTTPError as e:
        raise HTTPException(status_code=502, detail=f"HuggingFace API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/model/{model_id:path}")
async def get_hf_model_info(model_id: str):
    """
    Recover detailed metadata for a Hugging Face model.
    Example model_id: 'facebook/detr-resnet-50' or 'gpt2'
    """
    try:
        # Fetch model data using the SDK
        info = hf_api.model_info(model_id)
        
        return {
            "model_id": info.modelId,
            "author": info.author,
            "last_modified": info.lastModified,
            "downloads": info.downloads,
            "likes": info.likes,
            "tags": info.tags,
            "pipeline_tag": info.pipeline_tag,
            "private": info.private,
            "library_name": info.library_name
        }
    except Exception as e:
        # Handle 404 or connection issues
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found or SDK error: {str(e)}")    