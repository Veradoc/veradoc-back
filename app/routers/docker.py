import platform
import docker
from fastapi import APIRouter, HTTPException, Depends
from app.routers.auth import current_active_superuser

from app.config import settings
from app.utils.const import *

router = APIRouter(
    prefix="/api/v1",
    tags=["docker"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)

DEF_STACK="veradoc-web"

def get_docker_client():
    """
    Returns a Docker client configured for the current Operating System.
    """
    system = platform.system()
    
    try:
        if system == "Windows":
            # Windows uses Named Pipes for Docker Desktop
            return docker.DockerClient(base_url='npipe:////./pipe/docker_engine')
        
        elif system == "Darwin":  # macOS
            # Check for the specific Mac user-level socket found in Solution 2
            mac_socket = f"unix://{os.path.expanduser('~')}/.docker/run/docker.sock"
            if os.path.exists(mac_socket.replace('unix://', '')):
                return docker.DockerClient(base_url=mac_socket)
            # Fallback to default
            return docker.from_env()
            
        else:
            # Linux and others (default behavior)
            return docker.from_env()
            
    except Exception as e:
        print(f"Error connect∫∫ing to Docker on {system}: {e}")
        raise e
    
# Initialize the Docker client
# This connects to the local Docker daemon via the default socket
try:
    # cross-platform docker client
    client = get_docker_client()  
except Exception as e:
    print(f"Error connecting to Docker: {e}")

@router.get("/containers",
    summary="List Docker Containers",
    description="""
    List Docker Containers into the Docker engine.
    """,
    response_description="A text stream of the CLI execution logs.",    
    dependencies=[Depends(current_active_superuser)])
async def list_containers(stack_name: str = None, all: bool = True):
    """
    List containers, optionally filtered by Docker Compose stack name.
    """
    try:
        filters = {}
        #if stack_name:
            # Docker Compose automatically labels containers with the project name
        #    filters = {'label': f"com.docker.compose.project={stack_name}"}
        
        filters = {'label': f"com.docker.compose.project={DEF_STACK}"}

        # We pass the filters dictionary to the list method
        containers = client.containers.list(all=all, filters=filters)
        
        container_data = []
        for c in containers:
            # Reusing the logic above
            networks = c.attrs['NetworkSettings']['Networks']
            ip = next(iter(networks.values()))['IPAddress'] if networks else "N/A"

            ports = c.attrs['NetworkSettings']['Ports']
            # Create a list like ["1883:1883", "9001:9001"]
            p_list = [f"{m[0]['HostPort']}->{p.split('/')[0]}" for p, m in ports.items() if m]

            container_data.append({
                "id": c.short_id,
                "name": c.name,
                "stack": c.labels.get("com.docker.compose.project", "standalone"),
                "image": c.image.tags[0] if c.image.tags else "untagged",
                "health": c.health,
                "attrs": c.attrs,
                "ip": ip or "N/A",
                "ports": ", ".join(p_list) if p_list else "-",
                "status": c.status
            })
            
        return {"stack": stack_name or "all", "containers": container_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/containers/{container_id}/start",
    summary="Start a Docker Container",
    description="""
    Docker Containers started into the Docker engine.
    """,
    response_description="A text stream of the CLI execution logs.",             
    dependencies=[Depends(current_active_superuser)])
async def start_container(container_id: str):
    """Start a specific container by ID or Name."""
    try:
        container = client.containers.get(container_id)
        container.start()

        return {"message": f"Container {container_id} started successfully"}
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail="Container not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/containers/{container_id}/stop",
    summary="Stop a Docker Container",
    description="""
    Docker Containers stoped into the Docker engine.
    """,             
    dependencies=[Depends(current_active_superuser)])
async def stop_container(container_id: str):
    """Stop a specific container by ID or Name."""
    try:
        container = client.containers.get(container_id)
        container.stop()

        return {"message": f"Container {container_id} stopped successfully"}
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail="Container not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    