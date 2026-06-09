from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.utils.websocket_manager import ws_manager

router = APIRouter(
    prefix="/api/v1/ws",
    tags=["websockets"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)

@router.websocket("/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await ws_manager.connect(websocket, user_id)

    try:
        while True:
            # mantener conexión viva; puedes procesar mensajes del cliente aquí
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, user_id)