from fastapi import WebSocket
from typing import DefaultDict
from collections import defaultdict

class WebSocketManager:
    def __init__(self):
        # keyed by user_id → list of active connections (multi-tab support)
        self.active: DefaultDict[str, list[WebSocket]] = defaultdict(list)

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active[str(user_id)].append(websocket)

    def disconnect(self, websocket: WebSocket, user_id: str):
        try:
            self.active[str(user_id)].remove(websocket)
        except ValueError:
            pass
        
        # Clean up empty keys so your dictionary doesn't grow indefinitely
        if not self.active[str(user_id)]:
            del self.active[str(user_id)]

    async def send_to_user(self, user_id: str, payload: dict):
        dead = []
        for ws in self.active.get(str(user_id), []):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)

        for ws in dead:
            try:
                self.active[str(user_id)].remove(ws)
            except ValueError:
                pass

    async def broadcast(self, payload: dict):
        for user_id in list(self.active):
            await self.send_to_user(user_id, payload)

ws_manager = WebSocketManager()