from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket, userId: int):
        await websocket.accept()
        self.active_connections[websocket] = userId

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            userWhoLeft = self.active_connections[websocket]
            del self.active_connections[websocket]
            return userWhoLeft
        else:
            return "nobody"

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

    async def receive(self, websocket: WebSocket):
        data = await websocket.receive()
        return data
    
    async def broadcast_image(self, image: bytes):
        for connection in self.active_connections:
            await connection.send_bytes(image)

    def getActiveConnections(self):
        return self.active_connections
    


