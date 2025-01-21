from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

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
    


