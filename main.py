from fastapi import FastAPI, WebSocket
import string
import random
from pydantic import BaseModel
import os

app = FastAPI()
users = []
roomsAndUsers = {}
class User(BaseModel):
    id: int
    name: str
    score: int
    
# class Room(BaseModel):
#     id: str
    
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/rooms")
async def createRoom():
    # print(room);
    room_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5));
    return {        
        "message": "room created",
        "Room Number": room_id    
    }

@app.get("/api/rooms")
async def getRooms():
    room_list = list(roomsAndUsers.keys()); 
    return {"rooms": room_list}

@app.post("/api/users/{room_id}")
async def createUser(user: User, room_id: str):
    print(user);
    roomsAndUsers[room_id].append(user);
    return {"message": "user created"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_bytes()
        file_path = os.path.join("images", "uploaded_image.png")
        with open(file_path, "wb") as f:
            f.write(data)
        # await websocket.send_text(f"Message text was: {data}")
        print(f"Received data of size: {len(data)} bytes")
        # await websocket.send_text(f"File received and saved to {file_path}")
