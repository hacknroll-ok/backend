from fastapi import FastAPI, WebSocket, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import string
import random
from pydantic import BaseModel
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Optional
import json



app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
users = []
roomsAndUsers = {}

userId = 0
playerCounter = 0
websockets = []
roundNumber = 0
categories =[
    "ant",
    "bat",
    "bear",
    "bee",
    "butterfly",
    "camel",
    "cat",
    "cow",
    "crab",
    "crocodile",
    "dog",
    "dolphin",
    "dragon",
    "duck",
    "elephant",
    "fish",
    "flamingo",
    "frog",
    "giraffe",
    "hedgehog",
    "horse",
    "kangaroo",
    "lion",
    "lobster",
    "monkey",
    "mosquito",
    "mouse",
    "octopus",
    "owl",
    "panda",
    "parrot",
    "penguin",
    "rabbit",
    "raccoon",
    "rhinoceros",
    "scorpion",
    "sea turtle",
    "shark",
    "sheep",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "swan",
    "tiger",
    "whale",
    "zebra"
]


# TensorFlow Lite model loading
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocessing function for the image
def preprocess_image(image: Image.Image):
    image = image.convert("L")
    image = image.resize((28, 28))  # Resize to model input size
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension if needed
    return image.astype(np.float32)

class Username(BaseModel):
    name: str
    # score: int
    # num: int

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/rooms")
async def createRoom():
    room_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    return {        
        "message": "room created",
        "Room Number": room_id    
    }

@app.get("/api/rooms")
async def getRooms():
    room_list = list(roomsAndUsers.keys()) 
    return {"rooms": room_list}

# @app.post("/api/users/{room_id}")
# async def createUser(user: User, room_id: str):
#     if room_id not in roomsAndUsers:
#         roomsAndUsers[room_id] = []
#     roomsAndUsers[room_id].append(user)
#     return {"message": "user created"}

@app.post("/api/users")
async def createUser(username: Username):
    global userId
    newUser = {"id": userId, "name": username.name, "score": 0}
    users.append(newUser);    
    userId += 1

    for ws in websockets:
        await ws.send_text(json.dumps(users))
        if (len(users) == 5):
            await ws.send_text(json.dumps({
                "round": 1,
                "playerDrawing": 0,
                "drawingSubject": "apple",
            }))
    return {
        "message": "user created",
        "userId": userId - 1
    }



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websockets.append(websocket)
    while True:
        # Receive image data from WebSocket
        data = await websocket.receive_bytes()
        global playerCounter
        global roundNumber
        global categories
        playerCounter += 1
        randomCategoryNumber = random.randint(0, len(categories) - 1)
        await websocket.send_text(json.dumps({
                "round": roundNumber,
                "playerDrawing": 0,
                "drawingSubject": randomCategoryNumber,
        }))
        if (playerCounter == 4):
            roundNumber += 1
        
        file_path = os.path.join("images", "uploaded_image.png")

        # Save the received image
        with open(file_path, "wb") as f:
            f.write(data)
        
        print(f"Received data of size: {len(data)} bytes")
        
        # Load and preprocess the image
        image = Image.open(BytesIO(data))
        processed_image = preprocess_image(image)
        
        # Set the input tensor for the model
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        
        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Get the class with the highest probability
        prediction = np.argmax(output_data)
        
        # Send prediction result back to WebSocket
        await websocket.send_text(f"Prediction: {prediction}")
        



        #send current player who is drawing
        
