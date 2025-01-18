from fastapi import FastAPI, WebSocket, File, UploadFile, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import string
import random
from pydantic import BaseModel
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from io import BytesIO
from typing import Optional
import json
import asyncio



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
websockets_lock = asyncio.Lock()
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


# # TensorFlow Lite model loading
# interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
# interpreter.allocate_tensors()

# # Get model input and output details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

model = load_model('./model/model.keras') 

def preprocess_image(img_path):
     # Load the image in grayscale mode
    img = Image.open(img_path)

    # Resize the image to 28x28 pixels, preserving the aspect ratio and padding with white
    img_resized = ImageOps.fit(img, (50, 50), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    img_resized = img_resized.convert('L')

    # Convert the image to a NumPy array
    img_array = np.array(img_resized)

    # Expand dimensions to add a batch dimension for model input
    img_array = np.expand_dims(img_array, axis=0)

    # Save the processed image for verification (optional)
    img_resized.save('./images/processed_image.png')

    return img_array

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

    return {
        "message": "user created",
        "userId": userId - 1
    }



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global categories
    async with websockets_lock:
        websockets.append(websocket)
    randomCategoryNumber = random.randint(0, len(categories) - 1)
    disconnected_websockets = []

    # Initial game setup
    async with websockets_lock:
        for ws in websockets:
            try:
                await ws.send_text(json.dumps(users))  # Send the player list
                if len(users) == 5:
                    await ws.send_text(json.dumps({
                        "round": 1,
                        "playerDrawing": 0,  # The first player starts drawing
                        "drawingSubject": categories[randomCategoryNumber],
                    }))
            except WebSocketDisconnect:
                disconnected_websockets.append(ws)

        # Remove disconnected websockets
        for ws in disconnected_websockets:
            websockets.remove(ws)

    try:
        while True:
            # Receive image data from WebSocket
            data = await websocket.receive_bytes()
            global playerCounter
            global roundNumber
            randomCategoryNumber = random.randint(0, len(categories) - 1)
            
            # Broadcast to all players whose turn it is
            for ws in websockets:
                await ws.send_text(json.dumps({
                    "type": "newTurn",
                    "round": roundNumber,
                    "playerDrawing": playerCounter,
                    "drawingSubject": categories[randomCategoryNumber]
                }))

            # Move to the next player
            playerCounter += 1
            if playerCounter == len(users):  # After the last player, start a new round
                roundNumber += 1
                playerCounter = 0

            # Create a PIL image from the byte data
            img = Image.open(BytesIO(data))

            # Check if the image has an alpha channel (RGBA)
            if img.mode == 'RGBA':
                # Create a white background image the same size as the original
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # Use the alpha channel as mask
                img = background

            # Define the file path to save the image
            file_path = os.path.join("images", "uploaded_image.png")
            img.save(file_path)

            # Preprocess the image
            processed_image = preprocess_image(file_path)

            # Run prediction (for the player drawing)
            prediction = model.predict(processed_image)[0]
            predicted_class = np.argmax(prediction)

            # Send the prediction result to the current player (drawing)
            await websocket.send_text(f"Prediction: {categories[predicted_class]}")

    except WebSocketDisconnect:
        async with websockets_lock:
            websockets.remove(websocket)
