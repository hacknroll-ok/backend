from fastapi import FastAPI, WebSocket, File, UploadFile, WebSocketDisconnect, WebSocketException
from websockets.exceptions import ConnectionClosed
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
from ConnectionManager import ConnectionManager
import traceback


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
manager = ConnectionManager()
responseCount = 0
roundTracker = { 1: [], 2: [], 3: [], 4: [], 5:[]}
users = []
roomsAndUsers = {}
websockets_lock = asyncio.Lock()
userId = 0
playerCounter = 0
websockets = []
roundNumber = 1
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
    img_resized = ImageOps.fit(img, (100, 100), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

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

@app.post("/api/game")
async def updateScore(userId: int, isCorrect: bool):
    global responseCount
    if isCorrect:
        for user in users:
            if user["id"] == userId:
                user["score"] += 1
                break
    responseCount += 1
    # if responseCount == 5:
        # async with websockets_lock:
        #     for ws in websockets:
        #         try:
        #             await ws.send_text(json.dumps(users))  # Send the player list
        #             if len(users) == 5:
        #                 await ws.send_text("current round ended")
        #         except WebSocketDisconnect:
        #             disconnected_websockets.append(ws)

        #     # Remove disconnected websockets
        #     for ws in disconnected_websockets:
        #         websockets.remove(ws)



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global categories
    global websockets
    global manager
    randomCategoryNumber = random.randint(0, len(categories) - 1)

    # disconnected_websockets = []
    await manager.connect(websocket)
    # async with websockets_lock:
    #     CLIENTS.add(websocket)
    # print(CLIENTS)

    

    try:
        await manager.broadcast(json.dumps(users));
    except Exception as e:
        print(f"Exception type: {type(e).__name__}")
        print(e)
        manager.disconnect(websocket)
        print(manager.getActiveConnections())
    
    if len(users) == 5:
        try: 
            await manager.broadcast(json.dumps({
                "type": "newTurn",
                "round": 1,
                "playerDrawing": 0,  # The first player starts drawing
                "drawingSubject": categories[randomCategoryNumber],
            }))
        
        except Exception as e:
            print(f"Exception type: {type(e).__name__}")
            print(e)
            manager.disconnect(websocket)
            print(manager.getActiveConnections())



        
    

    # Initial game setup
    # async with websockets_lock:
    #     for ws in websockets:
    #         try:
    #             # await ws.send_text(json.dumps(users))  # Send the player list
    #             # websocket.send_text(json.dumps(users))

    #         except WebSocketDisconnect:
    #             disconnected_websockets.append(ws)

    #     # Remove disconnected websockets
    #     for ws in disconnected_websockets:
    #         websockets.remove(ws)

    try:
        while True:
            global playerCounter
            global roundNumber
            global roundTracker

            # Receive image data from WebSocket 

            data = await manager.receive(websocket)
            # Create a PIL image from the byte data
            if ("bytes" in data):
                data = data["bytes"]
                # print(data)                
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
                
                
                await manager.broadcast(f"Prediction: {categories[predicted_class]}")
                await manager.broadcast_image(data)
  

            elif "text" in data:
                # coordinateData = data["text"]
                # coordinateData = json.loads(coordinateData)
                # # print(coordinateData)
                # await websocket.send_text(json.dumps(coordinateData))
                convertedData = json.loads(data["text"])
                print(convertedData)
                if "type" in convertedData:
                    if convertedData["type"] == "disconnect":
                        del users[convertedData[userId]]
                        try:
                            await manager.broadcast(json.dumps(users));
                        except Exception as e:
                            print(f"Exception type: {type(e).__name__}")
                            print(e)
                            print(manager.getActiveConnections())
                            manager.disconnect(websocket)
                            

                    if convertedData["type"] == "roundTracking":
                        # if all rounds have ended, reset the game
                        if len(roundTracker[5]) == 5:
                            roundTracker = { 1: [], 2: [], 3: [], 4: [], 5:[]}
                        else:
                            print(roundTracker)
                            if (len(roundTracker[convertedData["round"]]) < 4):
                                roundTracker[convertedData["round"]].append({"playerId": convertedData["playerId"], "guess": convertedData["guess"]})
                                print(roundTracker)
                                randomCategoryNumber = random.randint(0, len(categories) - 1)

                                # if all other players except the one drawing have submitted their answers, go on to the next turn
                                if len(roundTracker[convertedData["round"]]) == 4:
                                    playerCounter+= 1
                                    roundNumber+=1
                                    try: 
                                        await manager.broadcast(json.dumps({
                                            "type": "newTurn",
                                            "round": roundNumber,
                                            "playerDrawing": playerCounter,
                                            "drawingSubject": categories[randomCategoryNumber]
                                        }))
                                    except WebSocketDisconnect:
                                        print("webscocket disconnect")
                                        manager.disconnect(websocket)
                                   
                                    # if playerCounter == 4:
                                    #     playerCounter = 0
                                    #     roundNumber += 1
  
                    if convertedData["type"] == "drawing":
                        try: 
                            await manager.broadcast(data["text"])
                        except WebSocketDisconnect:
                            print("webscocket disconnect")
                            manager.disconnect(websocket)
            
           
    except Exception as e:
        print(f"Exception type: {type(e).__name__}")
        # if (e == 'Cannot call "receive" once a disconnect message has been received.'):
        #     print(e)
        # else:
        #     # print(e)
        #     traceback.print_exc()
        #     raise RuntimeError(e)            
        # manager.disconnect(websocket)
        print(e)

