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
from pathlib import Path


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
roundTracker = {}
answerCounter = 0
wrongCounter = 0
users = []
roomsAndUsers = {}
# websockets_lock = asyncio.Lock()
userId = 0
playerCounter = 0
websockets = []
roundNumber = 1
allUsersGotWrong = False
aiGuessCorrect = False
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
MODEL_PATH = Path(__file__).resolve().parent / "model" / "model50.keras"
print(MODEL_PATH)
PROCESSED_IMAGE_PATH = Path(__file__).resolve().parent / "images" / "processed_image.png"
UPLOADED_IMAGE_PATH = Path(__file__).resolve().parent / "images" / "uploaded_image.png"
model = load_model(MODEL_PATH)

def preprocess_image(img_path):
     # Load the image in grayscale mode
    img = Image.open(img_path)

    # Resize the image to 50x50 pixels, preserving the aspect ratio and padding with white
    img_resized = ImageOps.fit(img, (50, 50), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    img_resized = img_resized.convert('L')

    # Convert the image to a NumPy array
    img_array = np.array(img_resized)

    # Expand dimensions to add a batch dimension for model input
    img_array = np.expand_dims(img_array, axis=0)

    # Save the processed image for verification (optional)
    img_resized.save(PROCESSED_IMAGE_PATH)

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

randomCategoryNumber = random.randint(0, len(categories) - 1)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global categories
    global websockets
    global manager
    global users
    global userId
    global randomCategoryNumber
    


    await manager.connect(websocket, userId-1)


    

    try:
        await manager.broadcast(json.dumps(users));
    except WebSocketDisconnect:
        # print(f"Exception type: {type(e).__name__}")
        # print(e)
        userWhoLeft = manager.disconnect(websocket)
        print(f"user left: {userWhoLeft}")
        if userWhoLeft != "nobody":
            users = list(filter(lambda user: user["id"] != userWhoLeft, users))
            await manager.broadcast(json.dumps(users));
        print(users)
        print(manager.getActiveConnections())
        return
    
    if len(users) > 1:
        try: 
            await manager.broadcast(json.dumps({
                "type": "newTurn",
                "round": 1,
                "playerDrawing": 0,  # The first player starts drawing
                "drawingSubject": categories[randomCategoryNumber],
            }))
        
        except WebSocketDisconnect:
            # print(f"Exception type: {type(e).__name__}")
            # print(e)
            # manager.disconnect(websocket)
            # print(manager.getActiveConnections())
            userWhoLeft = manager.disconnect(websocket)
            print(f"user left: {userWhoLeft}")
            if userWhoLeft != "nobody":
                users = list(filter(lambda user: user["id"] != userWhoLeft, users))
                await manager.broadcast(json.dumps(users));
            print(users)
            print(manager.getActiveConnections())
            return




        
    

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
    #         

    try:
        while True:
            global playerCounter
            global roundNumber
            global roundTracker
            global answerCounter
            global wrongCounter
            global aiGuessCorrect
            global allUsersGotWrong

            if aiGuessCorrect and allUsersGotWrong:
                print("drawing user add 1 point")
                users[playerCounter]["score"] += 1
                aiGuessCorrect = False
                allUsersGotWrong = False
                try: 
                    await manager.broadcast(json.dumps(users))

                except WebSocketDisconnect:
                    # print(f"Exception type: {type(e).__name__}")
                    # print(e)
                    # manager.disconnect(websocket)
                    # print(manager.getActiveConnections())
                    userWhoLeft = manager.disconnect(websocket)
                    print(f"user left: {userWhoLeft}")
                    if userWhoLeft != "nobody":
                        users = list(filter(lambda user: user["id"] != userWhoLeft, users))
                        await manager.broadcast(json.dumps(users));
                    print(users)
                    print(manager.getActiveConnections())
                    return



            # Receive image data from WebSocket 
            try:
                data = await manager.receive(websocket)
            except RuntimeError as e:
                # print(e)
                # manager.disconnect(websocket);
                # print(manager.getActiveConnections());
                # return
                userWhoLeft = manager.disconnect(websocket)
                print(f"user left: {userWhoLeft}")
                if userWhoLeft != "nobody":
                    users = list(filter(lambda user: user["id"] != userWhoLeft, users))
                    await manager.broadcast(json.dumps(users));
                print(users)
                print(manager.getActiveConnections())
                return

            
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
                file_path = UPLOADED_IMAGE_PATH
                img.save(file_path)

                # Preprocess the image
                processed_image = preprocess_image(file_path)

                # Run prediction (for the player drawing)
                prediction = model.predict(processed_image)[0]
                predicted_class = np.argmax(prediction)

                # Send the prediction result to the current player (drawing)
                if (categories[randomCategoryNumber] == categories[predicted_class]):
                    aiGuessCorrect = True
                try: 
                    await manager.broadcast(f"Prediction: {categories[predicted_class]}")
                    await manager.broadcast_image(data)
                except WebSocketDisconnect:
                    # manager.disconnect(websocket)
                    # print(manager.getActiveConnections())
                    # return
                    userWhoLeft = manager.disconnect(websocket)
                    print(f"user left: {userWhoLeft}")
                    if userWhoLeft != "nobody":
                        users = list(filter(lambda user: user["id"] != userWhoLeft, users))
                        await manager.broadcast(json.dumps(users));
                    print(users)
                    print(manager.getActiveConnections())
                    return


  

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
                        except WebSocketDisconnect:
                            # print(f"Exception type: {type(e).__name__}")
                            # print(e)
                            # print(manager.getActiveConnections())
                            # manager.disconnect(websocket)
                            # return
                            userWhoLeft = manager.disconnect(websocket)
                            print(f"user left: {userWhoLeft}")
                            if userWhoLeft != "nobody":
                                users = list(filter(lambda user: user["id"] != userWhoLeft, users))
                                await manager.broadcast(json.dumps(users));
                            print(users)
                            print(manager.getActiveConnections())
                            return

                    if convertedData["type"] == "roundTracking":
                        answerCounter+=1

                        
                        # answerCounter = 0
                        # if guessing player guesses wrongly
                        if convertedData["guess"] == "wrong":
                            # add to wrong counter
                            print("player guessed wrong")
                            wrongCounter += 1
                        # if all guessing players answer wrongly and the AI manage to guess the drawing subject
                        # the drawer wins 1 point
                        if (wrongCounter == answerCounter):
                            print("all users got wrong")
                            allUsersGotWrong = True;
                            # users[playerCounter].score += 1

                        
                        
                        # playerCounter+=1
                        print(f"roundnumber: {roundNumber}")
                        print(f"playerCounter: {playerCounter}")
                        print(f"answerCounter: {answerCounter}")
                        # if the number of rounds do not exceed the available users, continue the game
                        if roundNumber < len(users):
                            if answerCounter == len(users) - 1:
                                try: 
                                    # increment values for next round
                                    roundNumber+=1
                                    randomCategoryNumber = random.randint(0, len(categories) - 1)
                                    await manager.broadcast(json.dumps({
                                            "type": "newTurn",
                                            "round": roundNumber,
                                            "playerDrawing": roundNumber-1,
                                            "drawingSubject": categories[randomCategoryNumber]
                                    }))
                                    wrongCounter = 0
                                    answerCounter = 0
                                except WebSocketDisconnect:
                                    # print("webscocket disconnect")
                                    # manager.disconnect(websocket)
                                    # return
                                    userWhoLeft = manager.disconnect(websocket)
                                    print(f"user left: {userWhoLeft}")
                                    if userWhoLeft != "nobody":
                                        users = list(filter(lambda user: user["id"] != userWhoLeft, users))
                                        await manager.broadcast(json.dumps(users));
                                    print(users)
                                    print(manager.getActiveConnections())
                                    return
                        elif roundNumber == len(users):
                            # in the last round, if all players have ans
                            if answerCounter == len(users) - 1:
                                # reset rounds for next set of players\
                                roundNumber = 1
                                answerCounter = 0
                                wrongCounter = 0
                                userId = 0
                                
                                try: 
                                    await manager.broadcast(json.dumps({
                                        "type": "end",
                                    }))
                                except WebSocketDisconnect:
                                    # print("webscocket disconnect")
                                    # manager.disconnect(websocket)
                                    # return
                                    userWhoLeft = manager.disconnect(websocket)
                                    print(f"user left: {userWhoLeft}")
                                    if userWhoLeft != "nobody":
                                        users = list(filter(lambda user: user["id"] != userWhoLeft, users))
                                        await manager.broadcast(json.dumps(users));
                                    print(users)
                                    print(manager.getActiveConnections())
                                    return
                    
                if convertedData["type"] == "drawing":
                    try: 
                        await manager.broadcast(data["text"])
                    except WebSocketDisconnect:
                        print("webscocket disconnect")
                        # manager.disconnect(websocket)
                        # return
                        userWhoLeft = manager.disconnect(websocket)
                        print(f"user left: {userWhoLeft}")
                        if userWhoLeft != "nobody":
                            users = list(filter(lambda user: user["id"] != userWhoLeft, users))
                            await manager.broadcast(json.dumps(users));
                        print(users)
                        print(manager.getActiveConnections())
                        return
                    
            
           
    except RuntimeError as e:
        print(f"Exception type: {type(e).__name__}")
        # if (e == 'Cannot call "receive" once a disconnect message has been received.'):
        #     print(e)
        # else:
        #     # print(e)
        #     traceback.print_exc()
        #     raise RuntimeError(e)            
        # manager.disconnect(websocket)
        print(e)

