from fastapi import FastAPI, WebSocket, File, UploadFile
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

app = FastAPI()
users = []
roomsAndUsers = {}

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
    img_resized = ImageOps.fit(img, (28, 28), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    img_resized = img_resized.convert('L')

    # Convert the image to a NumPy array
    img_array = np.array(img_resized)

    # Expand dimensions to add a batch dimension for model input
    img_array = np.expand_dims(img_array, axis=0)

    # Save the processed image for verification (optional)
    img_resized.save('./images/processed_image.png')

    return img_array

class User(BaseModel):
    id: int
    name: str
    score: int
    
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

@app.post("/api/users/{room_id}")
async def createUser(user: User, room_id: str):
    if room_id not in roomsAndUsers:
        roomsAndUsers[room_id] = []
    roomsAndUsers[room_id].append(user)
    return {"message": "user created"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Receive image data from WebSocket
        data = await websocket.receive_bytes()
        
        # Create a PIL image from the byte data
        img = Image.open(BytesIO(data))

        # Check if the image has an alpha channel (RGBA)
        if img.mode == 'RGBA':
            # Create a white background image the same size as the original
            background = Image.new('RGB', img.size, (255, 255, 255))
            
            # Paste the original image onto the white background, using the alpha channel as mask
            background.paste(img, mask=img.split()[3])  # Use the alpha channel as mask
            
            img = background

        # Define the file path to save the image
        file_path = os.path.join("images", "uploaded_image.png")

        # Save the image with a white background
        img.save(file_path)

        # file_path = os.path.join("images", "uploaded_image.png")

        # # Save the received image
        # with open(file_path, "wb") as f:
        #     f.write(data)
        
        # print(f"Received data of size: {len(data)} bytes")
        
        # Load and preprocess the image
        processed_image = preprocess_image(file_path)

        
        # # Set the input tensor for the model
        # interpreter.set_tensor(input_details[0]['index'], processed_image)
        
        # # Run inference
        # interpreter.invoke()

        # # Get the output tensor
        # output_data = interpreter.get_tensor(output_details[0]['index'])

        # # Get the class with the highest probability
        # prediction = np.argmax(output_data)
        
        prediction = model.predict(processed_image)[0]
        predicted_class = np.argmax(prediction)

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

        # Send prediction result back to WebSocket
        await websocket.send_text(f"Prediction: {categories[predicted_class]}")