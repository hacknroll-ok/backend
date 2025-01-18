import asyncio
import websockets
import base64
import json

async def send_png_file():
    uri = "ws://127.0.0.1:8000/ws"
    file_path = "ss.png"

    async with websockets.connect(uri) as websocket:
        while True:
            # Read the file as binary data
            with open(file_path, "rb") as f:
                binary_data = f.read()

            print("Sending binary data...")
            # Encode the binary data to base64 to send as a text message
            encoded_image = base64.b64encode(binary_data).decode('utf-8')

            # Create the message as a dictionary and serialize it to JSON
            message = {
                "room_number": 123,
                "image": encoded_image
            }
            # Send the binary data
            
            await websocket.send(json.dumps(message))
            print(json.dumps(message))
            print(f"Sent file: {file_path}")

            # Receive a response
            response = await websocket.recv()
            
            # Check if the response is in binary
            if isinstance(response, bytes):
                print(f"Received binary data: {len(response)} bytes")
                
                # Optionally, save the binary data to a file (e.g., image.jpg)
                with open("received_image.jpg", "wb") as f:
                    f.write(response)
            else:
                print(f"Server response: {response}")

            # Optionally, wait for some time before sending again
            await asyncio.sleep(1)  # Sleep for 1 second or adjust based on your need

asyncio.run(send_png_file())
