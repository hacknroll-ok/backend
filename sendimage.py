import asyncio
import websockets

async def send_png_file():
    uri = "ws://127.0.0.1:8000/ws"
    file_path = "ss.png"

    async with websockets.connect(uri) as websocket:
        # Read the file as binary data
        with open(file_path, "rb") as f:
            binary_data = f.read()

        # Send the binary data
        await websocket.send(binary_data)
        print(f"Sent file: {file_path}")

        # Receive a response
        response = await websocket.recv()
        print(f"Server response: {response}")

asyncio.run(send_png_file())
