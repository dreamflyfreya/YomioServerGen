import asyncio
import websockets

async def send_audio(audio_file_path, speaker, format, channels, rate):
    async with websockets.connect("ws://localhost:40432/get_response") as websocket:
        # Send audio format information
        format_info = f"{rate},{channels},{format}"
        await websocket.send(format_info)

        # Send audio data
        with open(audio_file_path, "rb") as audio_file:
            while True:
                chunk = audio_file.read(1024)
                if not chunk:
                    await websocket.send("END")
                    break
                await websocket.send(chunk)

        # Receive resulting audio
        resulting_audio = b""
        while True:
            chunk = await websocket.recv()
            if not chunk:
                break
            resulting_audio += chunk

        return resulting_audio

async def main():
    audio_file_path = "/Users/bytedance/Downloads/output.wav"
    speaker = "八重神子_ZH"
    format = 8
    channels = 1
    rate = 44100

    resulting_audio = await send_audio(audio_file_path, speaker, format, channels, rate)

    with open("resulting_audio.wav", "wb") as output_file:
        output_file.write(resulting_audio)
    print("Resulting audio saved as 'resulting_audio.mp3'")

if __name__ == "__main__":
    asyncio.run(main())