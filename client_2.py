import io
import requests
from pydub import AudioSegment


# URL of the server endpoint
url = "http://localhost:8000/get_response"

# Path to the audio file you want to send
audio_file_path = "/Users/bytedance/Downloads/output.wav"

# Additional parameters
speaker = "八重神子_ZH"
channels = 1
rate = 44100

# Open the audio file in binary mode
with open(audio_file_path, "rb") as file:
    # Create a dictionary containing the file and additional parameters
    files = {"audio_file": file}
    data = {"speaker": speaker, "channels": channels, "sample_rate": rate}

    # Send the POST request to the server
    response = requests.post(url, files=files, data=data)
    print(response.content)

# Check the response status code
if response.status_code == 200:
    # Create a BytesIO object from the response content
    with open('resulting.wav', 'wb') as f:
        f.write(response.content)
    print("Audio file received and saved successfully.")
else:
    print("Failed to receive the audio file.")