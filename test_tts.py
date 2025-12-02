import pyttsx3
import os

try:
    print("Initializing pyttsx3...")
    engine = pyttsx3.init()
    print("Testing speech...")
    engine.save_to_file('Hello world', 'test.wav')
    engine.runAndWait()
    print("Speech saved to test.wav")
    if os.path.exists('test.wav'):
        print(f"Success! File size: {os.path.getsize('test.wav')} bytes")
    else:
        print("Error: File not created")
except Exception as e:
    print(f"Error: {e}")
