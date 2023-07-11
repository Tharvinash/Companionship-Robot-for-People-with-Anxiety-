import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play

# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()

def listen_and_process():
    while True:  # Keep running until a return statement is encountered
        # Use the microphone as source for input.
        with sr.Microphone() as source:
            print("Speak something...")
            r.adjust_for_ambient_noise(source)  # Adjust for ambient noise

            try:
                # Set timeout for listening and record the audio
                audio = r.listen(source, timeout=30)
                print("Recognizing...")
                # Using google speech recognition
                text = r.recognize_google(audio)
                print(f"You said: {text}")

                if "i am sad" in text.lower() or "i'm sad" in text.lower():
                    song = AudioSegment.from_file("/home/pi/audio.mp3")  # Use the path to your audio file
                    play(song)
                    print("Don't worry, everything will be alright!")  # User said 'I am sad', print "Don't worry"
                elif "goodbye" in text.lower() or "exit" in text.lower():
                    print("Goodbye!")
                    return  # Terminates the function
            except sr.WaitTimeoutError:
                print("Timeout error: No speech detected.")
                continue  # Go to the next iteration of the loop
            except sr.UnknownValueError as e:
                print("Sorry, I couldn't understand what you said.")
                print(f"Error: {str(e)}")
                continue  # Go to the next iteration of the loop
            except sr.RequestError as e:
                print(f"Speech recognition error: {e}")
                print(f"Error: {str(e)}")
                continue  # Go to the next iteration of the loop

# Call the function to start
listen_and_process()