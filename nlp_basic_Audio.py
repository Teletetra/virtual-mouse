import os 
import win32com.client as wincl
speak=wincl.Dispatch("SAPI.SpVoice")
while True:
    text=input("ENTER text to speak (or exit to quit):")
    if text.lower()=="exit":
        speak.speak("bye bye ")
        break
    speak.speak(text)