import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import keyboard
from playsound import playsound
import winsound
import asyncio
import time
class mymain:
    can = True
    othercan = True
    sleepcount = 0
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
mymain.can = True
mymain.othercan = True
if not cap.isOpened():
    raise IOError ("Cannot open Camera!")
async def makecan(delay):
    await asyncio.sleep(delay)
    mymain.can = True
async def main():
    mymain.can = True
    mymain.othercan = True
    mymain.sleepcount = 0
    while True:

        color = (0,0,255)
        try:
            if(mymain.sleepcount ==0):
                mymain.sleepcount -=1
            ret,frame = cap.read()
            result = DeepFace.analyze(frame,actions = ['emotion'])
            emotions = result['emotion']
            emote= result['dominant_emotion']
            DisplayText = ''
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray,1.1,4)
            for(x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255,0),2)
                gray_face = gray[y:y+h, x:x+w]
                face = frame[y:y+h, x:x+w]
                eyes = eyeCascade.detectMultiScale(gray)
                z= 0
                for (ex,ey,ew,eh) in eyes: 
                    if ex > x and ew+x < x+w and ey > y and ey+eh < y+h:
                        z +=1
                        cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(255,225,255),2)
                if(z < 2):
                    
                    mymain.sleepcount +=2
                else:
                    mymain.sleepcount = 0
            x = 0
            if emotions['angry']>20 and emotions['angry'] <40:
                x+= 1
            if emotions['disgust']>.002 and emotions['disgust'] <.016:
                x+=1
            if emotions['fear']>.5 and emotions['fear'] <2:
                x+= 1
            if emotions['happy']>0 and emotions['happy'] <.006:
                x+=1
            if emotions['sad']>0 and emotions['sad'] <.01:
                x+= 1
            if emotions['surprise']>.002 and emotions['surprise'] <.01:
                x+=1
            if emotions['neutral']>15 and emotions['neutral'] < 50:
                x+=1
            if x >=2:
                DisplayText = 'Are you Confused? Ask a question'
                color = (255,0,0)
        except ValueError:
            DisplayText = "YOU ARE NOT LOOKING AT THE CAMERA!"
        font = cv2.FONT_HERSHEY_PLAIN
        x = cv2.waitKey(2)&0xFF
        if(mymain.sleepcount >=15):
            DisplayText = "Open your eyes! Look at the camera"
        cv2.putText(frame, DisplayText, (50,50), font, 1, color, 2, cv2.LINE_4)
        
        cv2.imshow('Original video',frame)
        if(DisplayText ==  "YOU ARE NOT LOOKING AT THE CAMERA!" or mymain.sleepcount >=15):
            if(mymain.can):
                winsound.PlaySound(r'Alarm.wav', winsound.SND_ASYNC)
                mymain.can = False
                await makecan(2)
        if x == ord('q'):
            break
        
asyncio.run(main())
cap.release()
cv2.destroyAllWindows()
