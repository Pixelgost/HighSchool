import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import keyboard
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError ("Cannot open Camera!")
isDone = False
Bluff = [0, 0, 0, 0, 0, 0, 0]
Truth = [0, 0, 0, 0, 0, 0, 0]
BluffDataPoints = 0
TruthDataPoints = 0
while True:
    def back():
        isDone = True
    ret,frame = cap.read()
    result = DeepFace.analyze(frame,actions = ['emotion'])
    emotions = result['emotion']
    emote= result['dominant_emotion']
    DisplayText = ''
    if BluffDataPoints < 3 or TruthDataPoints < 3:
        DisplayText = 'Not Enough Data Points!'
    else:
        TruthGap = [0,0,0,0,0,0,0]
        BluffGap = [0,0,0,0,0,0,0]
        TruthGap[0] = abs(Truth[0]-emotions['angry'])
        TruthGap[1] = abs(Truth[1]-emotions['disgust'])
        TruthGap[2] = abs(Truth[2]-emotions['fear'])
        TruthGap[3] = abs(Truth[3]-emotions['happy'])
        TruthGap[4] = abs(Truth[4]-emotions['sad'])
        TruthGap[5] = abs(Truth[5]-emotions['surprise'])
        TruthGap[6] = abs(Truth[6]-emotions['neutral'])
        BluffGap[0] = abs(Bluff[0]-emotions['angry'])
        BluffGap[1] = abs(Bluff[1]-emotions['disgust'])
        BluffGap[2] = abs(Bluff[2]-emotions['fear'])
        BluffGap[3] = abs(Bluff[3]-emotions['happy'])
        BluffGap[4] = abs(Bluff[4]-emotions['sad'])
        BluffGap[5] = abs(Bluff[5]-emotions['surprise'])
        BluffGap[6] = abs(Bluff[6]-emotions['neutral'])
        TruthSum = TruthGap[0]+TruthGap[1]+TruthGap[2]+TruthGap[3]+TruthGap[4]+TruthGap[5]+TruthGap[6]
        BluffSum = BluffGap[0]+BluffGap[1]+BluffGap[2]+BluffGap[3]+BluffGap[4]+BluffGap[5]+BluffGap[6]
        if(TruthSum> BluffSum):
            DisplayText = 'Bluff'
        elif(TruthSum<BluffSum):
            DisplayText = 'Truth'
        else:
            DisplayText = 'Unclear'
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255,0),2)
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(frame, DisplayText, (50,50), font, 3, (255,0,0), 2, cv2.LINE_4)
    cv2.imshow('Original video',frame)
    x = cv2.waitKey(2)&0xFF
    if x == ord('t'):
        Truth[0] = ((Truth[0]*TruthDataPoints)+emotions['angry'])/(TruthDataPoints+1)
        Truth[1] = ((Truth[1]*TruthDataPoints)+emotions['disgust'])/(TruthDataPoints+1)
        Truth[2] = ((Truth[2]*TruthDataPoints)+emotions['fear'])/(TruthDataPoints+1)
        Truth[3] = ((Truth[3]*TruthDataPoints)+emotions['happy'])/(TruthDataPoints+1)
        Truth[4] = ((Truth[4]*TruthDataPoints)+emotions['sad'])/(TruthDataPoints+1)
        Truth[5] = ((Truth[5]*TruthDataPoints)+emotions['surprise'])/(TruthDataPoints+1)
        Truth[6] = ((Truth[6]*TruthDataPoints)+emotions['neutral'])/(TruthDataPoints+1)
        TruthDataPoints += 1
    elif x == ord('b'):
        Bluff[0] = ((Bluff[0]*BluffDataPoints)+emotions['angry'])/(BluffDataPoints+1)
        Bluff[1] = ((Bluff[1]*BluffDataPoints)+emotions['disgust'])/(BluffDataPoints+1)
        Bluff[2] = ((Bluff[2]*BluffDataPoints)+emotions['fear'])/(BluffDataPoints+1)
        Bluff[3] = ((Bluff[3]*BluffDataPoints)+emotions['happy'])/(BluffDataPoints+1)
        Bluff[4] = ((Bluff[4]*BluffDataPoints)+emotions['sad'])/(BluffDataPoints+1)
        Bluff[5] = ((Bluff[5]*BluffDataPoints)+emotions['surprise'])/(BluffDataPoints+1)
        Bluff[6] = ((Bluff[6]*BluffDataPoints)+emotions['neutral'])/(BluffDataPoints+1)
        BluffDataPoints += 1
    if x == ord('q'):
        break
    isDone = False
cap.release()
cv2.destroyAllWindows()
