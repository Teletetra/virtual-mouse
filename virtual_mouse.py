import cv2
import mediapipe as mp
import pyautogui
cap=cv2.VideoCapture(0)
hand_detector=mp.solutions.hands.Hands()
mpDraw=mp.solutions.drawing_utils
pTime=0
cTime=0
while True:
    success,frame=cap.read()
    frame=cv2.flip(frame,1)
    f_h,f_w,success=frame.shape
    frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=hand_detector.process(frameRGB)
    hands = results.multi_hand_landmarks
    if hands:
        for hand in hands:
            mpDraw.draw_landmarks(frame,hand)
            landmarks=hand.landmark
            for id ,landmark in enumerate(landmarks):
                x= int(landmark.x*f_w)
                y=int(landmark.y*f_h)
                print(x,y)
                if id ==8:
                    cv2.circle(img=frame,center=(x,y),radius=10,color=(0,255,255))
                    pyautogui.moveTo(x,y)
            
    cv2.imshow("virtual mouse",frame)
    cv2.waitKey(1)