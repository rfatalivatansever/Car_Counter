import cv2
import numpy as np

cam  = cv2.VideoCapture("araçSayısı.mp4")
fgbd = cv2.createBackgroundSubtractorMOG2()

kernel = np.ones((5,5),np.uint8)

class Coordinate() :
    def __init__(self,x,y):
        self.x = x
        self.y = y
class Sensor :
    def __init__(self,C1,C2,Square_w,Square_h):
        self.C1 = C1
        self.C2 = C2
        self.Square_w = Square_w
        self.Square_h = Square_h
        self.Mask_Area = abs(self.C2.x - self.C1.x)*abs(self.C2.y-self.C1.y)
        self.mask = np.zeros((Square_w,Square_h,1),np.uint8)
        cv2.rectangle(self.mask,(self.C1.x,self.C1.y),(self.C2.x,self.C2.y),(255),cv2.FILLED)
        self.situation = False
        self.Car_Counter = 0

Sensor = Sensor(Coordinate(380,240),Coordinate(660,220),1080,250)

font = cv2.FONT_HERSHEY_SIMPLEX

while True :
    ret,frame = cam.read()
    frame1 = frame[350:600,90:1000]
    bcgrnd_delete_frame = fgbd.apply(frame1)
    bcgrnd_delete_frame = cv2.morphologyEx(bcgrnd_delete_frame,cv2.MORPH_OPEN,kernel)
    ret1,bcgrnd_delete_frame = cv2.threshold(bcgrnd_delete_frame,127,255,cv2.THRESH_BINARY)

    cnts,hieararchy = cv2.findContours(bcgrnd_delete_frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    filled_square = np.zeros((frame1.shape[0],frame1.shape[1],1),np.uint8)

    for cnt in cnts :
        x,y,w,h = cv2.boundingRect(cnt)
        if (w>50 and h>50) :
            cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,0),1)
            cv2.rectangle(filled_square,(x,y),(x+w,y+h),(255),cv2.FILLED)

    Sensor_Mask_Result = cv2.bitwise_and(filled_square,filled_square , Sensor.mask)
    Sensor_White_Pixel_Number = np.sum(Sensor_Mask_Result == 255)
    print(Sensor_White_Pixel_Number)
    Sensor_Rate = Sensor_White_Pixel_Number / Sensor.Mask_Area
    print(Sensor_Rate)

    if (Sensor_Rate >= 0.75 and Sensor.situation == False) :
        cv2.rectangle(frame1,(Sensor.C1.x,Sensor.C1.y),
                      (Sensor.C2.x,Sensor.C2.y),
                      (0,255,0),cv2.FILLED)
        Sensor.situation= True
    elif(Sensor_Rate <= 0.75 and Sensor.situation == True) :
        cv2.rectangle(frame1,(Sensor.C1.x,Sensor.C1.y),
                      (Sensor.C2.x,Sensor.C2.y),
                      (0,0,255),cv2.FILLED)
        Sensor.situation = False
        Sensor.Car_Counter += 1
    else:
        cv2.rectangle(frame1,(Sensor.C1.x,Sensor.C1.y),
                      (Sensor.C2.x,Sensor.C2.y),
                      (255,0,0),cv2.FILLED)

    cv2.putText(frame1, str(Sensor.Car_Counter),
                (Sensor.C1.x, Sensor.C1.y + 1), font, 2, (255, 255, 255))

    cv2.imshow("FRAME", frame1)

    if cv2.waitKey(30) & 0xFF == ord("q") :
        break

cam.release()
cv2.destroyAllWindows()
