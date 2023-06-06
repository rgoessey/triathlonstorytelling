import json
import pandas as pd

import sys
sys.path.insert(0,'drive/MyDrive')
from deep_sort_realtime.deepsort_tracker import DeepSort
from storytelling import storytelling,triatleteview
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import math
from torchvision import transforms
from pathlib import Path
import torch
from ultralytics import YOLO,nn
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import requests
from random import randrange
import re
import pytesseract
if sys.argv[1]=="cpu":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import sys

print(sys.argv)

modelCaps = YOLO(sys.argv[2])
modelNr = YOLO(sys.argv[4])
modelClass =  YOLO(sys.argv[3])

if sys.argv[1]=="cuda":
    modelCaps.to("cuda")
    modelNr.to("cuda")
    modelNr.to("cuda")
data = {'apikey':"2543796db5b8b0ef15505ce751a22a28"}
shots={}
shotsList=[]
lower_blue = np.array([90, 0, 50])
upper_blue = np.array([130, 255, 215])

lower_white = np.array([0,0,168])
upper_white = np.array([170,75,255])

#get an intersection between two bounding boxes
def get_iou(bb1, bb2):
    """assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]"""

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def getTriatletes(loc):
    r = requests.get(url = "https://api.triathlon.org/v1/search/events?query="+ loc + "&filters=year,2022", headers=data)
    #print(r.json()["data"][0])
    #omzetten naar json
    eventid=r.json()["data"][0]["event_id"]
    r=requests.get(url = "https://api.triathlon.org/v1/events/"+str(eventid),headers=data)
    #print(r.json()["data"]["programs"])
        
    for i in r.json()["data"]["programs"]:
        if i["prog_name"]=="Elite Men":
            programid=i["prog_id"]
            
    r = requests.get(url = "https://api.triathlon.org/v1/events/{}/programs/{}/entries?type&entry_id&country_id".format(eventid,programid), headers=data)
    tnames=[" "]*70
    k=1
    #print(r.json()["data"]["entries"])
    for i in r.json()["data"]["entries"]:
        tnames[i["start_num"]-1]=i["athlete_title"].replace('é',"e").replace("á","a").replace("ö","o").replace("ü","u")
        k+=1
        
    for index,athletes in enumerate(tnames):
        print(index+1,athletes)
    return tnames

"""def getNumber(result):
    string=[]
    for k in result[0].boxes.data:
        if (len(string) == 0):
            string.append(k)
        else:
            b=True
            for index,s in enumerate(string):
                if (k[4]>s[4]):
                    if (get_iou([s[0],s[1],s[2],s[3]],[k[0],k[1],k[2],k[3]])>0.3):
                        string[index]=k
                        b=False
                        break
            if b:
                string.append(k)
                    
    return string"""

def getNumber(result,kleur):
    string=[]
    for k in result[0].boxes.data:
        if k[4]>0.6 and (len(string) == 0):
            string.append(k)
        else:
            b=True
            for index,s in enumerate(string):
                if (k[4]>s[4]):
                    if (get_iou([s[0],s[1],s[2],s[3]],[k[0],k[1],k[2],k[3]])>0.3):
                        string[index]=k
                        b=False
                        break
            if b and k[4]>0.6:
                string.append(k)
                    
    return string

def geefRechtseBox(badmutsen):
    mrechts=0
    boxnr=-1
    for index,i in enumerate(badmutsen):
        i=i.cpu().numpy()
        if mrechts<i[2]:
            mrechts=i[2]
            boxnr=index
    return badmutsen[boxnr]

def resultHasOnlyBuoy(result):
    if result is not None: 
        if result[0].boxes is not None:
            for b in result[0].boxes.data:
                if result[0].names[int(b[5])]!="buoy":
                    return False
    return True

def checkSwim(image):
    b=True
    s=(720,1280)
    y,x=image.shape[0:2]
    l=int(637/s[1]*x)
    b=int(1015/s[0]*y)
    r=int(666/s[1]*x)
    t=int(1150/s[0]*y)
    time=[]
    box=image[l:r, b:t]
    if box.shape[0]>3:
        imgHsv = cv2.cvtColor(box,cv2.COLOR_BGR2HSV)


        # Maak een masker op basis van het blauwbereik
        mask = cv2.inRange(imgHsv, lower_white, upper_white)
        res = cv2.bitwise_and(box,box,mask = mask)
        d = pytesseract.image_to_string(res, config="--psm 10")
        cv2.putText(image, str(d), (640,640), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3, cv2.LINE_AA)
        time=re.findall(r'\d+',str(d))
        """cv2.imshow("",res)
        cv2.waitKey(0)"""
    else:
        b=False
    
    
    l=int(585/s[1]*x)
    b=int(940/s[0]*y)
    r=int(612/s[1]*x)
    t=int(1150/s[0]*y)
    box2=image[l:r, b:t]
    if box2.shape[0]>3:
        imgHsv = cv2.cvtColor(box2,cv2.COLOR_BGR2HSV)


        # Maak een masker op basis van het blauwbereik
        mask = cv2.inRange(imgHsv, lower_blue, upper_blue)
        res = cv2.bitwise_and(box2,box2,mask = mask) 
        d = pytesseract.image_to_string(res,config="--psm 10")
        cv2.putText(image, str(d), (340,640), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3, cv2.LINE_AA)
        b=len(re.findall(r"(.*?)[s,w,i,m,S,W,I,M]{2}(.*?)", d))>0
        """cv2.imshow("",res)
        cv2.waitKey(0)"""
    else:
        b=False
    return b


kleuren={"yellow":[1,10],"gray":[11,20],"red":[21,30],"blue":[31,40],"green":[41,50],"purple":[51,60],"orange":[61,70]}
def checkNumber(badmutsen,kleur):
    if kleur!="buoy":
        string=""
        betr=0
        for i in badmutsen:
            i=i.cpu().numpy()
            string+=str(int(i[5]))
            betr+=i[4]
        if len(badmutsen)>0:
            if int(geefRechtseBox(badmutsen).cpu().numpy()[5])==0:
                return kleuren[kleur][1],(geefRechtseBox(badmutsen).cpu().numpy()[4]-0.20)
            else:
                if int(kleuren[kleur][0]/10)==int(geefRechtseBox(badmutsen).cpu().numpy()[5]):
                    return (kleuren[kleur][0]-1)+int(geefRechtseBox(badmutsen).cpu().numpy()[5]),(geefRechtseBox(badmutsen).cpu().numpy()[4]-0.25)
                else:
                    return (kleuren[kleur][0]-1)+int(geefRechtseBox(badmutsen).cpu().numpy()[5]),geefRechtseBox(badmutsen).cpu().numpy()[4]
    return -1,-1

def checkChange(shot,shotList):
    global shots
    global shotsList
    if len(shotList)>30:
        totalProb=0
        total=0
        for i in range(-1,-30,-1):
            if shotList[i][0] != shot:
                total+=1
                totalProb+=float(shotList[i][1])
        if total>22 and (totalProb/total)>0.70 and checkFarChange(shotList):
            return False
        if len(shotList)>60:
            shots={}
            shotsList=[]
    return True

def checkFarChange(shotsList):
    totalStart=0
    totalEnd=0
    if len(shotsList)>24:
        for i in range(-1,-int(len(shotsList)/2),-1):
            totalStart+=shotsList[i][2]
            totalEnd+=shotsList[-i][2]
    if (totalStart < int(len(shotsList)/2)*1.10 and totalEnd > int(len(shotsList)/2))*1.10 or ((totalEnd < int(len(shotsList)/2)*1.10 and totalStart > int(len(shotsList)/2)*1.10)):
        return True
    else:
        return False
        
        
    
if __name__ == "__main__":
    

    tnames=getTriatletes("leeds")
    st=storytelling.Storytelling()        
    scenes=[]
    cap = cv2.VideoCapture(sys.argv[5])
    swim=[]
    #Check if they are swimming
    while sum(swim[-5:]) < 5:
        ret,frame = cap.read()
        swim.append(int(checkSwim(frame)))
    swim=[]
    inti=0
    if len(sys.argv)>1 and sys.argv[1]=="cuda":
        model=nn.tasks.attempt_load_weights(weights=sys.argv[4],device=torch.device('cuda'))
    else:
        model=nn.tasks.attempt_load_weights(weights=sys.argv[4],device=torch.device('cpu'))
    #for mmodel
    #tracker = DeepSort(max_age=24,n_init=2,nms_max_overlap=0.2,max_iou_distance=0.6,max_cosine_distance=0.5,embedder=None)
    #for xmodel

    #tracker=DeepSort(max_age=30,n_init=3,nms_max_overlap=0.3,max_iou_distance=0.3)
    # Option 1: passing weights param as string
    def copy_embeddings(m, i, o):
        """Copy embeddings from the penultimate layer.
        """
        if len(sys.argv)>1 and sys.argv[1]=="cuda":
            o = o[:, :, 0, 0].detach().cpu().numpy().tolist()
        else:
            o = o[:, :, 0, 0].detach().numpy().tolist()
        outputs.append(o)
        


    outputs=[]
    #layer = model._modules.get('avgpool')
    #layer=model.blocks[5].pool
    #layer=model.model[21].m[0]
    layer=model.model[19]


    # attach hook to the penulimate layer
    _ = layer.register_forward_hook(copy_embeddings)
    convert_tensor = transforms.ToTensor()


    model.eval() # Inference mode
    persons={}
    prob={}
    trackVerbinding={}
    tracker = DeepSort(max_age=22,n_init=3,nms_max_overlap=0.7,max_iou_distance=0.8,max_cosine_distance=0.5,embedder=None,end=cap.get(cv2.CAP_PROP_FRAME_COUNT)/25)
    sceneNr=1
    vorigeScene="geen"
    #while(cap.get(cv2.CAP_PROP_POS_FRAMES)<(cap.get(cv2.CAP_PROP_FRAME_COUNT)-1) and sum(swim[-5:]) < 5):
    while(cap.get(cv2.CAP_PROP_POS_FRAMES)<77 and sum(swim[-5:]) < 5):
        print()
        startTime=0 if (cap.get(cv2.CAP_PROP_POS_FRAMES)-30)/25<=0 else (cap.get(cv2.CAP_PROP_POS_FRAMES)-30)/25
        print("Dit is scene {}: gestart op seconde {} ".format(sceneNr,startTime))
        shot=""
        shots={}
        shotsList=[]
        inti+=1
        print("scene",inti)
        while checkChange(shot,shotsList):
            #print("fnr",cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret,frame = cap.read()
            if frame is None:
                break
            start_time = time.time()
            #frame=cv2.resize(frame,(640,640))
            resultClass=modelClass(frame,verbose=False)
            resultCaps = modelCaps(frame,verbose=False)
            c={}
            for index,k in enumerate(resultClass[0].probs):
                c[resultClass[0].names[index]]=str(float(k))
            c=sorted(c.items(), key=lambda x:x[1],reverse=True)
            if c[0][0]!=shot:
                shot=c[0][0]
            shotsList.append([shots,c[0][1],len(resultCaps[0].boxes.data)])    
            if shot in shots:
                shots[shot]+=1
            else:
                shots[shot]=1
            
            if True:

                outputs = []
                detections=[]
                persons=[]
                for k in resultCaps[0].boxes.data:
                    if len(sys.argv)>1 and sys.argv[1]=="cuda":
                        k=k.cpu().numpy()
                    else:
                        k=k.numpy()
                    l,t,r,b=k[0:4]
                    detections.append([[k[0],k[1],k[2]-k[0],k[3]-k[1]],k[4],str(resultCaps[0].names[k[5]])])
                    roi = frame[int(t):int(b), int(l):int(r)]
                    roi=cv2.resize(roi,(128,128))
                    if len(sys.argv)>1 and sys.argv[1]=="cuda":
                        resultNr=model(convert_tensor(roi).cuda().unsqueeze(0))
                    else:
                        resultNr=model(convert_tensor(roi).unsqueeze(0))
                    person={"name":"","number":-1,"prob":-1}
                    if (roi.shape[0] > 5) & (roi.shape[1] > 5):
                        badnummers=getNumber(modelNr(roi,verbose=False),str(resultCaps[0].names[k[5]]))
                        if len(badnummers) > 0:
                            nr,prob=checkNumber(badnummers,str(resultCaps[0].names[k[5]]))
                            person={"name": tnames[nr-1],"number":nr,"prob":prob}
                    persons.append(person)
                for i in range(len(outputs)):
                    outputs[i]=np.squeeze(np.asarray(outputs[i]))
                
                
                    
                tracks = tracker.update_tracks(detections, persons,embeds=outputs)
                #tracks=tracker.update_tracks(detections, frame=frame)
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    l,t,r,b=track.to_ltrb()
                    kl=track.det_class
                    cv2.rectangle(frame, pt1=(int(l),int(t)), pt2=(int(r),int(b)), color=(0,0,0), thickness=1)
                    #cv2.putText(frame, "id:"+str(track_id), (int(l),int(t)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
                    name=""
                    number=-1
                    if track.triatlete is not None:
                        name=track.triatlete.name
                        number=track.triatlete.number
                    else:
                        name=kl
                        number=" "
                    cv2.putText(frame, str(track.track_id) + " : " + str(track.triatlete.numbers), (int(l),int(t)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
            fps=1.0 / (time.time() - start_time)
            cv2.putText(frame, "fps:" + str(fps), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, cv2.LINE_AA)
            #cv2.imshow('Frame', cv2.resize(frame,(1270,720)))
            if sys.argv[1]=="cpu":
                cv2.putText(frame, str(inti), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, cv2.LINE_AA)
                cv2.imshow("main",cv2.resize(frame,(720,720)))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
        """if frame is not None:
            cv2.imshow("main",cv2.resize(frame,(720,720)))
        else:
            break"""
        endTime=0 if (cap.get(cv2.CAP_PROP_POS_FRAMES)-30)/25<=0 else (cap.get(cv2.CAP_PROP_POS_FRAMES)-30)/25
        print()
        shot=sorted(shots.items(), key=lambda x:x[1],reverse=True)[0][0]
        print("eindtijd is {}: met shot in {}".format(endTime,shot))
        numberTracks,colorTracks,closeTracks=tracker.geefStorytellingScene()
        triatletes,kleuren=st.giveStory(startTime,endTime,numberTracks,colorTracks,closeTracks)
        scenes.append(triatletes)
        #tracker.startNewScene()
        sceneNr+=1
        swim.append(int(not checkSwim(frame)))

    print("zwemmen afgelopen")
    st.geefTotaal(scenes)

    # the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()