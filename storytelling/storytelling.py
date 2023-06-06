import numpy as np
import math

def get_dist(loc1,loc2):
    # Bereken het centrale punt van elke LTRB-locatie
    center1 = [(loc1[0] + loc1[2]) / 2, (loc1[1] + loc1[3]) / 2]
    center2 = [(loc2[0] + loc2[2]) / 2, (loc2[1] + loc2[3]) / 2]
    
    # Bereken de Euclidische afstand tussen de centrale punten
    distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    return distance
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
    """assert iou >= 0.0
    assert iou <= 1.0"""
    return iou

def ltwh_to_ltrb(ltwh):
    return (ltwh[0],ltwh[1],ltwh[0]+ltwh[2],ltwh[1]+ltwh[3])

### (dict (id,triatlete)) numerTracks keeps all the tracks where numbers were detected
### (dict (id,triatlete)) colorTracks keeps all the tracks where no numbers were detected
### (dict (id,list of ids)) closeTracks keeps all tracks which have a close track who can be the same ground truth object
### (list) sameTracks are all tracks that are already processed so can be skipped
class Storytelling:
    def __init__(
        self,
        numberTracks={},
        colorTracks={},
    ):
        self.numberTracks=numberTracks
        self.colorTracks=colorTracks
        self.closeTracks={}
        self.sameTracks=[]

    def giveStory(self,start,stop,nT,cT,closeT):
        self.numberTracks=nT
        self.colorTracks=cT
        self.closeTracks=closeT
        triatletes=self.geefTriatletenScene(start,stop,25)
        kleuren=self.geefBadmutsenScene(start,stop)
        return triatletes,kleuren
    
    """def checkTrackPos(self,t1,t2):
        if t1.starttime < t2.starttime:
            diff=t2.starttime-t1.starttime
            trpos2=[None]*diff + t2.trackPos
            trpos1=t1.trackPos
        else:
            diff=t1.starttime-t2.starttime
            trpos1=[None]*diff + t1.trackPos
            trpos2=t2.trackPos
        i=0
        while len(trpos1)>i and len(trpos2)>i:
            if trpos1[i] is not None and trpos2[i] is not None and get_iou(ltwh_to_ltrb(trpos1[i]),ltwh_to_ltrb(trpos2[i]))>0.0 and get_dist(ltwh_to_ltrb(trpos1[i]),ltwh_to_ltrb(trpos2[i])) < 100:
                return True
            i+=1
        return False"""
        
    """def getCloseTracks(self,start,stop):
        tracks=[self.numberTracks[k] for k in self.numberTracks if start <= self.numberTracks[k].starttime/25 <= stop] + [self.colorTracks[k] for k in self.colorTracks if start <= self.colorTracks[k].starttime/25 <= stop]
        for t1 in tracks:
            for t2 in tracks:
                if self.checkTrackPos(t1,t2) and t1.track_id != t2.track_id:
                    self.closeTracks[t1.track_id]=t2.track_id
                    self.closeTracks[t2.track_id]=t1.track_id"""
    
    # Called in "geefTriatlonScene" to take all tracks together that are probably the same object by returning a list of ids
    # works with a stack to work recursively
    # after adding the id to the ids returning list, its also added to sametracks 
    def checkCloseTracks(self,tr):
        ids=[tr.track_id]
        stack=[tr.track_id]
        while(len(stack)>0):
            a=stack.pop()
            if a in self.closeTracks:
                arrayTracks=self.closeTracks[a]
                for a in arrayTracks:
                    if a not in ids:
                        self.sameTracks.append(a)
                        ids.append(a)
                        stack.append(a)
            try:
                stack.index(a)
            except:
                continue
            else:
                stack.remove(a)
        return ids
    
    # Called in "geefTriatlonScene" to get the best number with highest prob with the given same tracks ids
    # Insert this in a dict and take the best number with the highest prob
    def getNumberOfSameTracks(self,ids):
        numbers={}
        for id in ids:
            if id in self.numberTracks:
                if self.numberTracks[id].triatlete.number!=-1:
                    nr=self.numberTracks[id].triatlete.pickNumberHighestProb()
                    if nr[0] not in numbers or numbers[nr[0]]<nr[1]:
                        numbers[nr[0]]=nr[1]
        best=sorted(numbers.items(), key=lambda x:x,reverse=True)[0]
        return best[0],best[1]
    
    def checkColor(self,ids):
        colors={}
        for id in ids:
            if id not in self.numberTracks and id in self.colorTracks:
                tr=self.colorTracks[id]
                if tr.det_class not in colors:
                    colors[tr.det_class]=1
                else:
                    colors[tr.det_class]+=1
        color=sorted(colors.items(), key=lambda x:x,reverse=True)[0][0]
        return color
        
    # called in tracker to give triatletes and their corresponding time
    # calles the checkCloseTracks and getNumberOfSameTracks function to get the best number of tracks that can be the same object
    # Only tracks that have detected numbers are processed
    # Returns a triatletes array where (probability, start, end, name) can be found
    # returning tracks is still empty but here can all corresponding tracks be saved
    def geefTriatletenScene(self,start,stop,fps):
        triatletes={}
        #tracks={}
        for tr in self.numberTracks:
            tr=self.numberTracks[tr]
            if tr.triatlete.number==-1 or ((tr.starttime/fps) < start or (tr.stoptime/fps) > stop) or tr.track_id in self.sameTracks:
                continue
            ids=self.checkCloseTracks(tr)
            bestNr,prob=self.getNumberOfSameTracks(ids)
            if bestNr not in triatletes:
                if tr.stoptime==-1:
                    if tr.starttime/fps < stop:
                        triatletes[bestNr]=[prob,(tr.starttime/fps),stop,tr.triatlete.name]
                else:
                    triatletes[bestNr]=[prob,(tr.starttime/fps),(tr.stoptime/fps),tr.triatlete.name]
            else:
                if bestNr%10==1:
                    if tr.stoptime==-1:
                        if tr.starttime/fps < stop:
                            triatletes[bestNr-1+7]=[prob-0.10,(tr.starttime/fps),stop," "]
                    else:
                        triatletes[bestNr-1+7]=[prob-0.10,(tr.starttime/fps),(tr.stoptime/fps)," "]
                if triatletes[bestNr][1]>(tr.starttime/fps):
                    triatletes[bestNr][1]=(tr.starttime/fps)
                elif (tr.stoptime/fps) != stop and triatletes[bestNr][2]<(tr.stoptime/fps):
                    triatletes[bestNr][2]=(tr.stoptime/fps)
        for tri in triatletes:
            if round(triatletes[tri][0],2)>0.65:
                print("{} ({}) met probabiliteit van {} was te zien van {} tot {}".format(triatletes[tri][3],tri,round(triatletes[tri][0],2),round(triatletes[tri][1],2),round(triatletes[tri][2],2)))
        return triatletes
        
    # called in tracker to get all tracks that don't have a number to only show their caps 
    # Makes a dict to count the colors that can be seen in the scene
    def geefBadmutsenScene(self,start,stop):
        kleuren={}
        for t in self.colorTracks:
            t=self.colorTracks[t]
            if t.triatlete.number!=-1 or ((t.starttime/25) < start or (t.stoptime/25) > stop) or t.track_id in self.sameTracks or t.hits < 4:
                continue
            ids=self.checkCloseTracks(t)
            kleur=self.checkColor(ids)
            if kleur not in kleuren:
                kleuren[kleur]=1
            else:
                kleuren[kleur]+=1
        for k in kleuren:
            print("There {} {} {} anonymous {}".format("is" if kleuren[k]==1 else "are",kleuren[k],k,"cap" if kleuren[k] == 1 else "caps"))
        return kleuren

    def geefTotaal(self,scenes):
        #triatleet geeft naam probability en start en stop
        triatletes={}
        for s in scenes:
            for t in s:
                if s[t][3] in triatletes:
                    triatletes[s[t][3]].append(s[t])
                else:
                    triatletes[s[t][3]]=[s[t]]
        for t in triatletes:
            print(t,end=" is te zien met prob. van")
            for k in triatletes[t]:
                print(round(k[0],2),end=", tussen ")
                print("{} - {} seconde".format(round(k[1],2),round(k[2],2)))
            print()
            print()


            
        