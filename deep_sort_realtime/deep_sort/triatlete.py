
### Keeps a numbers list to save all detected numbers
### keeps an assoc list to save the names coupled to the detected numbers (maybe not efficient??)
class Triatlete:
    def __init__(
        self,
        name="",
        prob="",
        number=-1,
    ):
        self.name=name
        self.prob=prob
        self.number=number
        self.numbers={}
        self.assoc={}

    # Updates triatlete called from tracker class to update a detected number to see which triatlete the track is coupled to
    # Fills the assoc + number list with the given triatlete
    # recalculates the best number after adding the detected number
    def update(self,triatlete):
        if triatlete["number"]!=-1:
            if triatlete["number"] not in self.assoc:
                self.assoc[triatlete["number"]]=triatlete["name"]
            if triatlete["number"]!=-1:
                if triatlete["number"] in self.numbers and triatlete["prob"]>self.numbers[triatlete["number"]]:
                    self.numbers[triatlete["number"]]=triatlete["prob"]
                elif triatlete["number"] not in self.numbers:
                    self.numbers[triatlete["number"]]=triatlete["prob"]
        self.pickNumberHighestProb()
    
    # Find best number in detected nummers with most appearances and highest probability
    def pickNumberHighestProb(self):
        nr=sorted(self.numbers.items(), key=lambda x:x[1],reverse=True)
        if len(nr)>0:
            self.prob=nr[0][1]
            self.number=nr[0][0]
            self.name=self.assoc[self.number]
        return self.number,self.prob

    # Looks less efficient
    def pickBestNumberWithMostAppearance(self):
        numbers={}
        aantal=0
        for n in self.numbers:
            if n[0] not in numbers:
                numbers[n[0]]=n[1]
            else:
                numbers[n[0]]+=n[1]
            aantal+=1
        bestNr=-1
        prob=-1
        for i in numbers:
            if numbers[i]>prob:
                prob=numbers[i]/aantal
                bestNr=i
        self.prob=prob
        self.number=bestNr
        self.name=self.assoc[bestNr]
        return bestNr,prob

    # Find best number in detected nummers with highest probability
    def pickBestNumberWithHighProb(self):
        numbers={}
        print(self.numbers)
        for n in self.numbers:
            if n[0] not in numbers:
                numbers[n[0]]=n[1]
            elif n[1]>numbers[n[0]]:
                numbers[n[0]]=n[1]
        best=sorted(numbers.items(), key=lambda x:x,reverse=True)
        if len(best)>0:
            self.prob=best[0][1]
            self.number=best[0][0]
            self.name=self.assoc[best[0][0]]
            return best[0]
        else:
            return (-1,-1)
        

