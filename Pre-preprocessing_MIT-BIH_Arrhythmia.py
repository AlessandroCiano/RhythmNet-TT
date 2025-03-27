#This is the python file of the notebook, so we can run it from a terminal

#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

#First, function to load the data
DataPath="mit_bih_csv_files/ecg_records/"
def RetrieveData(RecordNum):
    """Retrieves the ecg data from the record provided"""
    ecgData=pd.read_csv(DataPath+str(RecordNum)+"_ecg.csv")
    ecgAnno=pd.read_csv(DataPath+str(RecordNum)+"_annotations.csv")
    return np.array(ecgData),np.array(ecgAnno)

colNumforECG=0 #The column index for the ECG channel we are picking - either 0 or 1

def FileCreation(Seconds=3):
    MaxNumBeats=Seconds*5 #The maximum number of beats per window

    MaxNumBeats=MaxNumBeats*2
    TimeFrame=Seconds*360

    def ConvertAnnos(Anno,Intervals):
        """Takes annotation data and returns a labels within the intervals"""
        ReturnAnnotations=Anno[Anno[:,1]<=TimeFrame]
        ReturnAnnotations=ReturnAnnotations[ReturnAnnotations[:,0]!="+"] #Remove the classes with "+"
        ReturnAnnotations=np.ndarray.flatten(ReturnAnnotations)
        ReturnAnnotations=[np.pad(ReturnAnnotations,pad_width=(0,MaxNumBeats-len(ReturnAnnotations)))]
        for j,i in enumerate(Intervals[:-1]):
            AddArray=Anno[(Anno[:,1]<=Intervals[j+1])*(Anno[:,1]>=i)] #Get array of annotations and positions
            AddArray[:,1]=AddArray[:,1]-i #Adjust the position values so they are relative to the window
            AddArray=AddArray[AddArray[:,0]!="+"] #Remove the + values from the data
            AddArray=np.ndarray.flatten(AddArray) #Flatten array to append
            ReturnAnnotations=np.append(ReturnAnnotations,[np.pad(AddArray,pad_width=(0,MaxNumBeats-len(AddArray)))],axis=0) #Append the array
        return ReturnAnnotations


    def ProcessECGData(Data,Anno):
        """Takes ECG data and the annotations and outputs a matrix with t second chunks with an annotation at the end"""
        #Intervals:
        Intervals=np.arange(TimeFrame,Data.shape[0],TimeFrame)
        NewData=Data[0:TimeFrame][:,np.newaxis].T #New data is our data that is segmented into 30 second chunks
        Labels=ConvertAnnos(Anno,Intervals)
        for j,i in enumerate(Intervals[:-1]):
            NewData=np.append(NewData,[Data[i:Intervals[j+1]]],axis=0)
        NewData=np.append(NewData,Labels,axis=1)
        return NewData

    def RetrievePatientData(Num):
        Data,Annotation=RetrieveData(Num)
        return ProcessECGData(Data[:,colNumforECG],Annotation)

    RetrievePatientData(101).shape


    #Now we have the functions, we can do this for each number
    #Numbers: 100-124, 200-234
    import os #Just to get the file names
    fileNames=os.listdir(DataPath)
    df=pd.DataFrame(RetrievePatientData(100))
    for fileIndex in np.arange(3,len(fileNames),3):
        Num=fileNames[fileIndex][:3]
        df2=pd.DataFrame(RetrievePatientData(Num))
        df=pd.concat([df,df2],ignore_index=True)



    MaxNumofAnnotations=Seconds*2-1 # This is the maximum number of beats that we want to classify in the time frame

    df=df[df[TimeFrame+(1+MaxNumofAnnotations)*2]==0] #Gets rid of any lines that are more than those annotations

    df=df.loc[:, (df != 0).any(axis=0)]

    #I am also going to remove the lines with unclassifiable beats


    JustAnnotations=df.iloc[:,(360*Seconds):]
    print("Orig length before cuts:",JustAnnotations.shape[0])
    usefulDataIndex=0
    for i in range(0,(Seconds*2),2):
        udi=(JustAnnotations.iloc[:,i]=="N")+(JustAnnotations.iloc[:,i]=="L")+(JustAnnotations.iloc[:,i]=="R")+(JustAnnotations.iloc[:,i]=="j")+(JustAnnotations.iloc[:,i]=="e")+(JustAnnotations.iloc[:,i]=="A")+(JustAnnotations.iloc[:,i]=="a")+(JustAnnotations.iloc[:,i]=="S")+(JustAnnotations.iloc[:,i]=="J")+(JustAnnotations.iloc[:,i]=="!")+(JustAnnotations.iloc[:,i]=="V")+(JustAnnotations.iloc[:,i]=="E")+(JustAnnotations.iloc[:,i]=="[")+(JustAnnotations.iloc[:,i]=="]")+(JustAnnotations.iloc[:,i]=="F")
        if i==0:
            usefulDataIndex=np.array(udi)
        else:
            usefulDataIndex=usefulDataIndex*np.array(udi)

    print("Reduction: ",sum(usefulDataIndex)/df.shape[0])
    df=df[usefulDataIndex==True]
    print("resulting length: ",df.shape[0])

    #Save it
    df.to_csv(f"MIT{Seconds}sec.csv")


SecondsList=[3,7,15] #The number of seconds we look at

for second in tqdm.tqdm(SecondsList):
    FileCreation(second)