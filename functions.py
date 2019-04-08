import keras
#from sklearn import datasets
import numpy as np
import os
import numpy as np
import cv2 # OpenCV
import matplotlib.pyplot as plt
import collections
import math

import tensorflow as tf
from sklearn import datasets
from keras.models import Sequential
from keras.layers.core import Dense,Activation, Dropout
from keras.optimizers import SGD

def crossingLine():
    i=1

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def img_thl(image):
    ret,image_bin = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def invert(image):
    return 255-image


def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

def findBlueLine(frame):

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_blue= np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    img = cv2.bitwise_and(frame, frame, mask = mask)
    
    gray = image_gray(img)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=50,maxLineGap=13)

    line=lines[0]
    x1,y1,x2,y2=line[0]

    for line in lines:
        x1t,y1t,x2t,y2t=line[0]
        if y2t < y2:
            y1=y1t
            x1=x1t
            x2=x2t
            y2=y2t
        
    #cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    #cv2.imshow('nnnn',img)
    #print('--------------------------------------------')
    #print(x1)
    #print(y1)
    #print(x2)
    #print(y2)
    coefficients = np.polyfit([x1,x2], [y1,y2], 1)
   
    return x1,y1,x2,y2,coefficients[0], coefficients[1]

def findGreenLine(frame):
    frame=erode(frame)
    frame=dilate(frame)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_green = np.array([60, 100, 100])
    upper_green = np.array([60, 255, 255])
    mask = cv2.inRange(hsv,lower_green,upper_green)
    img = cv2.bitwise_and(frame, frame, mask = mask)

    gray = image_gray(img)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength=50,maxLineGap=13)

    line=lines[0]
    x1,y1,x2,y2=line[0]

    for line in lines:
        x1t,y1t,x2t,y2t=line[0]
        if y2t < y2:
            y1=y1t
            x1=x1t
            x2=x2t
            y2=y2t
        
    #cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
    #cv2.imshow('nnnn',img)
    #print('--------------------------------------------')
    #print(x1)
    #print(y1)
    #print(x2)
    #print(y2)
    coefficients = np.polyfit([x1,x2], [y1,y2], 1)
    
    return x1,y1,x2,y2,coefficients[0], coefficients[1]

def select_roi(image_orig, image_bin):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28. 
        Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
        i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    coordinates=[] #koordinate gornje levog ugla pravougoanika, sirina, visin
    sorted_regions = [] # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        
        if(area > 13 and h < 43 and h > 13 and w > 1): 
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            coordinates.append([x,y,w,h])
            region = image_bin[y:y+h+1,x:x+w+1]
            regions_array.append([resize_region(region), (x,y,w,h)])       
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
    
    sorted_coordinates=sorted(coordinates, key=lambda coordinate: coordinate[0])
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    
    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions, sorted_coordinates

def prepareInputs(inputs):

    size=len(inputs)
    for i in range(size):
        input_bin =img_thl(inputs[i])
        img, contours, hierarchy = cv2.findContours(input_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if (area > 15 and h < 40 and h > 13 and w > 1):
                region = input_bin[y:y + h + 1, x:x + w + 1]
                region = resize_region(region)
                inputs[i]=region
                break

    return inputs

def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255. 
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image/255

def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()
    
def prepare_for_ann(regions):
    '''Regioni su matrice dimenzija 28x28 čiji su elementi vrednosti 0 ili 255.
        Potrebno je skalirati elemente regiona na [0,1] i transformisati ga u vektor od 784 elementa '''
    ready_for_ann = []
    for region in regions:
        # skalirati elemente regiona 
        # region sa skaliranim elementima pretvoriti u vektor
        # vektor dodati u listu spremnih regiona
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
        
    return ready_for_ann



def convert_output(alphabet):
    '''Konvertovati alfabet u niz pogodan za obučavanje NM,
        odnosno niz čiji su svi elementi 0 osim elementa čiji je
        indeks jednak indeksu elementa iz alfabeta za koji formiramo niz.
        Primer prvi element iz alfabeta [1,0,0,0,0,0,0,0,0,0],
        za drugi [0,1,0,0,0,0,0,0,0,0] itd..
    '''
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def create_ann():
    '''Implementacija veštačke neuronske mreže sa 784 neurona na uloznom sloju,
        128 neurona u skrivenom sloju i 10 neurona na izlazu. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(10, activation='softmax'))  
    ann.summary()

    return ann
    
def train_ann(ann, X_train, y_train):
    '''Obucavanje vestacke neuronske mreze'''
    
    X_train = np.array(X_train, np.float32)  # dati ulazi
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi za date ulaze

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    # ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, batch_size=256, epochs=25, verbose=1, shuffle=False)

      
    return ann

def winner(output): # output je vektor sa izlaza neuronske mreze
    '''pronaći i vratiti indeks neurona koji je najviše pobuđen'''
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs, alphabet):
    '''za svaki rezultat pronaći indeks pobedničkog
        regiona koji ujedno predstavlja i indeks u alfabetu.
        Dodati karakter iz alfabet u result'''
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result

def isCaught(numDet,myNumbers):
    closeNumbers = []
    numDetBTC=[numDet.get_x() + numDet.get_w(), numDet.get_y() + numDet.get_h()]

    for number in myNumbers:
        numberBTC=[number.get_x() + number.get_w(), number.get_y() + number.get_h()]
        distance = math.sqrt( (numberBTC[0] - numDetBTC[0])**2 + (numberBTC[1] - numDetBTC[1])**2 )
        if distance < 20:
            closeNumbers.append([distance, number])

    closeNumbers = sorted(closeNumbers, key=lambda num: num[0])   

    if len(closeNumbers) > 0:
        return closeNumbers[0][1]
    else:
        return None

def clearList(myNumbers):
    newList=[]

    for number in myNumbers:
        numberBTC=[number.get_x() + number.get_w(), number.get_y() + number.get_h()]
        if(not(numberBTC[0] > 630 or numberBTC[1] > 470)):
            newList.append(number)
            #print('HOP')
            #print(number.get_value())

    return newList





    

