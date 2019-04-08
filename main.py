
from functions import *
from keras import models
from keras.utils import np_utils
from numberTracer import NumberTracer
import math
import os

alphabet = [0,1,2,3,4,5,6,7,8,9]

def makeNeuralNetwork():

    if os.path.exists('ann.h5') is not False:
        ann=models.load_model('ann.h5')
        return ann

    print('-----------------------------------------------')
    print('DATASET Loading...')
    (x_train, y_train), (x_test, y_test)=keras.datasets.mnist.load_data()
    
    print('-----------------------------------------------')
    print('DATASET loaded')
    
    input_train=prepareInputs(x_train)    
    input_train=prepare_for_ann(x_train)
    outputs = np_utils.to_categorical(y_train, 10)
    
    ann = create_ann()
    ann=train_ann(ann, input_train, outputs)

    ann.save('ann.h5')

    return ann
    


ann=makeNeuralNetwork()
videosNames=[]
results=[]
myNumbers=[]

def main(path,myNumbers):
    frame_num = 0
    cap = cv2.VideoCapture(path)
    cap.set(1, frame_num) # indeksiranje frejmova

    ret_val, frame = cap.read()
    #cv2.imshow('aaa',frame)
    if(ret_val==True):
        x1g,y1g,x2g,y2g,kg,ng=findGreenLine(frame=frame.copy())
        x1b,y1b,x2b,y2b,kb,nb=findBlueLine(frame=frame.copy())
    print(kg)
    print(ng)
    
    video_result=0
    detectedNumbers=[]
    
    
     # analiza videa frejm po frejm
    while True:
        frame_num += 1
        ret_val, frame = cap.read()
        #cv2.imshow('aaa',frame)
        if ret_val==True:
            

            frame_bin = image_bin(image_gray(frame))
            selected_regions, numbers, coordinates = select_roi(frame.copy(), frame_bin)

            novi=[]
            for num in numbers:
                novi.append(resize_region(num))
            numbers=novi
            
            temp_frame=frame.copy()
            ready_inputs=prepare_for_ann(numbers)
            results=ann.predict(np.array(ready_inputs, np.float32))
            values=display_result(results, alphabet)
            
            i=0
            for coordinate in coordinates:
                
                if coordinate[3] > 10:
                    numDet = NumberTracer(coordinate)
                    numDet.set_value(values[i])
                    #provera da li ima brojeva u blizini
                    isPresent = isCaught(numDet,myNumbers)
            
                i=i+1
                #ako nema dodaj kao novi
                if isPresent is None:
                
                    myNumbers.append(numDet)

                #setuj nove koordinate ako je vec uhvacen
                else:
                    isPresent.set_coordinates(coordinate)
            
            myNumbers=clearList(myNumbers)

            for numDet in myNumbers:
                numDetBTC=[numDet.get_x() + numDet.get_w(), numDet.get_y() + numDet.get_h()]
                coordinate=numDet.get_coordinates()
                #print('leeen')
                #print(len(myNumbers))
                 #da li prelazi zelenu
                #if((xBRC+3 >= x1g or coordinate[0] <= x2g+1 ) and (yBRC+3>= y1g or coordinate[1]+1 >= y2g)):
                if (x2g+2>=numDetBTC[0] >=x1g-5 and y1g+5>=numDetBTC[1] >=y2g-1):
                    y=kg*(numDetBTC[0])+ng
                    if (abs(int(numDetBTC[1])-int(y))<= 2 and not numDet.get_green()):
                        #test_input=prepare_for_ann([numbers[i]])
                        #result=ann.predict(np.array(test_input, np.float32))
                        #value=display_result(result, alphabet)
                        #value=value[0]
                        value=numDet.get_value()
                        #print(value)
                        cv2.rectangle(temp_frame,(coordinate[0],coordinate[1]),(coordinate[0]+coordinate[2],coordinate[1]+coordinate[3]),(0,255,0),2)
                        video_result-=value
                        numDet.set_green(True)
                        #print(numDet.get_green())
                
                #da li prelazi plavu
                #if((xBRC+3 >= x1g or x2g+1 >= coordinate[0]) and (yBRC+3>= y1b or coordinate[1]+1 >= y2b)):
                if (x2b+2>=numDetBTC[0] >=x1b-5 and y1b+5>=numDetBTC[1] >=y2b-1):
                    y=kb*(numDetBTC[0])+nb
                    if ((abs(int(numDetBTC[1])-int(y))<= 2) and not numDet.get_blue()):
                        #test_input=prepare_for_ann([numbers[i]])
                        #result=ann.predict(np.array(test_input, np.float32))
                        #value=display_result(result, alphabet)
                        #value=value[0]
                        value=numDet.get_value()
                        #print(value)
                        cv2.rectangle(temp_frame,(coordinate[0],coordinate[1]),(coordinate[0]+coordinate[2],coordinate[1]+coordinate[3]),(0,255,0),2)
                        video_result+=value
                        numDet.set_blue(True)
                        #print(numDet.get_blue())
               

            
            #cv2.imshow('Video', temp_frame)
            #cv2.waitKey(00) == ord('k')
            # Quit when 'q' is pressed
            cv2.waitKey(10)
            if cv2.waitKey(1) == ord('q'):
                break
            # ako frejm nije zahvacen
            if not ret_val:
                break
        else:
            break
        
        # dalje se sa frejmom radi kao sa bilo kojom drugom slikom, npr
        #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cap.release()
    #return sum_of_nums
    return video_result

for num in range(10):
    path='video-'+str(num)+'.avi'
    videosNames.append(path)
    result=main(path=path,myNumbers=myNumbers)
    results.append(result)
    myNumbers=[]
    print('Result=')
    print(result)
    print('--------------------------------')
    #results.append(main(path))

outFile = open('out.txt', 'w')
text = 'RA 47/2015 Mirko Ivic\n'
text+='file\t\tsum\n'


for cnt in range(10):
    text+=videosNames[cnt]
    text +='\t'
    text+= str(results[cnt])
    text+='\n'

outFile.write(text)
outFile.close()

import test


