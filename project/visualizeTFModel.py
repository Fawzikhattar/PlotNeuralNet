import tensorflow as tf
import sys
sys.path.append('../')
from pycore.tikzeng import *
import pdb
from collections import defaultdict


def recursiveLayerConnectionFinder(modelConfigFile,layerName,availableLayers=['Concatenate','InputLayer','SeparableConv2D','Conv2D','Dense','Flatten','MaxPooling2D','AveragePooling2D']):
     
    for l in modelConfigFile:
        if(l["name"]==layerName):
            if(l['class_name'] in availableLayers):
                return l["name"]
            else:
                return recursiveLayerConnectionFinder(modelConfigFile,l['inbound_nodes'][0][0][0])


def getDataOfLayerByName(name,layers):
    for l in layers:
        if(l.name==name):
            _,outH,outW,outD=l.output_shape
            return _,outH,outW,outD



def generateTex(modelPath,nameFile="out.tex",putCaption=False):
    model=tf.keras.models.load_model(modelPath)
    arch=[]
    prevDepth=0
    ##prevLayerName="Lambda"
    layerOutputCount=defaultdict(int)
    #print(model.get_config()['layers'])
    for i,(lData,modelLayer) in enumerate(zip(model.get_config()['layers'],model.layers)):
        opType=lData['class_name']
        inputShape=modelLayer.input_shape
        if(opType=="Model" or opType=="Lambda"):
            continue
        #print(opType)
        if(opType=="BatchNormalization"):
            continue
        elif(opType!="Flatten" and opType!="Dense" ):
            
            _,outH,outW,outD=modelLayer.output_shape
            outDDisp=outD/30
            #print("outD:",outD,opType)
            
        else:
            #print(modelLayer)
            outL=modelLayer.output_shape[1]
            outLDisp=min(outL,100)
        #print("prevDepth:",prevDepth,opType)
        
        layerName=modelLayer.name
        #print(layerName)
        prevDepth=1 
        if(opType=='Activation'):
            #print(lData['inbound_nodes'])
            continue
        if(opType=='InputLayer'):
            arch.append(to_head( '..' ))
            arch.append(to_cor())
            arch.append(to_begin())
            #latexCmd=to_UnPool(layerName, offset="(0,0,0)", to="(0,0,0)", width=outDDisp, height=outH, depth=outW, opacity=0.5, caption="Input ")
            #arch.append(latexCmd)
            prevDepth=1           
            continue
        elif(opType=='UpSampling2D'):
            if(putCaption):
                caption="Up sample"
            else:
                caption=""

            fromLayer=recursiveLayerConnectionFinder(model.get_config()['layers'],lData['inbound_nodes'][0][0][0])
            to="(0,0,0)" if i<=1 else "("+str(fromLayer)+"-east)"
            latexCmd=to_UnPool(layerName, offset="("+str(prevDepth)+",0,0)", to=to, width=outDDisp, height=outH, depth=outW, opacity=0.5, caption=caption)

        elif(opType=='Concatenate'):
            if(putCaption):
                caption="Concatenate"
            else:
                caption=""
            tmpName=None
            #print(lData['inbound_nodes'])
            nbConcatLayer=len(lData['inbound_nodes'][0])
            for k,lay in enumerate(lData['inbound_nodes'][0]):
                lay=lay[0]
                if(isinstance(lay, str)):
                    
                    fromLayer=recursiveLayerConnectionFinder(model.get_config()['layers'],lay)
                    _,outH,outW,outD=getDataOfLayerByName(fromLayer,model.layers)
                    outDDisp=outD/30
                    if(tmpName is None):
                        tmpName=fromLayer
                    #print(lay,fromLayer,lData['inbound_nodes'][0][0])
                    to="(0,0,0)" if i<=1 else "("+str(tmpName)+"-east)"
                    if(k+1==nbConcatLayer):
                        arch.append(to_Concat(layerName, "","" , offset="("+str(0)+",0,0)", to=to,caption=caption,height=outHDisp, depth=outWDisp , width=outDDisp))
                    
                        ##print("layer",layerName," trace back to",fromLayer)
                        arch.append(to_skip(fromLayer,layerName))
                    elif(k==0):
                        outWDisp,outHDisp=outW,outH
                        arch.append(to_Concat(tmpName+"_"+str(k), "","" , offset="("+str(2)+",0,0)", to=to,caption=caption,height=outHDisp, depth=outWDisp , width=outDDisp ))

                        arch.append(to_skip(fromLayer,tmpName+"_"+str(k)))
                    else:
                        arch.append(to_Concat(tmpName+"_"+str(k), "","" , offset="("+str(0)+",0,0)", to=to,caption=caption,height=outHDisp, depth=outWDisp , width=outDDisp ))
                        #latexCmd=to_skip(lData['inbound_nodes'][0][0][0],lData['inbound_nodes'][0][1][0])
                        ##print("layer",layerName," trace back to",fromLayer)
                        arch.append(to_skip(fromLayer,tmpName+"_"+str(k)))
                    tmpName=tmpName+"_"+str(k) 
            continue

        elif(opType=='Conv2D' or opType=='SeparableConv2D'):
            strides=str(lData['config']['strides'][0])+"x"+str(lData['config']['strides'][1])

            kernelSize=str(lData['config']['kernel_size'][0])+"x"+str(lData['config']['kernel_size'][1])
             
            fromLayer=recursiveLayerConnectionFinder(model.get_config()['layers'],lData['inbound_nodes'][0][0][0])
            layerOutputCount[fromLayer]+=5
            to="(0,0,0)" if i<=1 else "("+str(fromLayer)+"-east)"
            if(putCaption):
                caption="$"+kernelSize+"$"
            else:
                caption="" 
                caption="$"+kernelSize+"$"

            if(opType=='SeparableConv2D'):
                latexCmd=to_SeparableConv(layerName, outW,outD , offset="("+str(prevDepth)+","+str(layerOutputCount[fromLayer]-5)+",0)", to=to,caption=caption ,height=outH, depth=outW , width=outDDisp )
            else:
                latexCmd=to_Conv(layerName, outW,outD , offset="("+str(prevDepth)+","+str(layerOutputCount[fromLayer]-5)+",0)", to=to,caption=caption ,height=outH, depth=outW , width=outDDisp )
        elif(opType=='MaxPooling2D' or opType=='AveragePooling2D'):

            fromLayer=recursiveLayerConnectionFinder(model.get_config()['layers'],lData['inbound_nodes'][0][0][0])
            
            layerOutputCount[fromLayer]+=10
            to="(0,0,0)" if i<=1 else "("+str(fromLayer)+"-east)"
            arch.append(to_Pool(layerName, offset="("+str(0)+","+str(layerOutputCount[fromLayer]-10)+",0)",to=to , height=outH, depth=outW, width=outDDisp))
            if(layerOutputCount[fromLayer]-10>0):
                arch.append(to_connection(fromLayer,layerName))
            continue

        elif(opType=='Flatten'):
            if(putCaption):
                caption="Flatten"
            else:
                caption=""

            fromLayer=recursiveLayerConnectionFinder(model.get_config()['layers'],lData['inbound_nodes'][0][0][0])
            to="(0,0,0)" if i<=1 else "("+str(fromLayer)+"-east)"
            latexCmd=to_Flatten(layerName, outL ,offset="("+str(2)+",0,0)", to=to, caption=caption,height=1,depth=outLDisp,width=1  )
            

            #pdb.set_trace()
        
        elif(opType=='Dense'):
            if(putCaption):
                caption="Dense"
            else:
                caption=""

            fromLayer=recursiveLayerConnectionFinder(model.get_config()['layers'],lData['inbound_nodes'][0][0][0])
            
            layerOutputCount[fromLayer]+=10
            to="(0,0,0)" if i<=1 else "("+str(fromLayer)+"-east)"
            latexCmd=to_SoftMax(layerName, outL ,offset="("+str(2)+","+ str(layerOutputCount[fromLayer]-10)+",0)", to=to,caption=caption,height=1,depth=outLDisp,width=1  )
        else:
            #print("[WARN]: Undefined layer "+ opType)
            continue

        """elif(opType=='Dense'):
            latexCmd=
        elif(opType=='Dense'):
            latexCmd=
        elif(opType=='Dense'):
            latexCmd=
        elif(opType=='Dense'):
            latexCmd=
        """
        
        arch.append(latexCmd)
        if(i>1 and opType!='MaxPooling2D' and opType!='Concatenate'): 
            
            fromLayer=recursiveLayerConnectionFinder(model.get_config()['layers'],lData['inbound_nodes'][0][0][0])
            arch.append(to_connection(fromLayer,layerName))
        #if(opType!='Concatenate'):
        ###prevLayerName=layerName
        prevDepth=2
        #if(layerName=="conv24seg"):
        #    break
        #prevDepth=2*outD
    arch.append(to_end())
    to_generate(arch, nameFile )



if(__name__=="__main__"):

    #generateTex("./models/beardModel.h5",nameFile="out.tex")
    #generateTex("./models/beardModel.h5",nameFile="out.tex")
    #generateTex("./models/glasses.h5",nameFile="out.tex")
    #generateTex("./models/hairColorModel.h5",nameFile="out.tex")
    generateTex("./models/modelAge.h5",nameFile="out.tex")
