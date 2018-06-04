import ArchLib
import numpy as np
print("Running Controller \n")

C = ArchLib.Core(4)
#C.PrintPEGrid()

np.random.seed(0)

#input activation dict
InAct=dict()
OutAct=dict()
LayerWeights=dict()
#_----------------------------
#NOTE all arrays are in (C,H,W) and (K,C,H,W)
#-----------------------------
#create a 7x7x16 sample activation
act = np.random.randint(5,size=(16,3,3))
InAct[0]=act
#create a 16,16,3,3  sample weight
LayerWeights[0]=np.random.randint(5,size=(16,16,2,2))
#---------------------------------------
#basic test of convolution functionality
testact=np.random.randint(5,size=(2,3,3))
#print(testact)
#testweights=np.random.randint(5,size=(2,2,2,2))
#print(testweights)
#testout=ArchLib.Convolve(testact,testweights)
#print(testout)
#---------------------------------------

#------------------------------------------------------------------
#as a baseline, perform forward pass convolutions
#print out all activations for debug
#print ("Original Activations")
#print(InAct[0])
#print out weights for debug
#print ("Original weights")
#print (LayerWeights[0])

baseline=ArchLib.Convolve(InAct[0],LayerWeights[0])
#print ("baseline --------")
#print (baseline)
#-------------------------------------------------------------------
#random gradient for backwards pass testing
gradient = np.random.randint(3,size=baseline.shape)
base_dact,base_dw = ArchLib.BackConvolve(gradient,InAct[0],LayerWeights[0])
print("The baseline output gradient is:")
print(base_dact)
#------------------------------------------------------------------
#split that activaiton in the C dimension and load into PEs
#first row gets C=0,1,2,3 etc
#split the output grad in the K dimension to match for backward pass
for i in range(0,4):
    for j in range(0,4):
        C.PEGrid[i,j].SetInAct(0,InAct[0][(i*4)+j,:,:])
        C.PEGrid[i,j].inGrad[0]=base_dact[(i*4)+j,:,:]
        
#split the weights in K and C dimension and load into PEs
for i in range(0,4):
    for j in range(0,4):
        #TODO K blocked loop insetad of hard coded
        C.PEGrid[i,j].weights[0]=LayerWeights[0][j,(4*i):(4*(i+1)),:,:]
        C.PEGrid[i,j].weights[1]=LayerWeights[0][j+4,(4*i):(4*(i+1)),:,:]
        C.PEGrid[i,j].weights[2]=LayerWeights[0][j+8,(4*i):(4*(i+1)),:,:]
        C.PEGrid[i,j].weights[3]=LayerWeights[0][j+12,(4*i):(4*(i+1)),:,:]

#-----------------FORWARD PASS START---------------------------------
#Each PE convolves its input activations with its local weights
C.Forward(0)
#Check forward convolution output against golden
error=0
for i in range(0,4):
    for j in range(0,4):
        if np.array_equal(baseline[(4*i)+j,:,:],C.PEGrid[i,j].outAct[0])!=1:
            print ("UH OH ISSUE WITH OUTPUT INDEX {K}".format(K=(4*i)+j))
            print(baseline[0,:,:])
            print(C.PEGrid[i,j].outAct[0])
            error=1
if error==0:
    print("forward pass convoultion output matches golden!!")
#-----------------FORWARD PASS END---------------------------------


#-----------------BACKWARD PASS ---------------------------------