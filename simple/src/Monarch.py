import xml.etree.ElementTree as ET
import argparse
from Cocoon import *


#parse input arguments to get XML design file and build options
parser = argparse.ArgumentParser(description='Analyze performance of a network specified in RTL format on the caterpillar architecture')
parser.add_argument('xml_file',help='absolute path of an XML file containing a valid network description')
args = parser.parse_args()

#first create the high level architecture object
Monarch = Network()
Monarch.ParseXMLNetwork(args.xml_file)
#sanity check via print outs
Monarch.PrintNetwork()

#VERIF for 8x8

#layer HW_0 at index 0
#input act should be 28*28*3*16*.000000125=.004704
#output act should be 28*28*64*16*.000000125=.100352
#input grad should be 28*28*64*16*.000000125=.100352
#output grad should be 28*28*3*16*.000000125=.004704
#weights should be 7*7*3*64*16*.000000125=.018816
#print '------ HW Conv ------ '
#print Monarch.GetPEInputActMem(0)
#print Monarch.GetPEOutputActMem(0)
#print Monarch.GetPEInputGradMem(0)
#print Monarch.GetPEOutputGradMem(0)
#print Monarch.GetPEWeightMem(0)

#RES_1_block_0_Cnv_0 at index 1
#---HW
#input act=.006272
#output act=.006272
#input grad=.006272
#output grad=.006272
#weights=.008192
#---CK
#input act=.050176
#output act=.050176
#input grad=.050176
#output grad=.050176
#weights=.0000128

#print '------Conv ------'
#print Monarch.GetPEInputActMem(1)
#print Monarch.GetPEOutputActMem(1)
#print Monarch.GetPEInputGradMem(1)
#print Monarch.GetPEOutputGradMem(1)
#print Monarch.GetPEWeightMem(1)

#RES_1_block_0_Bn_0 at index 2
#CK
#input act=.6272
#output act=.6272
#input grad=.6272
#output grad=.6272
#weights=
#HW
#input act=5.0176
#output act=5.0176
#input grad=5.0176
#output grad=5.0176
#weights=
#print '------Batch ------'
#print Monarch.GetPEInputActMem(2)
#print Monarch.GetPEOutputActMem(2)
#print Monarch.GetPEInputGradMem(2)
#print Monarch.GetPEOutputGradMem(2)
#print Monarch.GetPEWeightMem(2)

(index,bnID)=Monarch.GetBNIDs()
BNactMem=[]
BNwgtMem=[]
BNfwdcyc=[]
BNbckcyc=[]
BNwgtcyc=[]
BNfwdactacc=[]
BNbckactacc=[]
BNwgtactacc=[]
BNfwdwgtacc=[]
BNbckwgtacc=[]
BNwgtwgtacc=[]
print index
print bnID
for idx,id in enumerate(bnID):
    BNactMem.append(Monarch.GetPEInputActMem(index[idx])[0])
    BNwgtMem.append(Monarch.GetPEWeightMem(index[idx])[0])
    BNfwdcyc.append(Monarch.GetPECycles(index[idx])[0][0])
    BNbckcyc.append(Monarch.GetPECycles(index[idx])[1][0])
    BNwgtcyc.append(Monarch.GetPECycles(index[idx])[2][0])
    #combine activation and gradient
    BNfwdactacc.append(Monarch.GetPEActAccesses(index[idx])[0][0]+Monarch.GetPEGradAccesses(index[idx])[0][0])
    BNbckactacc.append(Monarch.GetPEActAccesses(index[idx])[1][0]+Monarch.GetPEGradAccesses(index[idx])[1][0])
    BNwgtactacc.append(Monarch.GetPEActAccesses(index[idx])[2][0]+Monarch.GetPEGradAccesses(index[idx])[2][0])
    BNfwdwgtacc.append(Monarch.GetPEWeightAccesses(index[idx])[0][0])
    BNbckwgtacc.append(Monarch.GetPEWeightAccesses(index[idx])[1][0])
    BNwgtwgtacc.append(Monarch.GetPEWeightAccesses(index[idx])[2][0])            
#Monarch.PlotOps('Activation Memory Per BN Op',bnID,'Memory in MB',BNactMem)
#Monarch.PlotOps('Weight Memory Per BN Op',bnID,'Memory in MB',BNwgtMem)
#Monarch.PlotOps('Forward Pass Cycles',bnID,'cycles',BNfwdcyc)
#Monarch.PlotOps('Backward Pass Cycles',bnID,'cycles',BNbckcyc)
#Monarch.PlotOps('Weight Update Cycles',bnID,'cycles',BNwgtcyc
#Monarch.PlotOps('Activation and Gradient Accesses Fwd Bn',bnID,'cycles',BNfwdactacc)
#Monarch.PlotOps('Activation and Gradient Accesses Bck Bn',bnID,'cycles',BNbckactacc)
#Monarch.PlotOps('Activation and Gradient Accesses Wgt Bn',bnID,'cycles',BNwgtactacc)
#Monarch.PlotOps('Weight Accesses Fwd Bn',bnID,'cycles',BNfwdwgtacc)
#Monarch.PlotOps('Weight Accesses Bck Bn',bnID,'cycles',BNbckwgtacc)
#Monarch.PlotOps('Weight Accesses Wgt Bn',bnID,'cycles',BNwgtwgtacc)


(index,FID)=Monarch.GetFiltIDs()
OPmem=[]
OPactMem=[]
OPwgtMem=[]
OPfwdcyc=[]
OPbckcyc=[]
OPwgtcyc=[]
OPbckwgtcyc=[]
OPactacc=[]
OPwgtacc=[]
OPfwdactacc=[]
OPbckactacc=[]
OPwgtactacc=[]
OPfwdwgtacc=[]
OPbckwgtacc=[]
OPwgtwgtacc=[]
print index
print FID
for idx,id in enumerate(FID):
    OPactMem.append(Monarch.GetPEInputActMem(index[idx])[0])
    OPwgtMem.append(Monarch.GetPEWeightMem(index[idx])[0])
    OPfwdcyc.append(Monarch.GetPECycles(index[idx])[0][0])
    OPbckcyc.append(Monarch.GetPECycles(index[idx])[1][0])
    OPwgtcyc.append(Monarch.GetPECycles(index[idx])[2][0])
    OPbckwgtcyc.append(Monarch.GetPECycles(index[idx])[1][0]+Monarch.GetPECycles(index[idx])[2][0])
    #combine activation and gradient
    OPfwdactacc.append(Monarch.GetPEActAccesses(index[idx])[0][0]+Monarch.GetPEGradAccesses(index[idx])[0][0])
    OPbckactacc.append(Monarch.GetPEActAccesses(index[idx])[1][0]+Monarch.GetPEGradAccesses(index[idx])[1][0])
    OPwgtactacc.append(Monarch.GetPEActAccesses(index[idx])[2][0]+Monarch.GetPEGradAccesses(index[idx])[2][0])
    OPactacc.append(sum(Monarch.GetPEActAccesses(index[idx])[0])+sum(Monarch.GetPEGradAccesses(index[idx])[0]))
    OPwgtacc.append(sum(Monarch.GetPEWeightAccesses(index[idx])[0]))
    OPfwdwgtacc.append(Monarch.GetPEWeightAccesses(index[idx])[0][0])
    OPbckwgtacc.append(Monarch.GetPEWeightAccesses(index[idx])[1][0])
    OPwgtwgtacc.append(Monarch.GetPEWeightAccesses(index[idx])[2][0])
    OPmem.append(Monarch.GetPEInputActMem(index[idx])[0]+Monarch.GetPEWeightMem(index[idx])[0])
#Monarch.PlotOps('Total Memory Per Filt Op',FID,'Memory in MB',OPmem)
#Monarch.PlotOps('Activation Memory Per Filt Op',FID,'Memory in MB',OPactMem)
#Monarch.PlotOps('Weight Memory Per Filt Op',FID,'Memory in MB',OPwgtMem)
#Monarch.PlotOps('Forward Pass Cycles',FID,'cycles',OPfwdcyc)
#Monarch.PlotOps('Backward Pass Cycles',FID,'cycles',OPbckcyc)
#Monarch.PlotOps('Weight Update Cycles',FID,'cycles',OPwgtcyc)
#Monarch.PlotOps('Back Prop and Weight Update Cycles',FID,'cycles',OPbckwgtcyc)
#Monarch.PlotOps('Activation and Gradient Accesses Fwd Op',FID,'cycles',OPfwdactacc)
#Monarch.PlotOps('Activation and Gradient Accesses Bck Op',FID,'cycles',OPbckactacc)
#Monarch.PlotOps('Activation and Gradient Accesses Wgt Op',FID,'cycles',OPwgtactacc)
#Monarch.PlotOps('Weight Accesses Fwd Op',FID,'cycles',OPfwdwgtacc)
#Monarch.PlotOps('Weight Accesses Bck Op',FID,'cycles',OPbckwgtacc)
#Monarch.PlotOps('Weight Accesses Wgt Op',FID,'cycles',OPwgtwgtacc)
#Monarch.PlotOps('Total Act Accesses Op',FID,'cycles',OPactacc)
#Monarch.PlotOps('Total Weight Accesses Op',FID,'cycles',OPwgtacc)


#------------- Checkpointing -----------------------#
#when we checkpoint, we have to redo forward pass computations (2x) for some layers
#and the tradeoff is that we do not have to store the input activations for those layers

#for conv layers only we can look at forward related cycles
#and we can look at activation memory
OPcheckFwdCycle=[]
OPcheckActMem=[]
for x in range(0,len(OPfwdcyc[1:])):#just skip the first big conv layer for this
    if(x%2==0):
        OPcheckFwdCycle.append(2*OPfwdcyc[1+x])
        OPcheckActMem.append(OPactMem[1+x])
    else:
        OPcheckFwdCycle.append(2*OPfwdcyc[1+x])
        OPcheckActMem.append(0)
#Monarch.PlotOps('Conv Forward Pass Cycles with Checkpointing',FID[1:],'cycles',OPcheckFwdCycle)
#Monarch.PlotOps('Conv Activation Memory with Checkpointing',FID[1:],'Mem in MB',OPcheckActMem)
#print 'total forward cycles with check is : ' + str(sum(OPcheckFwdCycle)) + ' and without it is : '+ str(sum(OPfwdcyc))
#print 'total act mem with check is : ' + str(sum(OPcheckActMem)) + ' and without it is : '+ str(sum(OPactMem))

#----------
#Checkpointing with batch
#see slides, here is an example for a CK, BN, CK, BN combination
#using indices 1,2,3,4,5
#RES_1_block_0_Cnv_0
#RES_1_block_0_Bn_0
#RES_1_block_0_Cnv_1
#RES_1_block_0_Bn_1
#RES_1_block_0_Cnv_2

CTidx=[1,2,3,4,5]
#check test control
CTCcyc=[]
CTCmem=[]
#normal CYC operations
#normal memory
for idx in CTidx:
    #forward, backward, and weight update CYC are normal for batch
    if idx == 2 or idx==4:
        CTCcyc.append(Monarch.GetPECycles(idx)[0][0]+Monarch.GetPECycles(idx)[1][0]+Monarch.GetPECycles(idx)[2][0])
    #100X for conv layers to give batch totals
    else:
        CTCcyc.append(100*(Monarch.GetPECycles(idx)[0][0]+Monarch.GetPECycles(idx)[1][0]+Monarch.GetPECycles(idx)[2][0]))

    #memory is activations and weights for 1 and 3 and 5
    if idx==1 or idx==3 or idx==5:
        CTCmem.append(Monarch.GetPEInputActMem(idx)[0] + Monarch.GetPEWeightMem(idx)[0])
    else:    #memory is activations for 2 and 4, notice that we need a buffer the size of largest activation memory for gradients in addition to this
        CTCmem.append(Monarch.GetPEInputActMem(idx)[0] + Monarch.GetPEWeightMem(idx)[0])

        
#Monarch.PlotOps('CYC Operations No Checkpointing',CTidx,'cycles',CTCcyc)
#Monarch.PlotOps('Memory No Checkpointing',CTidx,'Total Memory MB',CTCmem)
print 'total CYC is:'+str(sum(CTCcyc))
print 'total Mem is:'+str(sum(CTCmem))

#check test
CTcyc=[]
CTmem=[]
for idx,val in enumerate(CTidx):
    #only BN layer changes mem
    if val!=4:
        CTmem.append(CTCmem[idx])
    else: #for ones that are checkpointing, don't need
        CTmem.append(Monarch.GetPEWeightMem(val)[0])#basically zero
    #only the middle BN layer and surrounding conv change cyc
    #BN layes do same amount of computation
    if val==1 or val==2 or val==4:
        CTcyc.append(CTCcyc[idx])
    elif val==3: #forward pass on incoming is 4X the CYC in fwd pass
        CTcyc.append(100*(4*Monarch.GetPECycles(val)[0][0]+Monarch.GetPECycles(val)[1][0]+Monarch.GetPECycles(val)[2][0]))
    elif val==5: #backward pass
        CTcyc.append(100*(Monarch.GetPECycles(val)[0][0]+Monarch.GetPECycles(val)[1][0]+Monarch.GetPECycles(val)[2][0]))

Monarch.PlotOps('CYC Operations Checkpointing',CTidx,'cycles',CTcyc)
#Monarch.PlotOps('Memory Checkpointing',CTidx,'Total Memory MB',CTmem)
print 'total CYC is:'+str(sum(CTcyc))
print 'total Mem is:'+str(sum(CTmem))        








