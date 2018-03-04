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
actMem=[]
wgtMem=[]
fwdcyc=[]
bckcyc=[]
wgtcyc=[]
fwdactacc=[]
bckactacc=[]
wgtactacc=[]
fwdwgtacc=[]
bckwgtacc=[]
wgtwgtacc=[]
print index
print bnID
for idx,id in enumerate(bnID):
    actMem.append(Monarch.GetPEInputActMem(index[idx])[0])
    wgtMem.append(Monarch.GetPEWeightMem(index[idx])[0])
    fwdcyc.append(Monarch.GetPECycles(index[idx])[0][0])
    bckcyc.append(Monarch.GetPECycles(index[idx])[1][0])
    wgtcyc.append(Monarch.GetPECycles(index[idx])[2][0])
    #combine activation and gradient
    print "index is" + str(idx)
    print Monarch.GetPEActAccesses(index[idx])
    print Monarch.GetPEGradAccesses(index[idx])
    fwdactacc.append(Monarch.GetPEActAccesses(index[idx])[0][0]+Monarch.GetPEGradAccesses(index[idx])[0][0])
    bckactacc.append(Monarch.GetPEActAccesses(index[idx])[1][0]+Monarch.GetPEGradAccesses(index[idx])[1][0])
    wgtactacc.append(Monarch.GetPEActAccesses(index[idx])[2][0]+Monarch.GetPEGradAccesses(index[idx])[2][0])
    fwdwgtacc.append(Monarch.GetPEWeightAccesses(index[idx])[0][0])
    bckwgtacc.append(Monarch.GetPEWeightAccesses(index[idx])[1][0])
    wgtwgtacc.append(Monarch.GetPEWeightAccesses(index[idx])[2][0])            
print actMem
#Monarch.PlotOps('Activation Memory Per BN Op',bnID,'Memory in MB',actMem)
#Monarch.PlotOps('Weight Memory Per BN Op',bnID,'Memory in MB',wgtMem)
#Monarch.PlotOps('Forward Pass Cycles',bnID,'cycles',fwdcyc)
#Monarch.PlotOps('Backward Pass Cycles',bnID,'cycles',bckcyc)
#Monarch.PlotOps('Weight Update Cycles',bnID,'cycles',wgtcyc
#Monarch.PlotOps('Activation and Gradient Accesses Fwd Bn',bnID,'cycles',fwdactacc)
#Monarch.PlotOps('Activation and Gradient Accesses Bck Bn',bnID,'cycles',bckactacc)
#Monarch.PlotOps('Activation and Gradient Accesses Wgt Bn',bnID,'cycles',wgtactacc)
#Monarch.PlotOps('Weight Accesses Fwd Bn',bnID,'cycles',fwdwgtacc)
#Monarch.PlotOps('Weight Accesses Bck Bn',bnID,'cycles',bckwgtacc)
#Monarch.PlotOps('Weight Accesses Wgt Bn',bnID,'cycles',wgtwgtacc)


(index,FID)=Monarch.GetFiltIDs()
actMem=[]
wgtMem=[]
fwdcyc=[]
bckcyc=[]
wgtcyc=[]
actacc=[]
wgtacc=[]
fwdactacc=[]
bckactacc=[]
wgtactacc=[]
fwdwgtacc=[]
bckwgtacc=[]
wgtwgtacc=[]
print index
print FID
for idx,id in enumerate(FID):
    actMem.append(Monarch.GetPEInputActMem(index[idx])[0])
    wgtMem.append(Monarch.GetPEWeightMem(index[idx])[0])
    fwdcyc.append(Monarch.GetPECycles(index[idx])[0][0])
    bckcyc.append(Monarch.GetPECycles(index[idx])[1][0])
    wgtcyc.append(Monarch.GetPECycles(index[idx])[2][0])
    #combine activation and gradient
    fwdactacc.append(Monarch.GetPEActAccesses(index[idx])[0][0]+Monarch.GetPEGradAccesses(index[idx])[0][0])
    bckactacc.append(Monarch.GetPEActAccesses(index[idx])[1][0]+Monarch.GetPEGradAccesses(index[idx])[1][0])
    wgtactacc.append(Monarch.GetPEActAccesses(index[idx])[2][0]+Monarch.GetPEGradAccesses(index[idx])[2][0])
    actacc.append(sum(Monarch.GetPEActAccesses(index[idx])[0])+sum(Monarch.GetPEGradAccesses(index[idx])[0]))
    wgtacc.append(sum(Monarch.GetPEWeightAccesses(index[idx])[0]))
    fwdwgtacc.append(Monarch.GetPEWeightAccesses(index[idx])[0][0])
    bckwgtacc.append(Monarch.GetPEWeightAccesses(index[idx])[1][0])
    wgtwgtacc.append(Monarch.GetPEWeightAccesses(index[idx])[2][0])                
#Monarch.PlotOps('Activation Memory Per Filt Op',FID,'Memory in MB',actMem)
#Monarch.PlotOps('Weight Memory Per Filt Op',FID,'Memory in MB',wgtMem)
#Monarch.PlotOps('Forward Pass Cycles',FID,'cycles',fwdcyc)
#Monarch.PlotOps('Backward Pass Cycles',FID,'cycles',bckcyc)
#Monarch.PlotOps('Weight Update Cycles',FID,'cycles',wgtcyc)
#Monarch.PlotOps('Activation and Gradient Accesses Fwd Op',FID,'cycles',fwdactacc)
#Monarch.PlotOps('Activation and Gradient Accesses Bck Op',FID,'cycles',bckactacc)
#Monarch.PlotOps('Activation and Gradient Accesses Wgt Op',FID,'cycles',wgtactacc)
#Monarch.PlotOps('Weight Accesses Fwd Op',FID,'cycles',fwdwgtacc)
#Monarch.PlotOps('Weight Accesses Bck Op',FID,'cycles',bckwgtacc)
#Monarch.PlotOps('Weight Accesses Wgt Op',FID,'cycles',wgtwgtacc)
Monarch.PlotOps('Total Act Accesses Op',FID,'cycles',actacc)
Monarch.PlotOps('Total Weight Accesses Op',FID,'cycles',wgtacc)


#Monarch.PlotOps('total Memory per Op',Monarch.opIDs,'Memory in MB',Monarch.GetPEMem())
#Monarch.PlotOps('Weight Memory in PEs',Monarch.opIDs,'Memory in MB',Monarch.GetPEWeightMem())
#Monarch.PlotOps('Total forward cycles per Op',Monarch.opIDs,'Memory in MB',Monarch.GetPECycles()[0])
#print Monarch.GetPEWeightMem()


#TODO
#Get what he asked for with PROW = PCOL=1
#then:
#-lets look at total memory for all non BN operators (HW vs CK)
#-lets look at total memory for all BN operators (HW vs CK)
#-lets look at total cycles for all non BN operators (HW vs CK) then best of two WS vs AS
#-lets look at total cycles for all BN operators (HW vs CK)



