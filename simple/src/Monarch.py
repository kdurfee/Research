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

#VERIF

#layer HW_0 at index 0
#input act should be 28*28*3*16*.000000125=.004704
#output act should be 28*28*64*16*.000000125=.100352
#input grad should be 28*28*64*16*.000000125=.100352
#output grad should be 28*28*3*16*.000000125=.004704
#weights should be 7*7*3*64*16*.000000125=.018816
print '------ HW Conv ------ '
print Monarch.GetPEInputActMem(0)
print Monarch.GetPEOutputActMem(0)
print Monarch.GetPEInputGradMem(0)
print Monarch.GetPEOutputGradMem(0)
print Monarch.GetPEWeightMem(0)

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

print '------Conv ------'
print Monarch.GetPEInputActMem(1)
print Monarch.GetPEOutputActMem(1)
print Monarch.GetPEInputGradMem(1)
print Monarch.GetPEOutputGradMem(1)
print Monarch.GetPEWeightMem(1)

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
print '------Batch ------'
print Monarch.GetPEInputActMem(2)
print Monarch.GetPEOutputActMem(2)
print Monarch.GetPEInputGradMem(2)
print Monarch.GetPEOutputGradMem(2)
print Monarch.GetPEWeightMem(2)


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



