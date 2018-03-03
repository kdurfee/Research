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
print '---input act mem---'
print Monarch.GetPEInputActMem()
print '---output act mem---'
print Monarch.GetPEOutputActMem()
print '---input grad mem---'
print Monarch.GetPEInputGradMem()
print '---output grad mem---'
print Monarch.GetPEOutputGradMem()
print '---weight mem---'
print Monarch.GetPEWeightMem()
print '---total mem---'
print Monarch.GetPEMem()
#print '---input act accesses--'
#print Monarch.GetPEInputActAccesses()
#print '---output act accesses--'
#print Monarch.GetPEOutputActAccesses()
#print '---input grad accesses--'
#print Monarch.GetPEInputGradAccesses()
#print '---output grad accesses--'
#print Monarch.GetPEOutputGradAccesses()
#print '---weight accesses --'
#print Monarch.GetPEWeightAccesses()
#print '---forward MAC --'
#print Monarch.GetPEForwardMAC()
#print '---backward MAC --'
#print Monarch.GetPEBackwardMAC()
#print '---forward Cycles --'
#print Monarch.GetPEForwardCycles()
#print '---backward Cycles --'
#print Monarch.GetPEBackwardCycles()
#print '---H systolic BW---'
#print Monarch.GetPEHSystolicBW()
#print '---V systolic BW---'
#print Monarch.GetPEVSystolicBW()
#print '---H Broadcast BW---'
#print Monarch.GetPEHBroadcastBW()
#print '---V Broadcast BW---'
#print Monarch.GetPEVBroadcastBW()
Monarch.PlotOps('Total Memory per Op','Memory in MB',Monarch.GetPEMem())
Monarch.PlotOps('Total Weight Memory per Op','Memory in MB',Monarch.GetPEWeightMem())



