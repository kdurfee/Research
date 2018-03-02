import xml.etree.ElementTree as ET
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from collections import defaultdict
import math
#Global value for conversions from bits to MB
MBCONVERSION = .000122070312

#activations and gradients connect all operator blocks
class Activation(object):
    def __init__(self,ID,H,W,C):
        #NOTE that 1,1,Z represents vector for FC layers
        self.H=H
        self.W=W
        self.C=C
        self.inputs=[] #operator modules whose outputs are this activation in forward pass
        self.outputs=[] #operator modules whose inputs are this activation in forward pass
        self.ID=ID #used when constructing graph from XML input, operators have IDs for inputs and outputs. Must be unique

    def AddInput(self,i):
        self.inputs.append(i)
    def AddOutput(self,o):
        self.outputs.append(o)
    #activations know how much memory they take up per PE

    #activations know how many times they are read by their output list
    
    #activaitons know how many times they are written to by their input list

    #TODO activations know how much power they burn based on reads, writes, and location (local memory or off chip)

    def PrintInfo(self):
        print "---Activation with ID " +str(self.ID) + ' and dimesions H:' +str(self.H) + ' W:' +str(self.W) + ' C:' +str(self.C)
        print "inputs: "
        for i in self.inputs:
            print str(i.ID)
        print "outputs: "
        for o in self.outputs:
            print str(o.ID)
        
class Gradient (object):
    def __init__(self,ID,H,W,C):
        #NOTE that 1,1,C represents vector for FC layers
        self.H=H
        self.W=W
        self.C=C
        self.inputs=[] #operator modules whose outputs are this gradient in back pass
        self.outputs=[] #operator modules whose inputs are this gradient in backward pass
        self.ID=ID #used when constructing graph from XML input, operators have IDs for inputs and outputs. Must be unique

    def AddInput(self,i):
        self.inputs.append(i)
    def AddOutput(self,o):
        self.outputs.append(o)
    #gradients know how much memory they take up per PE

    #gradients know how many times they are read from by their output list (just sum them up)

    #gradients know how many times they are written to by their input list (just sum them up)

    #TODO gradients know how much power they burn based on reads, writes, and location (local memory or off chip)

    def PrintInfo(self):
        print "---Gradient with ID " +str(self.ID) + ' and dimesions H:' +str(self.H) + ' W:' +str(self.W) + ' C:' +str(self.C)
        print "inputs: "
        for i in self.inputs:
            print str(i.ID)
        print "outputs: "
        for o in self.outputs:
            print str(o.ID)
    
#parent class for all operator blocks, they are all connected by activations and gradients
class Operator(object):
    def __init__(self,ID):
        self.ID=ID #for identification, printing etc. unique and in order that you want them printed / plotted
        self.inputActivations=[]
        self.inputGradients=[]
        self.outputActivations=[]
        self.outputGradients=[]
        self.core=None

    #which core is this associated with in the architecture.
    #this is done so that cores can have their costs per operation (layer) summed up at the end
    #if operators share inputs and are of the same core then they can share memeory (inception style parallel layers)
    #TODO how to deal with seperate cores sharing inputs (snake skip connection style... but what if not by 2)
    def SetCore(self,c):
        self.core=c
    def AddInputAct(self,a):
        self.inputActivations.append(a)
    def AddInputGrad(self,a):
        self.inputGradients.append(a)
    def AddOutputAct(self,a):
        self.outputActivations.append(a)
    def AddOutputGrad(self,a):
        self.outputGradients.append(a)
    def PrintInfo(self):
        print "---DEFAULT PRINT Operator ID is :" + str(self.ID) + ' and associated core is:' +str(self.core.ID)
        #STAT FUNCTIONS
    #counts for memory and cycles
    def GetInputActCount(self):
        pass
    def GetOutputActCount(self):
        pass
    def GetOutputGradCount(self):
        pass
    def GetInputGradCount(self):
        pass
    def GetWeightCount(self):
        pass
    #accesses for power and cycles
    def GetInputActAccesses(self):
        pass
    def GetOutputActAccesses(self):
        pass
    def GetInputGradAccesses(self):
        pass
    def GetOutputGradAccesses(self):
        pass
    def GetWeightAccesses(self):
        pass
    #MAC for power and cycles
    def GetForwardMAC(self):
        pass
    def GetBackwardMAC(self):
        pass
    #cycles
    def GetForwardCycles(self):
        pass
    def GetBackwardCycles(self):
        pass
    #BW
    def GetHSystolicBW(self):
        pass
    def GetVSystolicBW(self):
        pass    
    def GetHBroadcastBW(self):
        pass
    def GetVBroadcastBW(self):
        pass
    
class HWConv(Operator):
    def __init__(self,ID,R,S,C,K):
        super(HWConv,self).__init__(ID)
        self.R=R
        self.S=S
        self.C=C
        self.K=K
    def PrintInfo(self):
        print "---HWCOnv Operator ID is :" + str(self.ID) + ' and associated core is:' +str(self.core.ID)
        print '----- R: '+str(self.R)+ ' S:' + str(self.S) + ' K:' +str(self.K)
        print '-----inputActivations: ----'
        print self.inputActivations
        print '-----outputActivations: ----'
        print self.outputActivations        
    def GetInputActCount(self):
        count=0
        for ia in self.inputActivations:
            count += math.ceil(ia.H/self.core.pCol) * math.ceil(ia.W/self.core.pRow) * ia.C
        return count
    def GetOutputActCount(self):
        count=0
        for oa in self.outputActivations:
            count += math.ceil(oa.H/self.core.pCol) * math.ceil(oa.W/self.core.pRow) * oa.C
        return count
    def GetInputGradCount(self):
        count=0
        for ig in self.inputGradients:
            count += math.ceil(ig.H/self.core.pCol) * math.ceil(ig.W/self.core.pRow) * ig.C
        return count
    def GetOutputGradCount(self):
        count=0
        for og in self.outputGradients:
            count += math.ceil(og.H/self.core.pCol) * math.ceil(og.W/self.core.pRow) * og.C
        return count
    def GetWeightCount(self):
        count=self.R * self.S * self.K * self.C
        return count
    
    def GetInputActAccesses(self):
        acc=0
        #we read the input activations all once per weight (weight stationary) in the forward pass to calculate output
        acc+= self.GetInputActCount() * self.R * self.S * self.K
        #we read the input activations once per output gradient to calculate weight updates (output gradient stationary)
        if len(self.outputGradients)!=0:
            acc += self.GetInputActCount() * math.ceil(self.outputGradients[0].H/self.core.pCol) * math.ceil(self.outputGradients[0].W/self.core.pRow) * self.K
        return acc
    def GetOutputActAccesses(self):
        acc=0
        #output activations are written to with forward pass results and accumulated for each operation on input activations
        acc+=self.GetInputActCount() * self.R * self.S * self.K
        return acc
    def GetInputGradAccesses(self):
        acc=0
        #input gradients are read in the backward pass to produce output gradients
        acc+= self.GetInputGradCount() * self.R * self.S * self.C
        return acc
    def GetOutputGradAccesses(self):
        acc=0
        #output gradients are written to in the backward pass to produce partial gradient for each K
        acc+=self.GetInputGradCount() * self.R * self.S * self.C
        #output gradients are read from to calculate weight update values (gradient stationary so once each)
        if len(self.outputGradients)!=0:
            acc += math.ceil(self.outputGradients[0].H/self.core.pCol) * math.ceil(self.outputGradients[0].W/self.core.pRow) * self.K
        #output gradients are them accumulated in the K dimension
        self.GetOutputGradCount() * self.K
        return acc
    def GetWeightAccesses(self):
        acc=0
        #read weights once each in forward pass (weight stationary)
        acc+=self.GetWeightCount()
        #read weights once each in backward pass (weight stationary)
        acc+=self.GetWeightCount()
        #RMW to weights for updates
        acc+=2*self.GetWeightCount()
        return acc
    def GetForwardMAC(self):
        mac=0
        #every local weight is multiplied by input activations
        mac += self.GetInputActCount() * self.R * self.S * self.K
        return mac
    def GetBackwardMAC(self):
        mac=0
        #calculate per K output gradients
        mac += self.GetInputGradCount() * self.R * self.S * self.C
        #calculate weight update values
        mac+=self.GetInputActCount() * self.GetOutputGradCount()/self.C #each input is convolved by a spatial H,W tile of outputs, but one conv window
        return mac
    def GetForwardCycles(self):
        cycles=0
        #cycles for forward PE
        cycles += self.GetForwardMAC()
        #H,W has to share halos with adjacent. send/recieve overlap
        for ia in self.inputActivations:
            cycles += math.ceil(ia.H/self.core.pCol)*(self.R-1) + math.ceil(ia.W/self.core.pRow) * (self.S-1) * self.K
        #NOTE transition from HW to CK done in seperate operator
        return cycles
    def GetBackwardCycles(self):
        cycles=0
        #backward pass MAC
        cycles +=self.GetBackwardMAC()
        #backward halos
        for ig in self.inputGradients:
            cycles += math.ceil(ig.H/self.core.pCol)*(self.R-1) + math.ceil(ig.W/self.core.pRow) * (self.S-1) * ig.C
        #HW has expensive systolic all gather of weight updates, to/from all PEs to accumulate, then again to broadcast
        cycles+=2*(self.GetWeightCount()*self.core.pCol) + 2*(self.GetWeightCount()*self.core.pRow)
        #cycles to update local weight values
        cycles +=self.GetWeightCount()
        #NOTE transition from CK to HW done in seperate operator
        return cycles
    def GetHSystolicBW(self):
        bw=0
        #incoming input value to buffer
        bw+=1
        #outgoing result
        bw+=1
        #incoming halo
        bw+=1
        #outgoing halo
        bw+=1
        #two for weight accumulations during weight update
        bw+=2
        return bw
    def GetVSystolicBW(self):
        bw=0
        #incoming halo
        bw+=1
        #outgoing halo
        bw+=1
        #two for weight accumulations during weight updates
        bw+=2
        return bw
    def GetHBroadcastBW(self):
        #HW never does all to one broadcast. Always systolic so every PE can broadcast
        bw=0
        return bw
    def GetVBroadcastBW(self):
        bw=0
        return bw

class CKConv(Operator):
    def __init__(self,ID,R,S,C,K):
        super(CKConv,self).__init__(ID)
        self.R=R
        self.S=S
        self.C=C
        self.K=K
    def PrintInfo(self):        
        print "---CKConv Operator ID is :" + str(self.ID) + ' and associated core is:' +str(self.core.ID)
        print '----- R: '+str(self.R) + ' S:'+str(self.S) + ' K:' + str(self.K)
    def GetInputActCount(self):
        count=0
        for ia in self.inputActivations:
            count += (ia.H) * (ia.W) * math.ceil(ia.C/self.core.pCol)
        return count
    def GetOutputActCount(self):
        count=0
        for oa in self.outputActivations:
            count += oa.H * oa.W * math.ceil(oa.C/self.core.pRow)
        return count
    def GetInputGradCount(self):
        count=0
        for ig in self.inputGradients:
            count += ig.H * ig.W * math.ceil(ig.C/self.core.pRow)
        return count
    def GetOutputGradCount(self):
        count=0
        for og in self.outputGradients:
            count += og.H * og.W * math.ceil(og.C/self.core.pCol)
        return count
    def GetWeightCount(self):
        return self.R * self.S * math.ceil(self.K/self.core.pRow) * math.ceil(self.C/self.core.pCol)
    
    def GetInputActAccesses(self):
        acc=0
        #we read the input activations all once per weight (weight stationary) in the forward pass to calculate output
        acc+= self.GetInputActCount() * self.R * self.S * math.ceil(self.K/self.core.pRow)
        #we read the input activations once per output gradient to calculate weight updates (output gradient stationary)
        if len(self.outputGradients)!=0:
            acc += self.GetInputActCount() * self.outputGradients[0].H * self.outputGradients[0].W * math.ceil(self.K/self.core.pRow)
        return acc
    
    def GetOutputActAccesses(self):
        acc=0
        #output activations are written to with forward pass results and accumulated for each operation on input activations
        acc+=self.GetInputActCount() * self.R * self.S * math.ceil(self.K/self.core.pRow)
        return acc
    
    def GetInputGradAccesses(self):
        acc=0
        #input gradients are read in the backward pass to produce output gradients
        acc+= self.GetInputGradCount() * self.R * self.S * math.ceil(self.C/self.core.pCol)
        return acc
    
    def GetOutputGradAccesses(self):
        acc=0
        #output gradients are written to in the backward pass to produce partial gradient for each K
        acc+=self.GetInputGradCount() * (self.R * self.S * math.ceil(self.C/self.core.pCol))
        #output gradients are read from to calculate weight update values (gradient stationary so once each)
        acc+=self.GetOutputGradCount()
        #output gradients are the accumulated in the K dimension
        acc+=self.GetOutputGradCount() * math.ceil(self.K/self.core.pRow)
        return acc
    
    def GetWeightAccesses(self):
        acc=0
        #read weights once each in forward pass (weight stationary)
        acc+=self.GetWeightCount()
        #read weights once each in backward pass (weight stationary)
        acc+=self.GetWeightCount()
        #RMW to weights for updates
        acc+=2*self.GetWeightCount()
        return acc
    def GetForwardMAC(self):
        mac=0
        #every local weight is multiplied by input activations
        mac += self.GetInputActCount() * self.R * self.S * math.ceil(self.K/self.core.pRow)
        return mac
    def GetBackwardMAC(self):
        mac=0
        #calculate per K output gradients
        mac += self.GetInputGradCount() * (self.R * self.S * math.ceil(self.C/self.core.pCol))
        #calculate weight update values
        mac+=self.GetInputActCount() * self.GetOutputGradCount()/math.ceil(self.C/self.core.pCol)
        return mac
    
    def GetForwardCycles(self):
        cycles=0
        #cycles for forward PE
        cycles += self.GetForwardMAC()
        #cycles for accum in K dimension
        cycles += self.core.pRow * self.GetOutputActCount()
        #TODO SKIP CONNECTIONS any incoming activations from a different core we need to rotate incoming matrix to match
        return cycles
    def GetBackwardCycles(self):
        cycles=0
        #one to all column broadcast to split out in C
        cycles += self.GetInputGradCount()
        #backward pass MAC
        cycles +=self.GetBackwardMAC()
        #cycles to update local weight values
        cycles +=self.GetWeightCount()
        #cycles for accum in K dimension
        cycles += self.core.pRow * self.GetOutputActCount()
        return cycles
    def GetHSystolicBW(self):
        bw=0
        #accumulation in backward pass
        bw+=1
        #systolically shifted in inputs must be supported if HW->CK
        bw+=1
        return bw
    def GetVSystolicBW(self):
        bw=0
        #accumulate in forward pass
        bw+=1
        return bw    
    def GetHBroadcastBW(self):
        bw=0
        #output broadcast
        bw+=1
        #input broadcast
        bw+=1
        return bw
    def GetVBroadcastBW(self):
        #transpose required in backward pass, one to all
        bw=1
        return bw
    
class FC(Operator):
    def __init__(self,ID,M):
        super(FC,self).__init__(ID)
        self.M=M
    def PrintInfo(self):
        print "---FC Operator ID is :" + str(self.ID) + ' and associated core is:' +str(self.core.ID)
        print '----- M: '+str(self.M)
        
    def GetInputActCount(self):
        count=0
        for ia in self.inputActivations:
            count += ia.H * ia.W * math.ceil(ia.C/self.core.pRow)
        return count
    def GetOutputActCount(self):
        count=0
        for oa in self.outputActivations:
            count += oa.H * oa.W * math.ceil(oa.C/self.core.pRow)
        return count
    def GetInputGradCount(self):
        count=0
        for ig in self.inputGradients:
            count += ig.H * ig.W * math.ceil(ig.C/self.core.pRow)
        return count
    def GetOutputGradCount(self):
        count=0
        for og in self.outputGradients:
            count += og.H * og.W * math.ceil(og.C/self.core.pRow)
        return count
    def GetWeightCount(self):
        count=0
        #TODO assumes dimension of first input is correct
        #and will be combined with other inputs of same dimension before MAC
        if len(self.inputActivations) != 0:
            count+=math.ceil(self.M/self.core.pCol) * ((self.inputActivations[0].H*self.inputActivations[0].W*math.ceil(self.inputActivations[0].C)/self.core.pRow))
        return count
    
    def GetInputActAccesses(self):
        reads=0
        #we read the input activations all once in the forward. They get Broadcasted
        reads+= self.GetInputActCount()
        #we read the input activations once to calculate weight updates in the reverse direction (broadcast again)
        reads+= self.GetInputActCount()
        
    def GetOutputActAccesses(self):
        acc=0
        #output activations are written to with forward pass results
        acc+=self.GetOutputActCount()
        return acc
    
    def GetInputGradAccesses(self):
        acc=0
        #input gradients are read in the backward pass to produce output gradients
        acc+= self.GetInputGradCount()
        return acc
    
    def GetOutputGradAccesses(self):
        acc=0
        #output gradients are written to in the backward pass
        acc+=self.GetOutputGradCount()
        #output gradients are read from to calculate weight update values
        acc+=self.GetOutputGradCount()
        return acc
    
    def GetWeightAccesses(self):
        acc=0
        #read weights once each in forward pass (weight stationary)
        acc+=self.GetWeightCount()
        #read weights once each in backward pass (weight stationary)
        acc+=self.GetWeightCount()
        #RMW to weights for updates
        acc+=2*self.GetWeightCount()
        return acc
    
    def GetForwardMAC(self):
        mac=0
        #every local weight is multiplied by input activations
        mac += self.GetInputActCount() * math.ceil(self.M/self.core.pCol)
        return mac
    
    def GetBackwardMAC(self):
        mac=0
        #calculate per K output gradients
        mac += self.GetInputGradCount() * math.ceil(self.M/self.core.pCol)
        #calculate weight update values (one value per weight)
        mac+=self.GetWeightCount()
        return mac
    
    def GetForwardCycles(self):
        cycles=0
        #cycles for forward PE
        cycles += self.GetForwardMAC()
        #TODO SKIP CONNECTIONS any incoming activations from a different core we need to rotate incoming matrix to match
        return cycles
    def GetBackwardCycles(self):
        cycles=0
        #backward pass MAC
        cycles +=self.GetBackwardMAC()
        #accumulate in the M dimension on diagonals
        cycles += math.ceil(self.M/self.core.pRow)
        #cycles to update local weight values
        cycles +=self.GetWeightCount()
        return cycles        
    def GetHSystolicBW(self):
        bw=0
        #accumulate in backward pass
        bw+=1
        return bw
    def GetVSystolicBW(self):
        bw=0
        #accumulation in forward pass
        bw+=1
        return bw
    def GetHBroadcastBW(self):
        bw=0
        #input broadcast
        bw+=1
        #output broadcast (and internal for weight update)
        bw+=1
        return bw
    def GetVBroadcastBW(self):
        bw=0
        #gradient broadcast in backward pass
        bw+=1
        return bw
    
#TODO this would need to be seperate for HW and CK ??
class Transpose(Operator):
    def __init__(self,ID,start,end):
        super(Transpose,self).__init__(ID)
        self.startType=start
        self.endType=end
    #Block to convert from HW to CK blocking in PEs and back
    #note, this doesn't require any memory, it uses the memory of the previous convolution.
    #just here for clarity and to provide cycle / bandwidth / access stats
    def PrintInfo(self):
        print 'Transpose with ID: ' + str(self.ID)
    #to the outside world, these counts don't matter. They are all stored elsewhere
    #so we lists memory / counts as zero
    def GetInputActCount(self):
        count=0
        return count
            
    def GetOutputActCount(self):
        count=0
        return count
    
    def GetOutputGradCount(self):
        count=0
        return count

    def GetInputGradCount(self):
        count=0
        return count
    
    def GetWeightCount(self):
        #see above, weights are stored in convolution operators
        #transpose is just additional cycles, BW and accesses
        return 0

    #For HW->CK and for CK->HW
    #the steps required that are in additional to those normally performed in the conv operators
    #is to do a columnwise broadcast to redistribute
    
    #accesses for power and cycles
    def GetInputActAccesses(self):
        acc=0
        #need to read all of the 'inputs' to broadcast them in the forward direction
        for ia in self.inputGradients:
            if self.startType=='HW':
                acc += math.ceil(ia.H/self.core.pRow) * math.ceil(ia.W/self.core.pCol) * ia.C            
            elif self.startType=='CK':
                acc += ia.H * ia.W * math.ceil(ia.C/self.core.pRow)                            
            else:
                print "HIT ELSE STATEMENT WE SHOULD NOT HAVE HIT. UNSUPPORTED TRANSPOSE"
        return acc

    def GetOutputActAccesses(self):
        acc=0
        #need to write transpose outputs in forward direction
        for oa in self.inputGradients:
            if self.endType=='HW':
                acc += math.ceil(oa.H/self.core.pRow) * math.ceil(oa.W/self.core.pCol) * oa.C            
            elif self.endType=='CK':
                acc += oa.H * oa.W * math.ceil(oa.C/self.core.pRow)                            
            else:
                print "HIT ELSE STATEMENT WE SHOULD NOT HAVE HIT. UNSUPPORTED TRANSPOSE"
                
        return acc
    def GetInputGradAccesses(self):
        acc=0
        #need to read to broadcast in bakcward direction
        for ig in self.inputGradients:
            if self.endType=='HW':
                acc += math.ceil(ig.H/self.core.pRow) * math.ceil(ig.W/self.core.pCol) * ig.C            
            elif self.endType=='CK':
                acc += ig.H * ig.W * math.ceil(ig.C/self.core.pRow)                            
            else:
                print "HIT ELSE STATEMENT WE SHOULD NOT HAVE HIT. UNSUPPORTED TRANSPOSE"
        return acc
    def GetOutputGradAccesses(self):
        acc=0
        #need to write tranpose outputs in back direction
        for og in self.outputGradients:
            if self.startType=='HW':
                acc += math.ceil(og.H/self.core.pRow) * math.ceil(og.W/self.core.pCol) * og.C            
            elif self.startType=='CK':
                acc += og.H * og.W * math.ceil(og.C/self.core.pRow)                            
            else:
                print "HIT ELSE STATEMENT WE SHOULD NOT HAVE HIT. UNSUPPORTED TRANSPOSE"                
        return acc
    
    def GetWeightAccesses(self):
        #None, all in conv
        return 0
    #MAC for power and cycles
    def GetForwardMAC(self):
        return 0
    def GetBackwardMAC(self):
        return 0
    #cycles
    def GetForwardCycles(self):
        cyc=0
        #all to all broadcast means every PE needs to transmit all local input activations, for every PE in a row systolically
        #HACK use accesses
        cyc+=self.core.pRow * self.GetInputActAccesses()
        return cyc
    def GetBackwardCycles(self):
        cyc=0
        #all to all broadcast means every PE needs to transmit all local input activations, for every PE in a row systolically
        #HACK use accesses
        cyc+=self.core.pRow * self.GetInputGradAccesses()
        return cyc
    def GetHSystolicBW(self):
        bw=0
        return bw
    def GetVSystolicBW(self):
        #vertical systolic broadcast
        bw=1
        return bw
    def GetHBroadcastBW(self):
        bw=0
        return bw
    def GetVBroadcastBW(self):
        bw=0
        return bw
    
class MuxDemux(Operator):
    def __init__(self,ID):
        super(MuxDemux,self).__init__(ID)
    #Forward direction this takes multiple inputs and reduces to one output
    #backward direction this takes one input and splits out into multiple outputs
    def PrintInfo(self):
        print 'MuxDemux with ID: ' + str(self.ID)
    #TODO we need this for skip connections
    
class Pool(Operator):
    def __init__(self,ID):
        super(Pool,self).__init__(ID)
    #TODO
class RELU(Operator):
    def __init__(self,ID):
        super(RELU,self).__init__(ID)
    #TODO
    
class Batch(Operator):
    def __init__(self,ID):
        super(Batch,self).__init__(ID)
    #TODO
    
class Core(object):
    def __init__(self,ID,pRow,pCol):
        self.ID=ID
        self.pRow=pRow
        self.pCol=pCol        
    def PrintInfo(self):
        print '---Core ID:' + str(self.ID) + ' pRow:' +str(self.pRow)+' pCol:'+str(self.pCol)

#Class to represent a residual block
#this block usually has two inputs
#this block contains multiple convolution layers and batch norm layers
class Res(Operator):
    def __init__(self,ID,convCount):
        super(Res,self).__init__(ID)
        print "ID is " + ID
        print "self id is " + self.ID
        self.conv=[]
        self.bn=[]
        self.convCount=convCount
        self.internalActivations=defaultdict(Activation)
        self.internalGradients=defaultdict(Gradient)
        #super has input activation list and output activation list
        #but we need to create more internally to connect things up
    def PrintInfo(self):
        print "---Residual BLock ID is :" + str(self.ID) + ' and associated core is:' +str(self.core.ID)
        print '----- Convs are :'
        print self.conv
        print '----- BNs are :'
        print self.bn
        print '---end of info---'
    def PrintInternalInfo(self):
        print "Internal Convolution Ops"
        for c in self.conv:
            c.PrintInfo()
        print "Internal BN Ops"
        for b in self.bn:
            b.PrintInfo()
        print '----end of internal info---'
    #need to be able to add convolution layers to this residual block
    def AddConv(self,h,w,newConv):
        #conv object is expected to have parameters set, but not activations or gradients        
        #if this is the first conv layer, then we will use the 'primary' inputs
        #these will be added later so just leave for now
        if len(self.conv)!=0:
        #otherwise the input activation is the output from the previous
            newConv.AddInputAct(self.conv[len(self.conv)-1].outputActivations[0])
        #opposite for gradients,output is input of previous layer
            newConv.AddOutputGrad(self.conv[len(self.conv)-1].inputGradients[0])
           
        #if it is the last conv layer, then use the 'primary' output activations and gradients
        #set later so just leave for now
        if len(self.conv)!=self.convCount-1:
            internalID='Conv'+self.ID+'_'+str(len(self.conv))
            #otherwise create a new output activation                        
            #H and W are inputs
            #C is already set in the conv itself
            c=newConv.C
            newAct=Activation(internalID,h,w,c)
            newConv.AddOutputAct(newAct)
            self.internalActivations[internalID]=newAct
            #for gradients, same thing but need to create input not output
            newGrad=Gradient(internalID,h,w,c)
            newConv.AddInputGrad(newGrad)
            self.internalGradients[internalID]=newGrad

        #add the convolution block to internal list
        self.conv.append(newConv)
        
        #add corresponding batch norm block to the internal list
        newBN = Batch('BN'+self.ID+'_'+str(len(self.conv)))
        #HACK for now just create new ones
        dummyAct=Activation('Act'+self.ID+'_'+str(len(self.conv)),h,w,newConv.K)
        dummyGrad=Activation('Grad'+self.ID+'_'+str(len(self.conv)),h,w,newConv.K)
        #we dont actually need the graph to be intact internally (HUGE HACK)
        newBN.AddInputAct(dummyAct)
        newBN.AddOutputAct(dummyAct)
        newBN.AddInputGrad(dummyGrad)
        newBN.AddOutputGrad(dummyGrad)
        newBN.SetCore(newConv.core)
        self.bn.append(newBN)
        
        
        
    def AddInputAct(self,act):
        #input activation is also input to first conv
        self.conv[0].AddInputAct(act)
        super(Res,self).AddInputAct(act)
    def AddOutputAct(self,act):
        #output activation is also the output of the last conv
        self.conv[len(self.conv)-1].AddOutputAct(act)
        super(Res,self).AddOutputAct(act)
    def AddInputGrad(self,act):
        #input gradient is also input to last conv
        self.conv[len(self.conv)-1].AddInputGrad(act)
        super(Res,self).AddInputGrad(act)
    def AddOutputGrad(self,act):
        #output activation is also output of first conv
        self.conv[0].AddOutputGrad(act)
        super(Res,self).AddOutputGrad(act)
        self.PrintInternalInfo()

        








        
class Network(object):
    def __init__(self):
        self.name = None
        self.precision=16 #default
        self.inActID=None
        self.inGradID=None
        #TODO for now just hold everything in the graph so we can print debug. might just need a origin node
        self.activations=defaultdict(Activation)
        self.gradients=defaultdict(Gradient)
        self.ops=defaultdict(Operator)
        self.opIDs=[] #for printing and plotting in order
        self.cores=defaultdict(Core)

    def ParseXMLNetwork(self,filepath):
        tree = ET.parse(filepath)
        network = tree.getroot()
        self.name = network.tag #name is tag of high level network object
        self.precision = float(network.find('precision').text)
        self.inActID = network.find('inActID').text
        self.inGradID = network.find('inGradID').text
        #first go through and read all the core information, adding them to the dict
        for c in network.findall('Core'):
            id=c.find('ID').text
            newCore = Core(id,int(c.find('pRow').text),int(c.find('pCol').text))
            self.cores[id]=newCore

        print self.cores
            
        #then go through and read in all the activations and the gradients. These are stored in dictionaries based on ID
        for a in network.findall('Activation'):
            id=a.findall('ID')[0].text
            h=int(a.findall('H')[0].text)
            w=int(a.findall('W')[0].text)
            c=int(a.findall('C')[0].text)
            newAct = Activation(id,h,w,c)
            self.activations[id]=newAct

            
        for g in network.findall('Gradient'):
            id=g.findall('ID')[0].text
            h=int(g.findall('H')[0].text)
            w=int(g.findall('W')[0].text)
            c=int(g.findall('C')[0].text)
            newGrad = Gradient(id,h,w,c)
            self.gradients[id]=newGrad

        #then read in all the operator objects. create the correct sub class based on type and then use the IDs to create the graph structure
        for o in network.findall('Operator'):
            opID = o.find('ID').text
            core = o.find('core').text
            type = o.find('type').text
            if type == 'HWConv':
                r=int(o.findall('R')[0].text)
                s=int(o.findall('S')[0].text)
                c=int(o.findall('C')[0].text)
                k=int(o.findall('K')[0].text)
                newOp = HWConv(opID,r,s,c,k)
            elif type == 'CKConv':
                r=int(o.findall('R')[0].text)
                s=int(o.findall('S')[0].text)
                c=int(o.findall('C')[0].text)
                k=int(o.findall('K')[0].text)
                newOp = CKConv(opID,r,s,c,k)
            elif type == 'FC':
                m=int(o.findall('M')[0].text)
                newOp = FC(opID,m)
            elif type == 'Transpose':
                start = o.findall('startType')[0].text
                end = o.findall('endType')[0].text
                newOp=Transpose(opID,start,end)
            elif (type == 'ResHW' or type=='ResCK'):
                convCount = int(o.findall('convCount')[0].text)
                inH=int(o.findall('inH')[0].text)
                inW=int(o.findall('inW')[0].text)
                outH=int(o.findall('outH')[0].text)
                outW=int(o.findall('outW')[0].text)
                #create residual block
                newOp=Res(opID,convCount)
                #loop through all internal convolutions
                #TODO check == convCount
                for idx,conv in enumerate(o.findall('conv')):
                    r= int(conv.findall('R')[0].text)
                    s= int(conv.findall('S')[0].text)
                    if idx==0:
                        c= int(conv.findall('C')[0].text)
                        w=inW
                        H=inH
                    else:
                        c=k #c is equal to the k value from the last loop if not first
                        w=outW
                        h=outH
                        
                    k= int(conv.findall('K')[0].text)
                    if type =='ResHW':
                        toadd = HWConv(idx,r,s,c,k)
                    else:
                        toadd = CKConv(idx,r,s,c,k)
                    toadd.SetCore(self.cores[core])
                    newOp.AddConv(h,w,toadd)
                newOp.PrintInternalInfo()
            else:
                newOp = None #SHOULD NEVER GET HERE, SANITY

            #set core
            newOp.SetCore(self.cores[core])
            #set all input/output activations/gradients            
            for iAct in o.findall('inputAct'):
                id=iAct.text
                newOp.AddInputAct(self.activations[id])
                self.activations[id].AddOutput(newOp)
            for oAct in o.findall('outputAct'):
                id=oAct.text
                newOp.AddOutputAct(self.activations[id])
                self.activations[id].AddInput(newOp)
            for iGrad in o.findall('inputGrad'):
                id=iGrad.text
                newOp.AddInputGrad(self.gradients[id])
                self.gradients[id].AddOutput(newOp)
            for oGrad in o.findall('outputGrad'):
                id=oGrad.text
                newOp.AddOutputGrad(self.gradients[id])
                self.gradients[id].AddInput(newOp)
            self.ops[opID]=newOp
            self.opIDs.append(opID)
        print self.opIDs
            
    #TODO this printing is very messy
    #and we should at the very least have one function
    #which we can do if activation and gradient inherit "vertice"

    def ForwardPrint(self,Act,visited):
        Act.PrintInfo()
        for op in Act.outputs:
            if visited[op.ID]!=1:
                op.PrintInfo()
        for op in Act.outputs:
            if visited[op.ID]!=1:
                for oact in op.outputActivations:
                    self.ForwardPrint(oact,visited)
                visited[op.ID]=1

    def BackwardPrint(self,Grad,visited):        
        Grad.PrintInfo()
        for op in Grad.outputs:
            if visited[op.ID]!=1:
                op.PrintInfo()
            for op in Grad.outputs:
                if visited[op.ID]!=1:
                    for oact in op.outputGradients:
                        self.BackwardPrint(oact,visited)
                visited[op.ID]=1                

    
    
    def PrintNetwork(self):
        #Print all the Cores
        print "Cores in Architecture are:"
        for c in self.cores.values():
            c.PrintInfo()
        #Then start with the first activation and do a BFS on it
        print '--------PRINTING FORWARD PASS --------------------'
        #mark all of the operator nodes as "not visited" to start
        visitedOps = defaultdict()
        for o in self.ops.values():
            visitedOps[o.ID]=0
        self.ForwardPrint(self.activations[self.inActID],visitedOps)
        print '--------PRINTING BACKWARD PASS --------------------'
        visitedOps = defaultdict()
        for o in self.ops.values():
            visitedOps[o.ID]=0
        self.BackwardPrint(self.gradients[self.inGradID],visitedOps)
    #methods to Get the statistics from each of the operators in the network
    #returns an array with an element for each operation core
    #counts for memory and cycles
    def GetPEInputActMem(self,index=None):
        mem=[]
        if index==None:            
            for oid in self.opIDs:
                mem.append(self.ops[oid].GetInputActCount() * self.precision * MBCONVERSION)
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            mem.append(self.ops[self.opIDs[index]].GetInputActCount()*self.precision * MBCONVERSION)
        return mem
    
    def GetPEOutputActMem(self,index=None):
        mem=[]
        if index==None:            
            for oid in self.opIDs:
                mem.append(self.ops[oid].GetOutputActCount() * self.precision * MBCONVERSION)
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            mem.append(self.ops[self.opIDs[index]].GetOutputActCount()*self.precision * MBCONVERSION)
        return mem

    def GetPEOutputGradMem(self,index=None):
        mem=[]
        if index==None:            
            for oid in self.opIDs:
                mem.append(self.ops[oid].GetOutputGradCount() * self.precision * MBCONVERSION)
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            mem.append(self.ops[self.opIDs[index]].GetOutputGradCount()*self.precision * MBCONVERSION)
        return mem
    
    def GetPEInputGradMem(self,index=None):
        mem=[]
        if index==None:            
            for oid in self.opIDs:
                mem.append(self.ops[oid].GetInputGradCount() * self.precision * MBCONVERSION)
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            mem.append(self.ops[self.opIDs[index]].GetInputGradCount()*self.precision * MBCONVERSION)
        return mem
    
    def GetPEWeightMem(self, index=None):
        mem=[]
        if index==None:            
            for oid in self.opIDs:
                mem.append(self.ops[oid].GetWeightCount() * self.precision * MBCONVERSION)
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            mem.append(self.ops[self.opIDs[index]].GetWeightCount()*self.precision * MBCONVERSION)
        return mem

    def GetPEMem(self,index=None):
        mem=[]
        if index==None:            
            for oid in self.opIDs:
                mem.append((self.ops[oid].GetWeightCount()+self.ops[oid].GetInputActCount()+self.ops[oid].GetInputGradCount()) * self.precision * MBCONVERSION)
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            mem.append((self.ops[self.opIDs[index]].GetWeightCount()+self.ops[self.opIDs[index]].GetInputActCount()+self.ops[self.opIDs[index]].GetInputGradCount())*self.precision * MBCONVERSION)
        return mem
    
    #accesses for power and cycles
    def GetPEInputActAccesses(self,index=None):
        acc=[]
        if index==None:            
            for oid in self.opIDs:
                acc.append(self.ops[oid].GetInputActAccesses())
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            acc.append(self.ops[oid].GetInputActAccesses())
        return acc
    
    def GetPEOutputActAccesses(self,index=None):
        acc=[]
        if index==None:            
            for oid in self.opIDs:
                acc.append(self.ops[oid].GetOutputActAccesses())
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            acc.append(self.ops[oid].GetOutputActAccesses())
        return acc

    def GetPEInputGradAccesses(self,index=None):
        acc=[]
        if index==None:            
            for oid in self.opIDs:
                acc.append(self.ops[oid].GetInputGradAccesses())
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            acc.append(self.ops[oid].GetInputGradAccesses())
        return acc
    
    def GetPEOutputGradAccesses(self,index=None):
        acc=[]
        if index==None:            
            for oid in self.opIDs:
                acc.append(self.ops[oid].GetOutputGradAccesses())
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            acc.append(self.ops[oid].GetOutputGradAccesses())
        return acc
    
    def GetPEWeightAccesses(self,index=None):
        acc=[]
        if index==None:            
            for oid in self.opIDs:
                acc.append(self.ops[oid].GetWeightAccesses())
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            acc.append(self.ops[oid].GetWeightAccesses())
        return acc
    
    #MAC for power and cycles
    def GetPEForwardMAC(self,index=None):
        mac=[]
        if index==None:            
            for oid in self.opIDs:
                mac.append(self.ops[oid].GetForwardMAC())
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            mac.append(self.ops[oid].GetForwardMAC())
        return mac
    
    def GetPEBackwardMAC(self,index=None):
        mac=[]
        if index==None:            
            for oid in self.opIDs:
                mac.append(self.ops[oid].GetBackwardMAC())
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            mac.append(self.ops[oid].GetBackwardMAC())
        return mac
    
    #cycles
    def GetPEForwardCycles(self,index=None):
        cyc=[]
        if index==None:            
            for oid in self.opIDs:
                cyc.append(self.ops[oid].GetForwardCycles())
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            cyc.append(self.ops[oid].GetForwardCycles())
        return cyc

    def GetPEBackwardCycles(self,index=None):
        cyc=[]
        if index==None:            
            for oid in self.opIDs:
                cyc.append(self.ops[oid].GetBackwardCycles())
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            cyc.append(self.ops[oid].GetBackwardCycles())
        return cyc
    def GetPEHSystolicBW(self,index=None):
        bw=[]
        if index==None:            
            for oid in self.opIDs:
                bw.append(self.ops[oid].GetHSystolicBW()*self.precision* MBCONVERSION)
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            bw.append(self.ops[oid].GetHSystolicBW()*self.precision* MBCONVERSION)
        return bw
    def GetPEVSystolicBW(self,index=None):
        bw=[]
        if index==None:            
            for oid in self.opIDs:
                bw.append(self.ops[oid].GetVSystolicBW()*self.precision* MBCONVERSION)
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            bw.append(self.ops[oid].GetVSystolicBW()*self.precision* MBCONVERSION)
        return bw
    def GetPEHBroadcastBW(self,index=None):
        bw=[]
        if index==None:            
            for oid in self.opIDs:
                bw.append(self.ops[oid].GetHBroadcastBW()*self.precision* MBCONVERSION)
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            bw.append(self.ops[oid].GetHBroadcastBW()*self.precision* MBCONVERSION)
        return bw
    def GetPEVBroadcastBW(self,index=None):
        bw=[]
        if index==None:            
            for oid in self.opIDs:
                bw.append(self.ops[oid].GetVBroadcastBW()*self.precision* MBCONVERSION)
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            bw.append(self.ops[oid].GetVBroadcastBW()*self.precision* MBCONVERSION)
        return bw        

    #function to plot two lists, x being a list of operator IDs
    #x axis is assumed to be the operator IDs
    def PlotOps(self,title,ylabel,y):
        plt.xlabel('OP ID')
        plt.ylabel(ylabel)
        plt.title(title)
        x=[]
        for i in range(0,len(self.opIDs)):
            x.append(i)
        labels=self.opIDs
        plt.bar(x,y,align='center')
        plt.xticks(x,labels)
        plt.show()
        return
    

        

