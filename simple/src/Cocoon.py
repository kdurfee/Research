import xml.etree.ElementTree as ET
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from collections import defaultdict
import math
#Global value for conversions from bits to MB
MBCONVERSION = .000000125
PROW=8
PCOL=8
  
#parent class for all operator blocks
class Operator(object):
    def __init__(self,ID,H=None,W=None,C=None,K=None,R=None,S=None,BS=None,M=None):
        self.ID=ID #for identification, printing etc. unique and in order that you want them printed / plotted
        #For simplicity, all blocks have all dimensions some just might be None (by default) and wont get used
        self.H=H
        self.W=W
        self.C=C
        self.K=K
        self.R=R
        self.S=S
        self.BS=BS
        self.M=M

    #which core is this associated with in the architecture.
    #this is done so that cores can have their costs per operation (layer) summed up at the end
    #if operators share inputs and are of the same core then they can share memeory (inception style parallel layers)
    #TODO how to deal with seperate cores sharing inputs (snake skip connection style... but what if not by 2)
    def PrintInfo(self):
        print 'operator ID is ' + str(self.ID)
        print 'H:'+ str(self.H) +' W:' + str(self.W) +' C:' +str(self.C) +' K:' +str(self.K) +' R:' +str(self.R) +' S:' +str(self.S) +' BS:' +str(self.BS) +' M:' +str(self.M) 
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
    def __init__(self,id,h,w,r,s,c,k):
        super(HWConv,self).__init__(ID=id,H=h,W=w,R=r,S=s,C=c,K=k)
    def GetInputActCount(self):
        count = math.ceil(self.H/PCOL) * math.ceil(self.W/PROW) * self.C
        return count
    def GetOutputActCount(self):
        count = math.ceil(self.H/PCOL) * math.ceil(self.W/PROW) * self.K
        return count
    def GetInputGradCount(self):
        count = math.ceil(self.H/PCOL) * math.ceil(self.W/PROW) * self.K
        return count
    def GetOutputGradCount(self):
        count = math.ceil(self.H/PCOL) * math.ceil(self.W/PROW) * self.C
        return count
    def GetWeightCount(self):
        return (self.R * self.S * self.K * self.C)
    
    def GetInputActAccesses(self):
        acc=0
        #we read the input activations all once per weight (weight stationary) in the forward pass to calculate output
        acc+= self.GetInputActCount() * self.R * self.S * self.K
        #we read the input activations once per output gradient to calculate weight updates (output gradient stationary)
        acc += self.GetInputActCount() * math.ceil(self.H/PCOL) * math.ceil(self.W/PROW) * self.K
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
        acc += math.ceil(self.H/PCOL) * math.ceil(self.W/PROW) * self.K
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
        cycles += math.ceil(self.H/PCOL)*(self.R-1) + math.ceil(self.W/PROW) * (self.S-1) * self.K
        #NOTE transition from HW to CK done in seperate operator
        return cycles
    def GetBackwardCycles(self):
        cycles=0
        #backward pass MAC
        cycles +=self.GetBackwardMAC()
        #backward halos
        cycles += math.ceil(self.H/PCOL)*(self.R-1) + math.ceil(self.W/PROW) * (self.S-1) * self.C
        #HW has expensive systolic all gather of weight updates, to/from all PEs to accumulate, then again to broadcast
        cycles+=2*(self.GetWeightCount()*PCOL) + 2*(self.GetWeightCount()*PROW)
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
    def __init__(self,id,h,w,r,s,c,k):
        super(CKConv,self).__init__(ID=id,H=h,W=w,R=r,S=s,C=c,K=k)
    def GetInputActCount(self):
        count=0
        count += (self.H) * (self.W) * math.ceil(self.C/PCOL)
        return count
    def GetOutputActCount(self):
        count=0
        count += self.H * self.W * math.ceil(self.K/PROW)
        return count
    def GetInputGradCount(self):
        count=0
        count += self.H * self.W * math.ceil(self.K/PROW)
        return count
    def GetOutputGradCount(self):
        count=0
        count += self.H * self.W * math.ceil(self.C/PCOL)
        return count
    def GetWeightCount(self):
        return self.R * self.S * math.ceil(self.K/PROW) * math.ceil(self.C/PCOL)
    
    def GetInputActAccesses(self):
        acc=0
        #we read the input activations all once per weight (weight stationary) in the forward pass to calculate output
        acc+= self.GetInputActCount() * self.R * self.S * math.ceil(self.K/PROW)
        #we read the input activations once per output gradient to calculate weight updates (output gradient stationary)
        acc += self.GetInputActCount() * self.H * self.W * math.ceil(self.C/PROW)
        return acc
    
    def GetOutputActAccesses(self):
        acc=0
        #output activations are written to with forward pass results and accumulated for each operation on input activations
        acc+=self.GetInputActCount() * self.R * self.S * math.ceil(self.K/PROW)
        return acc
    
    def GetInputGradAccesses(self):
        acc=0
        #input gradients are read in the backward pass to produce output gradients
        acc+= self.GetInputGradCount() * self.R * self.S * math.ceil(self.C/PCOL)
        return acc
    
    def GetOutputGradAccesses(self):
        acc=0
        #output gradients are written to in the backward pass to produce partial gradient for each K
        acc+=self.GetInputGradCount() * (self.R * self.S * math.ceil(self.C/PCOL))
        #output gradients are read from to calculate weight update values (gradient stationary so once each)
        acc+=self.GetOutputGradCount()
        #output gradients are the accumulated in the K dimension
        acc+=self.GetOutputGradCount() * math.ceil(self.K/PROW)
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
        mac += self.GetInputActCount() * self.R * self.S * math.ceil(self.K/PROW)
        return mac
    def GetBackwardMAC(self):
        mac=0
        #calculate per K output gradients
        mac += self.GetInputGradCount() * (self.R * self.S * math.ceil(self.C/PCOL))
        #calculate weight update values
        mac+=self.GetInputActCount() * self.GetOutputGradCount()/math.ceil(self.C/PCOL)
        return mac
    
    def GetForwardCycles(self):
        cycles=0
        #cycles for forward PE
        cycles += self.GetForwardMAC()
        #cycles for accum in K dimension
        cycles += PROW * self.GetOutputActCount()
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
        cycles += PROW * self.GetOutputActCount()
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
    def __init__(self,id,c,m):
        super(FC,self).__init__(ID=id,M=m,C=c)
    def GetInputActCount(self):
        count=0
        count += math.ceil(self.C/PROW)
        return count
    def GetOutputActCount(self):
        count=0
        count += math.ceil(self.C/PROW)
        return count
    def GetInputGradCount(self):
        count=0
        count += math.ceil(self.C/PROW)
        return count
    def GetOutputGradCount(self):
        count=0
        count += math.ceil(self.C/PROW)
        return count
    def GetWeightCount(self):
        count=0
        count+=math.ceil(self.M/PCOL) * ((self.math.ceil(self.C)/PROW))
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
        mac += self.GetInputActCount() * math.ceil(self.M/PCOL)
        return mac
    
    def GetBackwardMAC(self):
        mac=0
        #calculate per K output gradients
        mac += self.GetInputGradCount() * math.ceil(self.M/PCOL)
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
        cycles += math.ceil(self.M/PROW)
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
   
class Batch(Operator):
    def __init__(self,id,h,w,c,bs,blockType):
        super(Batch,self).__init__(ID=id,H=h,W=w,C=c,BS=bs)
        self.blockType=blockType
    def GetWeightCount(self):
        #Bach Norm Doesn't Have Weights
        #but we will consider the metadata to be 'weights'
        #we have to save a mean, variance, beta and gamma for each channel
        #per batch size
        return 4*self.C
    def GetInputActCount(self):
        count=0
        if self.blockType=='HW':
            count += match.ceil(self.H/PROW) * math.ceil(self.W/PCOL) * self.C * self.BS
        else:
            count += self.H * self.W * math.ceil(self.C/PROW) * self.BS
        return count
            
    def GetOutputActCount(self):
        count=0
        if self.blockType=='HW':
            count += match.ceil(self.H/PROW) * math.ceil(self.W/PCOL) * self.K * self.BS
        else:
            count += self.H * self.W * math.ceil(self.K/PROW) * self.BS
        return count
    
    def GetOutputGradCount(self):
        count=0
        if self.blockType=='HW':
            count += match.ceil(self.H/PROW) * math.ceil(self.W/PCOL) * self.C * self.BS
        else:
            count += self.H * self.W * math.ceil(self.C/PROW) * self.BS
        return count

    def GetInputGradCount(self):
        count=0
        if self.blockType=='HW':
            count += match.ceil(self.H/PROW) * math.ceil(self.W/PCOL) * self.K * self.BS
        else:
            count += self.H * self.W * math.ceil(self.K/PROW) * self.BS        
        return count    
     #accesses for power and cycles
    def GetInputActAccesses(self):
        acc=0
        #forward pass each in read once to get the mean
        acc+=self.GetInputActCount()
        #then read again on output to compute variance
        #and to do scale ops on (all at once)
        acc+=self.GetInputActCount()
        #These then get read again in the backward pass for gradient calc
        return acc
    def GetOutputActAccesses(self):
        #output activations are written to once in the forward pass when computed
        acc+=self.GetOutputActCount()
        pass
    def GetInputGradAccesses(self):
        #these are read in the backward pass during gradient calculations
        pass
    def GetOutputGradAccesses(self):
        #these are written to once calculated
        acc+=self.GetOutputGradCount()
        return acc
    def GetWeightAccesses(self):
        #update mean every computation in forward pass
        #read mean and variance every computation in backward pass
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
        
class Network(object):
    def __init__(self):
        self.name = None
        self.precision=16 #default
        self.ops=defaultdict(Operator)
        self.opIDs=[] #for printing and plotting in order

    def ParseXMLNetwork(self,filepath):
        tree = ET.parse(filepath)
        network = tree.getroot()
        self.name = network.tag #name is tag of high level network object
        self.precision = float(network.find('precision').text)

        #read in all the operator objects. create the correct sub class based on type
        #add the ID to the dict and also append to the array in incoming order
        for o in network.findall('Operator'):
            opID = o.find('ID').text
            type = o.find('type').text
            if type == 'HWConv':
                h=int(o.findall('H')[0].text)
                w=int(o.findall('W')[0].text)
                c=int(o.findall('C')[0].text)
                r=int(o.findall('R')[0].text)
                s=int(o.findall('S')[0].text)
                k=int(o.findall('K')[0].text)
                newOp = HWConv(opID,h,w,r,s,c,k)
                self.ops[opID]=newOp
                self.opIDs.append(opID)
            elif type == 'CKConv':
                h=int(o.findall('H')[0].text)
                w=int(o.findall('W')[0].text)
                r=int(o.findall('R')[0].text)
                s=int(o.findall('S')[0].text)
                c=int(o.findall('C')[0].text)
                k=int(o.findall('K')[0].text)
                newOp = CKConv(opID,h,w,r,s,c,k)
                self.ops[opID]=newOp
                self.opIDs.append(opID)
            elif type == 'FC':
                m=int(o.findall('M')[0].text)
                c=int(o.findall('C')[0].text)
                newOp = FC(opID,m,c)
                self.ops[opID]=newOp
                self.opIDs.append(opID)
            elif (type == 'ResHW' or type=='ResCK'):
                batchSize = int(o.findall('batchSize')[0].text)
                multiple =int(o.findall('multiple')[0].text)
                inH = int(o.findall('inH')[0].text)
                inW = int(o.findall('inW')[0].text)
                inC = int(o.findall('inC')[0].text)
                outH = int(o.findall('outH')[0].text)
                outW = int(o.findall('outW')[0].text)
                #create lists of the different convolution args
                r=[]
                s=[]
                k=[]
                convCount=0
                for idx,val in enumerate(o.findall('R')):
                    r.append(int(val.text))
                    convCount+=1
                for idx,val in enumerate(o.findall('S')):
                    s.append(int(val.text))
                for idx,val in enumerate(o.findall('K')):
                    k.append(int(val.text))

                #loop through all multiples            
                for x in range(0,multiple):
                    for idx in range(0,convCount):
                        if x==0 and idx==0:
                            h=inH
                            w=inW
                            c=inC
                        else: #all others have the output dimensions
                            h=outH
                            w=outW
                            #C is the K dimension of the previous convolution
                            #still set from previous loop
                            c=newConv.K
                        COpID=opID+'_block_'+str(x)+'_Cnv_'+str(idx)
                        BOpID=opID+'_block_'+str(x)+'_Bn_'+str(idx)
                        if type == 'ResHW':
                            newConv=HWConv(COpID,h,w,c,r[idx],s[idx],k[idx])
                        else: #CK only other type
                            newConv=CKConv(COpID,h,w,c,r[idx],s[idx],k[idx])
                      
                        #create a convolution operator and add it
                        self.ops[COpID]=newConv
                        self.opIDs.append(COpID)
                        #create a batch norm operator and add it
                        newBatch = Batch(BOpID,h,w,c,batchSize)
                        self.ops[BOpID]=newBatch
                        self.opIDs.append(BOpID)
            else:
                print "SHOULD NEVER HIT THIS, UNKNOWN TYPE WHEN PARSING XML"            
            
        print self.opIDs

    def PrintNetwork(self):
        for idx in self.opIDs:
            self.ops[idx].PrintInfo()
    
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
            mem.append(self.ops[oid].GetWeightCount()*self.precision * MBCONVERSION)
        return mem

    def GetPEMem(self,index=None):
        mem=[]
        if index==None:            
            for oid in self.opIDs:
                mem.append((self.ops[oid].GetWeightCount()+self.ops[oid].GetInputActCount()+self.ops[oid].GetInputGradCount()) * self.precision * MBCONVERSION)
        else:
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
        plt.xticks(x,labels,rotation=90)
        plt.show()
        return
    

        

