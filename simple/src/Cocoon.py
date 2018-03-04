import xml.etree.ElementTree as ET
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from collections import defaultdict
import math
#Global value for conversions from bits to MB
MBCONVERSION = .000000125
PROW=1
PCOL=1
#easy enum for weight stationary or activation stationay
WS=1
AS=2
#easy enum for HW or CK
HW=3
CK=4

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
    def GetActAccesses(self):
        pass
    def GetGradAccesses(self):
        pass
    def GetWeightAccesses(self):
        pass
    def GetMAC(self):
        pass
    def GetCycles(self):
        pass
    #BW TODO
    def GetHSystolicBW(self):
        pass
    def GetVSystolicBW(self):
        pass    
    def GetHBroadcastBW(self):
        pass
    def GetVBroadcastBW(self):
        pass
    
class HWConv(Operator):
    def __init__(self,id,h,w,r,s,c,k,stationary):
        super(HWConv,self).__init__(ID=id,H=h,W=w,R=r,S=s,C=c,K=k)
        self.stationary=stationary
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

    def GetActAccesses(self):
        fwd=0
        bck=0
        wgt=0
        #--------------------
        #forward
        if self.stationary==WS:
            #weight stationary we read every input activation once per every weight
            #and for each of these we write to the output activation
            fwd+=2*(self.GetWeightCount()*math.ceil(self.H/PCOL) * math.ceil(self.W/PROW))
        else:
            #if we are activaiton stationary then we read every activation once
            #and write each output once for every calculation
            fwd+=self.GetInputActCount() + (self.GetWeightCount()*math.ceil(self.H/PCOL) * math.ceil(self.W/PROW))
        #--------------------
        #backward
        #back pass is between gradients and weights
                
        #--------------------
        #weight update
        if self.stationary==WS:
            #not activation stationary means for every gradient we multiply it by (RxS) input activation values
            wgt+=self.GetInputGradCount()*(self.R*self.S)
        else:
            #if activation is stationary then they are all read once
            wgt+=self.GetInputActCount()
            
        return (fwd,bck,wgt)
    
    def GetGradAccesses(self):
        fwd=0
        bck=0
        wgt=0
        #--------------------
        #forward
        #only activations and weights
        #--------------------
        #backward
        if self.stationary==WS:
            #if weight stationary, every gradient is multiplied by 1 full weight kernel
            #and each time we write to an output
            bck+=2*(math.ceil(self.GetWeightCount()/self.K)*self.GetInputGradientCount())
        else:
            #if not weight stationary, then we read each input once
            bck+=self.GetInputGradientCount()
        #--------------------
        #weight update
        if self.stationary==WS:
            #if not activation stationary, then each gradient is read once
            self.GetInputGradientCount()
        else:
            #if activation stationary, each gradient read (RxS) times per activation
            wgt+=self.R*self.S*self.GetInputActivationCount()
        return (fwd,bck,wgt)
    
    def GetWeightAccesses(self):
        fwd=0
        bck=0
        wgt=0
        #--------------------
        #forward
        if self.stationary==WS:
            #if weight stationary, each read once
            fwd+=self.GetWeightCount()
        else:
            #if activation stationary each is read once per input activation calc
            fwd+=(self.GetWeightCount()*math.ceil(self.H/PCOL) * math.ceil(self.W/PROW))
        #--------------------
        #backward
        if self.stationary==WS:
            #if weight stationary, each read once
            bck+=self.GetWeightCount()
        else:
            #if activation stationary a kernel is read once per input gradient
            bck+=math.ceil(self.GetWeightCount()/self.K)*self.GetInputGradCount()
        #--------------------
        #weight update
        #either way, we write to the weights to update them
        wgt+=self.GetWeightCount()
        
        return (fwd,bck,wgt)
    
    def GetMAC(self):
        fwd=0
        bck=0
        wgt=0

        
        return (fwd,bck,wgt)
    
    def GetCycles(self):
        fwd=0
        bck=0
        wgt=0
        #--------------------
        #forward
        #MAC cycles
        fwd+=(self.GetWeightCount()*math.ceil(self.H/PCOL) * math.ceil(self.W/PROW))
        #halo sharing cycles send and recieve overlap (after C reduction)        
        fwd += math.ceil(self.H/PCOL)*(self.R-1) + math.ceil(self.W/PROW) * (self.S-1) * self.K
        #--------------------
        #backward
        #MAC cycles
        bck+=self.GetInputGradCount() * math.ceil(self.GetWeightCount()/self.K)
        #halo sharing, need for every weight update so C dimension as well
        bck += math.ceil(self.H/PCOL)*(self.R-1) + math.ceil(self.W/PROW) * (self.S-1) * self.K * self.C
        #accumulation in K dimension of per weight gradients
        #mean K additions for every output gradient value
        bck += self.K * self.GetOutputGradCount()
        
        #--------------------
        #weight update
        #MAC Cycles
        wgt+=self.GetInputGradCount() * (self.R * self.S)
        #all gather and all broadcast of weight update values
        wgt+=2*(self.GetWeightCount()*PCOL) + 2*(self.GetWeightCount()*PROW)
        #actually updating weights requires another MAC
        wgt+=self.GetWeightCount()
        
        return (fwd,bck,wgt)

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
    def __init__(self,id,h,w,r,s,c,k,stationary):
        super(CKConv,self).__init__(ID=id,H=h,W=w,R=r,S=s,C=c,K=k)
        self.stationary=stationary
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
    
    def GetActAccesses(self):
        fwd=0
        bck=0
        wgt=0
        #-----------------------
        #forward
        if self.stationary==WS:
            #weight stationary we read every input activation once per every weight
            #and for each of these we write to the output activation
            fwd+=2*(self.GetWeightCount()*self.H * self.W)
        else:
            #if we are activaiton stationary then we read every activation once
            #and write each output once for every calculation
            fwd+=self.GetInputActCount() + (self.GetWeightCount()*self.H *self.W)
        #-----------------------
        #backward
        #gradients and weights only
        #-----------------------
        #weight update
        if self.stationary==WS:
            #not activation stationary means for every gradient we multiply it by (RxS) input activation values
            wgt+=self.GetInputGradCount()*(self.R*self.S)
        else:
            #if activation is stationary then they are all read once
            wgt+=self.GetInputActCount()
            
        return(fwd,bck,wgt)
    def GetGradAccesses(self):
        fwd=0
        bck=0
        wgt=0
        #-----------------------
        #forward
        #activations and weights only
        #-----------------------
        #backward
        if self.stationary==WS:
            #if weight stationary, every gradient is multiplied by 1weight kernel
            #and each time we write to an output
            bck+=2*(math.ceil(self.GetWeightCount())/(math.ceil(self.K/PCOL))*self.GetInputGradCount())
        else:
            #if not weight stationary, then we read each input once
            bck+=self.GetInputGradCount()
        #-----------------------
        #weight update
        if self.stationary==WS:
            #if not activation stationary, then each gradient is read once
            self.GetInputGradCount()
        else:
            #if activation stationary, each gradient read (RxS) times per activation
            wgt+=self.R*self.S*self.GetInputActCount()
            
        return(fwd,bck,wgt)
    
    def GetWeightAccesses(self):
        fwd=0
        bck=0
        wgt=0
        #-----------------------
        #forward
        #forward
        if self.stationary==WS:
            #if weight stationary, each read once
            fwd+=self.GetWeightCount()
        else:
            #if activation stationary each is read once per input activation calc
            fwd+=(self.GetWeightCount()* self.H* self.W)
            
        #-----------------------
        #backward
        if self.stationary==WS:
            #if weight stationary, each read once
            bck+=self.GetWeightCount()
        else:
            #if activation stationary a kernel is read once per input gradient
            bck+=math.ceil(self.GetWeightCount())/(math.ceil(self.K/PCOL))*self.GetInputGradCount()
        #-----------------------
        #weight update
        wgt+=self.GetWeightCount()
        
        return(fwd,bck,wgt)
    
    def GetMAC(self):
        fwd=0
        bck=0
        wgt=0

        #--------------------
        #forward
        #forward pass is every weight with HxW activations
        fwd+=(self.GetWeightCount()* self.H * self.W)
        #--------------------
        #backward
        #backward pass is every gradient with one weight kernel
        bck+=self.GetInputGradCount() * math.ceil(self.GetWeightCount())/(math.ceil(self.K/PCOL))
        #--------------------
        #weight update
        #each input is multiplied with RxS gradient vals
        wgt+=self.GetInputGradCount() * (self.R * self.S)        

        return(fwd,bck,wgt)        
    def GetCycles(self):
        fwd=0
        bck=0
        wgt=0
        #-----------------------
        #forward
        #MAC cycles
        fwd+=(self.GetWeightCount()*self.H * self.W)
        #Accumulate in C dimension along diagonals
        #to transmit this takes #rows * output size
        #assume that additions happen pipelined with reception
        fwd+=PROW*self.GetOutputActCount()
        #-----------------------
        #backward
        #one to all column broadcast cycles
        bck+=self.GetInputGradCount()
        #MAC
        bck+=self.GetInputGradCount() * math.ceil(self.GetWeightCount())/(math.ceil(self.K/PCOL))
        #local accumulation in K dimension of per weight gradients
        bck += math.ceil(self.K/PCOL)*self.GetOutputGradCount()
        #further accumulation in K dimension of per weight gradients done systolically
        bck += PCOL * self.GetOutputGradCount()
        #-----------------------
        #weight update
        #MAC cycles
        wgt+=self.GetInputGradCount() * (self.R * self.S)
        #actually update weights
        wgt+=self.GetWeightCount()
        
        return(fwd,bck,wgt)        

    #TODO BW
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
    def GetInputActCount(self):
        count=0
        if self.blockType==HW:
            count += math.ceil(self.H/PROW) * math.ceil(self.W/PCOL) * self.C * self.BS
        else: #if CK, then block in W and C
            count += self.H * math.ceil(self.W/PCOL) * math.ceil(self.C/PROW) * self.BS
        return count
            
    def GetOutputActCount(self):
        count=0
        if self.blockType==HW:
            count += math.ceil(self.H/PROW) * math.ceil(self.W/PCOL) * self.C * self.BS
        else:
            count += self.H * math.ceil(self.W/PCOL) * math.ceil(self.C/PROW) * self.BS
        return count
    
    def GetOutputGradCount(self):
        count=0
        if self.blockType==HW:
            count += math.ceil(self.H/PROW) * math.ceil(self.W/PCOL) * self.C * self.BS
        else:
            count += self.H * math.ceil(self.W/PCOL) * math.ceil(self.C/PROW) * self.BS
        return count

    def GetInputGradCount(self):
        count=0
        if self.blockType==HW:
            count += math.ceil(self.H/PROW) * math.ceil(self.W/PCOL) * self.C * self.BS
        else:
            count += self.H * math.ceil(self.W/PCOL) * math.ceil(self.C/PROW) * self.BS        
        return count

    def GetWeightCount(self):
        #Bach Norm Doesn't Have Weights
        #but we will consider the metadata to be 'weights'
        #we have to save a mean, variance, beta and gamma for each channel
        #per batch size
        return 4*math.ceil(self.C/PROW)
    
    #accesses for power and cycles
    def GetActAccesses(self):
        fwd=0
        bck=0
        wgt=0
        #-------------------------------------
        #forward pass
        #-read in and update mean, then store
        fwd+=self.GetInputActCount()
        #-read again on output and compute variance
        #-then do scale ops in place
        fwd+=self.GetInputActCount()
        #-write to output
        fwd+=self.GetOutputActCount()
        
        #-------------------------------------
        #back prop
        #this is stored from the forward pass
        #mu = 1./N*np.sum(h, axis = 0)
        #var = 1./N*np.sum((h-mu)**2, axis = 0)
        
        #this requires reading all gradients (no activations)
        #dbeta = np.sum(dy, axis=0)
        
        #this requires summations over entire back in H,W,C dimension, resulting in 1x1xC
        #so every input will be read once
        bck+=self.GetInputActCount()
        #dgamma = np.sum((h - mu) * (var + eps)**(-1. / 2.) * dy, axis=0)
        
        #another summation that requires both gradient and input activation
        #cant do on the same read because you need the value and the summation
        bck+=self.GetInputActCount()
        
        #dh = (1. / N) * gamma * (var + eps)**(-1. / 2.) * (N * dy - np.sum(dy, axis=0)
        #    - (h - mu) * (var + eps)**(-1.0) * np.sum(dy * (h - mu), axis=0))
        
        #-------------------------------------
        #weight update
        #no real weight updates for batch norm,but we do need to keep       
        #running averages of mean and variance. so:
        wgt+=2*self.GetInputActCount()
        
        #-------------------------------------
        return(fwd,bck,wgt)
    
    def GetGradAccesses(self):
        fwd=0
        bck=0
        wgt=0
        #-------------------------------------
        #forward pass
        #no gradients in forward pass
        
        #-------------------------------------
        #backward pass
        #these are stored from forward
        #mu = 1./N*np.sum(h, axis = 0)
        #var = 1./N*np.sum((h-mu)**2, axis = 0)
        
        #summation in C,H,W to produce 1x1xC beta
        #dbeta = np.sum(dy, axis=0)
        bck+=self.GetInputGradCount()
        
        #assume we use the same dy in the gamma equation here
        #dgamma = np.sum((h - mu) * (var + eps)**(-1. / 2.) * dy, axis=0)
        
        #will need dy again here, as we have to calculate dbeta first
        bck+=self.GetInputGradCount()       
        #dh = (1. / N) * gamma * (var + eps)**(-1. / 2.) * (N * dy - np.sum(dy, axis=0)
        #    - (h - mu) * (var + eps)**(-1.0) * np.sum(dy * (h - mu), axis=0))
        #have to write result
        bck+=self.GetOutputGradCount()
        #-------------------------------------
        
        
        #weight update
        #no real weight updates but we do need to update beta and gamma
        wgt+=2*self.GetInputGradCount()
        
        return (fwd,bck,wgt)
    
    def GetWeightAccesses(self):
        #update mean every computation in forward pass
        #read mean and variance every computation in backward pass
        fwd=0
        bck=0
        wgt=0
        #on the forward pass we update the mean and variance for every input value
        fwd+=2*self.GetInputActCount()
        
        #on the backward pass we read beta and gamma for every input grad
        bck+=2*self.C
        
        #which we will call the weight updates here
        wgt+=2*self.GetInputGradCount()

        return (fwd,bck,wgt)
    
    #MAC for power and cycles
    def GetMAC(self):
        fwd=0
        bck=0
        wgt=0
        #--------------------
        #forward pass we have one MAC per input value to scale it and produce output
        fwd+=self.GetInputActCount()
        #we also have a multiple for the mean for every channel
        fwd+=self.C

        #--------------------        
        #backward pass
        #three multiplication operations
        #dh = (1. / N) * gamma * (var + eps)**(-1. / 2.) *
        bck+=3*self.GetInputActCount()

        #then another multiply
        bck+=self.GetInputActCount()
        #(N * dy - np.sum(dy, axis=0)

        #three more multiplications, and then another in summation eq
        back+=self.GetInputActCount()
        back+=self.GetInputActCount()
        #- (h - mu) * (var + eps)**(-1.0) *
        #np.sum(dy * (h - mu), axis=0))

        
        #--------------------        
        #wgt updates
        #gamma requires three multiply operations per value
        #dgamma = np.sum((h - mu) * (var + eps)**(-1. / 2.) * dy, axis=0)
        wgt+=3*self.GetInputActCount()
        
        return (fwd,bck,wgt)

    
    def GetCycles(self):
        fwd=0
        bck=0
        wgt=0

        #---------------
        #forward pass
        #mean requires an addition for every incoming activation, then one multiply op     
        fwd+=self.GetInputActCount()+1
        #variance requires a subtraction and a square (mult) for each incoming activation within summation, then one multiply op
        #var = 1./N*np.sum((h-mu)**2, axis = 0)
        fwd+=(3*self.GetInputActCount())+1
        #HW needs to do all gather for values in HW dimension to get summations in C
        if self.blockType==HW:
            fwd+=self.GetInputGradCount() * ((PCOL) + (PROW))
        else:#CK only needs to do this in one dimension (the W dimension)
            fwd+=self.GetInputGradCount()*(PCOL)
                
        
        #---------------
        #backward pass
        #div,mult,mult,add,sqrt(12?),mult,mult,subtract,dbeta(counted in weight),subtract,substract,multiply,add,sqrt,add,mult,sub
        bck+=self.GetInputGradCount()*(2+1+1+1+12+1+1+1+1+1+1+1+12+1+1+1)
        #HW needs to do all gather for values in HW dimension to get summations in C
        if self.blockType==HW:
            bck+=self.GetInputGradCount() * ((PCOL) + (PROW))
        else:#CK only needs to do this in one dimension (the W dimension)
            bck+=self.GetInputGradCount()*(PCOL)
            
        #---------------
        #weight update
        #dbeta requires a summation
        wgt+=self.GetInputGradCount()
        #dgamma requires subtraction,multiplications,sqrt(12?) is reused, and another multiplication
        #all within a summation per input 
        wgt+=self.GetInputGradCount()*(1+1+1+1)

        return (fwd,bck,wgt)
    
    #BW TODO
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
        self.stationary = int(network.find('stationary').text)

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
                newOp = HWConv(opID,h,w,r,s,c,k,self.stationary)
                self.ops[opID]=newOp
                self.opIDs.append(opID)
            elif type == 'CKConv':
                h=int(o.findall('H')[0].text)
                w=int(o.findall('W')[0].text)
                r=int(o.findall('R')[0].text)
                s=int(o.findall('S')[0].text)
                c=int(o.findall('C')[0].text)
                k=int(o.findall('K')[0].text)
                newOp = CKConv(opID,h,w,r,s,c,k,self.stationary)
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
                            newConv=HWConv(COpID,h,w,r[idx],s[idx],c,k[idx],self.stationary)
                            newBatch = Batch(BOpID,h,w,c,batchSize,HW)
                        else: #CK only other type
                            newConv=CKConv(COpID,h,w,r[idx],s[idx],c,k[idx],self.stationary)
                            newBatch = Batch(BOpID,h,w,c,batchSize,CK)
                      
                        #create a convolution operator and add it
                        self.ops[COpID]=newConv
                        self.opIDs.append(COpID)
                        #create a batch norm operator and add it
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
            mem.append(self.ops[self.opIDs[index]].GetInputActCount()*self.precision * MBCONVERSION)
        return mem
    
    def GetPEOutputActMem(self,index=None):
        mem=[]
        if index==None:            
            for oid in self.opIDs:
                mem.append(self.ops[oid].GetOutputActCount() * self.precision * MBCONVERSION)
        else:
            mem.append(self.ops[self.opIDs[index]].GetOutputActCount()*self.precision * MBCONVERSION)
        return mem

    def GetPEOutputGradMem(self,index=None):
        mem=[]
        if index==None:            
            for oid in self.opIDs:
                mem.append(self.ops[oid].GetOutputGradCount() * self.precision * MBCONVERSION)
        else:
            mem.append(self.ops[self.opIDs[index]].GetOutputGradCount()*self.precision * MBCONVERSION)
        return mem
    
    def GetPEInputGradMem(self,index=None):
        mem=[]
        if index==None:            
            for oid in self.opIDs:
                mem.append(self.ops[oid].GetInputGradCount() * self.precision * MBCONVERSION)
        else:
            mem.append(self.ops[self.opIDs[index]].GetInputGradCount()*self.precision * MBCONVERSION)
        return mem
    
    def GetPEWeightMem(self, index=None):
        mem=[]
        if index==None:            
            for oid in self.opIDs:
                mem.append(self.ops[oid].GetWeightCount() * self.precision * MBCONVERSION)
        else:
            mem.append(self.ops[self.opIDs[index]].GetWeightCount()*self.precision * MBCONVERSION)
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
    
    #TODO these could all call the same unpacking function and just
    #provide a different function pointer to reduce code bloat
    
    def GetPEActAccesses(self,index=None):
        fwd=[]
        bck=[]
        wgt=[]
        if index==None:            
            for oid in self.opIDs:
                (f,b,w)=self.ops[oid].GetActAccesses()
                fwd.append(f)
                bck.append(b)
                wgt.append(w)
        else:
            (f,b,w)=self.ops[self.opIDs[index]].GetActAccesses()
            fwd.append(f)
            bck.append(b)
            wgt.append(w)            
        return (fwd,bck,wgt)
    
    def GetPEGradAccesses(self,index=None):
        fwd=[]
        bck=[]
        wgt=[]
        if index==None:            
            for oid in self.opIDs:
                (f,b,w)=self.ops[oid].GetGradAccesses()
                fwd.append(f)
                bck.append(b)
                wgt.append(w) 
        else:
            (f,b,w)=self.ops[self.opIDs[index]].GetGradAccesses()
            fwd.append(f)
            bck.append(b)
            wgt.append(w) 

        return(fwd,bck,wgt)
    
    def GetPEWeightAccesses(self,index=None):
        fwd=[]
        bck=[]
        wgt=[]
        if index==None:            
            for oid in self.opIDs:
                (f,b,w)=self.ops[oid].GetWeightAccesses()
                fwd.append(f)
                bck.append(b)
                wgt.append(w)
        else:
            (f,b,w)=self.ops[self.opIDs[index]].GetWeightAccesses()
            fwd.append(f)
            bck.append(b)
            wgt.append(w)

        return (fwd,bck,wgt)
    
    #MAC for power and cycles
    def GetPEMAC(self,index=None):
        fwd=[]
        bck=[]
        wgt=[]
        if index==None:            
            for oid in self.opIDs:
                (f,b,w)=self.ops[oid].GetMAC()
                fwd.append(f)
                bck.append(b)
                wgt.append(w)
        else:
            (f,b,w)=self.ops[self.opIDs[index]].GetMAC()
            fwd.append(f)
            bck.append(b)
            wgt.append(w)

        return (fwd,bck,wgt)

    #cycles
    def GetPECycles(self,index=None):
        fwd=[]
        bck=[]
        wgt=[]
        
        if index==None:            
            for oid in self.opIDs:
                (f,b,w)=self.ops[oid].GetCycles()
                fwd.append(f)
                bck.append(b)
                wgt.append(w)
        else:
            (f,b,w)=self.ops[self.opIDs[index]].GetCycles()
            fwd.append(f)
            bck.append(b)
            wgt.append(w)            

        return (fwd,bck,wgt)


    def GetPEHSystolicBW(self,index=None):
        #TODO
        pass
        bw=[]
        if index==None:            
            for oid in self.opIDs:
                bw.append(self.ops[oid].GetHSystolicBW()*self.precision* MBCONVERSION)
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            bw.append(self.ops[oid].GetHSystolicBW()*self.precision* MBCONVERSION)
        return bw
    def GetPEVSystolicBW(self,index=None):
        #TODO
        pass
        bw=[]
        if index==None:            
            for oid in self.opIDs:
                bw.append(self.ops[oid].GetVSystolicBW()*self.precision* MBCONVERSION)
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            bw.append(self.ops[oid].GetVSystolicBW()*self.precision* MBCONVERSION)
        return bw
    def GetPEHBroadcastBW(self,index=None):
        #TODO
        pass
        bw=[]
        if index==None:            
            for oid in self.opIDs:
                bw.append(self.ops[oid].GetHBroadcastBW()*self.precision* MBCONVERSION)
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            bw.append(self.ops[oid].GetHBroadcastBW()*self.precision* MBCONVERSION)
        return bw
    def GetPEVBroadcastBW(self,index=None):
        #TODO
        pass
        bw=[]
        if index==None:            
            for oid in self.opIDs:
                bw.append(self.ops[oid].GetVBroadcastBW()*self.precision* MBCONVERSION)
        else:
            self.ops[self.opIDs[index]].PrintInfo()
            bw.append(self.ops[oid].GetVBroadcastBW()*self.precision* MBCONVERSION)
        return bw        

    #BN layers function very differently from conv layers
    #so we have these functions to allow for stand alone analysis of one or the other
    def GetBNIDs(self):
        index=[]
        BnIDs=[]
        for idx,oid in enumerate(self.opIDs):
            if 'Bn' in oid:
                BnIDs.append(oid)
                index.append(idx)

        return (index,BnIDs)

    def GetFiltIDs(self):
        index=[]
        FIDs=[]
        for idx,oid in enumerate(self.opIDs):
            if 'Bn' not in oid:
                FIDs.append(oid)
                index.append(idx)

        return (index,FIDs)
    
    #function to plot two lists, x being a list of operator IDs
    #x axis is assumed to be the operator IDs
    def PlotOps(self,title,labels,ylabel,y):
        #labels is probably self.opids, but you might want only a portion
        plt.xlabel('OP ID')
        plt.ylabel(ylabel)
        plt.title(title)
        x=[]
        for i in range(0,len(labels)):
            x.append(i)
        plt.bar(x,y,align='center')
        plt.xticks(x,labels,rotation=90)
        plt.show()
        return
    

        

