import numpy as np

def Convolve(act,weights):
    #input:
    #activation of shape C,H,W
    #weight of shape K,C,H,W
    #we assume stride and no padding here

    #check if case where C is just one, and this is 2D
    if act.ndim==2 and weights.ndim==2:
        H,W=act.shape
        HH,WW=weights.shape
        OH = int((H-HH)+1)
        OW = int((W-WW)+1)
        out = np.zeros((OH,OW))
        for j in range(OH):
            for k in range(OW):
                input = act[j:j+HH,k:k+WW]
                filter=weights
                result = np.sum(input*filter,axis=(0,1))
                out[j,k]=result
    #-------end of 2 dim ------------------
    else: #------ 3 dim ----------
        C,H,W=act.shape
        K,C,HH,WW=weights.shape

        #for now, assume no padding and stride of one
        #(H-HH+2*pad)/stride+1
        OH = int((H-HH)+1)
        OW = int((W-WW)+1)
        out = np.zeros((K,OH,OW))
        
        for i in range(K):
            for j in range(OH):
                for k in range(OW):
                    input = act[:,j:j+HH,k:k+WW]
                    filter = weights[i]
                    result = np.sum(input*filter,axis=(0,1,2))
                    out[i:i+1,j,k]=result
    #--------end of 3 dim
    return out

def BackConvolve(grad,act,weight):
    #inputs:
    #grad: input gradients
    #act: original input activations
    #weight: input weights
    #----------------
    #outputs:
    #Gradient with respect to input (act)
    #Gradient with respect to weights (weight)

    #need to pad the output so we can do a full deconvolution and match dimensions
    dact = np.zeros_like(act)
    dw = np.zeros_like(weight)    
    #Hack to make it work with gradient of only one 'channel


    if grad.ndim==2 and weight.ndim==2:
        #this is the distributed case where we are only doing a single 'slice' of the computation        
        OH,OW=grad.shape
        H,W=act.shape
        HH,WW=weight.shape
        for j in range(OH):
            for k in range(OW):
                input = act[j:j+HH,k:k+WW]
                grad_curr = grad[j:j+1,k:k+1]
                dw += input*grad_curr
                temp = weight*grad_curr
                dact[j:j+HH,k:k+WW]+=temp
    else:
        #this is the general purpose case where we loop in the K and C dimensions
        K,C,HH,WW = weight.shape
        K,OH,OW = grad.shape
        C,H,W=act.shape #NOTE: only one activation at a time (no N param)        
        for i in range(K):
            for j in range(OH):
                for k in range(OW):
                    input = act[:,j:j+HH,k:k+WW]
                    grad_curr = grad[i:i+1,j:j+1,k:k+1]
                    dw[i] += input * grad_curr #TODO original sums in N axis ???
                    dact[:,j:j+HH,k:k+WW] += weight[i]*grad_curr
            print("iteration {i}".format(i=i))
            print(weight[i][0])

    return dact,dw
                                
class PE:
    def __init__(self,x,y):
        self.weights = dict() #dictionary to store layer weights with a kernel lookup
        self.id=(x,y) #store location in PE grid
        self.inAct=dict() #variable for input activations
        self.outAct=dict() #variablef or output activaitons
        self.inGrad=dict()
        self.outDAct=dict()
        self.outDW=dict()
        self.inBuf=None
        self.outBuf=None
        self.weightBuf=None

    def SetInAct(self,layer,act):
        self.inAct[layer]=act

    def LoadInBuf(self,data):
        self.inBuf=data
    def FlushInBuf(self):
        self.inBuf=None
    def FlushOutBuf(self):
        self.outBuf=None
        
    def ForwardConv(self,kernel,channel):
        conv = Convolve(self.inBuf,self.weights[kernel][channel,:,:])
        if self.outBuf==None:
            H,W=conv.shape
            self.outBuf=np.zeros((H,W))
            
        self.outBuf=np.add(self.outBuf,conv)

    def BackwardConv(self,kernel,layer):
        #in backwards pass
        #assume buffers have been filled prior to calling

        #we compute the backprop gradient and the weight gradient using
        #the gradient in the input buffer, the weight in the weight buffer and
        #he locally stored activation
        #grad: loaded into input buffer by one to all broadcast
        #act: stored locally from forward pass
        #weight:loaded into weight buffer by per row scatter op
        dact,dw = BackConvolve(self.inBuf,self.inAct[layer],self.weightBuf)

        #output backprop gradient is stored in out dact dict (notice, accumulated over all K weights)
        if(layer in self.outDAct):
            self.outDAct[layer] += dact
        else:
            self.outDAct[layer]=dact

        if(self.id==(0,0)):
            print("PE 0,0")
#            print (self.outDAct[layer])
            print(self.inBuf)
            print(self.weightBuf)

        #weight buffer for current kernel hold weight gradient 'slice'
        #(notice, each dw is seperate for each K weight, not accum)
        self.outDW[(kernel,layer)]=dw
        
    def PrintInAct(self,layer):
        print("PE ID is {x},{y}".format(x=self.id[0],y=self.id[1]))
        print("Input Activations for layer {x}".format(x=layer))
        print (self.inAct[layer])

    def PrintInBuf(self):
        print("PE ID is {x},{y}".format(x=self.id[0],y=self.id[1]))
        print("Input Buffer")
        print (self.inBuf)        

    def PrintWeights(self,kernel):
        print("PE ID is {x},{y}".format(x=self.id[0],y=self.id[1]))
        print("Weights for kernel {k}".format(k=kernel))
        print (self.weights[kernel])

    def PrintOutAct(self,layer):
        print("PE ID is {x},{y}".format(x=self.id[0],y=self.id[1]))
        print("Output Activations for layer {x}".format(x=layer))
        print(self.outAct[layer])

    def PrintOutBuf(self):
        print("PE ID is {x},{y}".format(x=self.id[0],y=self.id[1]))
        print("Output Buffer")
        print (self.outBuf)            
        
class Core:
    def __init__(self,size=4):
        self.PEGrid=dict()
        self.size=size
        for i in range(0,size):
            for j in range (0,size):
                self.PEGrid[i,j]=PE(i,j)

    def PrintPEGrid(self):
        for i in range(0,self.size):
            print ("")
            for j in range(0,self.size):
                print (self.PEGrid[i,j].id,end="")
        print("")


    def RowWeightScatter(self,row,col,k=0,c_index=0,layer=0):
        #in the specified row, the PE in the specified col will split its weight out to the other PEs in that row
        #regardless of column, lowest index weight is assigned to the lowest index PE (leftmost =0 etc)
        #for use in backward pass to distribute one weight kernal among all PEs

        #index - argument to index into weights in a C loop if required
        for c in range(0,self.size):
            self.PEGrid[row,c].weightBuf=self.PEGrid[row,col].weights[k][(c_index*self.size)+c,:,:]

            
    def GradBroadcast(self,row,col,layer=0):
        #during the backwards pass the current gradient 'slice' in the K dimension is broadcast to all
        #PEs for local computation
        for i in range(0,self.size):
            for j in range(0,self.size):
                self.PEGrid[i,j].LoadInBuf(self.PEGrid[row,col].inGrad[0])

    def RowActBroadcast(self,row,col,layer):
        #in the specified row, take the activation information stored in the given column
        #and broadcast it to the input buffer of all PEs in that row (including broadcaster)
        for c in range (0,self.size):            
            self.PEGrid[row,c].LoadInBuf(self.PEGrid[row,col].inAct[layer])
            #TODO currently one channel only
            
    def ColReduce(self,row,col):
        #within the given column, gather and reduce all output buffers
        #storing in output buffer of specified row
        for i in range(0,self.size):
            if(i!=row):
                self.PEGrid[row,col].outBuf=np.add(self.PEGrid[row,col].outBuf,self.PEGrid[i,col].outBuf)

    def AllForwardConv(self,k,c):
        #all PEs in the grid perform forward convolution of input buffer with layer weights
        #must specify kernel index
        for i in range(0,self.size):
            for j in range(0,self.size):
                #todo for now this is just one channel per row
                self.PEGrid[i,j].ForwardConv(k,c)

    def AllBackwardConv(self,k,l):
        #all PEs in the grid perform the backward convolution computation
        for i in range(0,self.size):
            for j in range(0,self.size):
                self.PEGrid[i,j].BackwardConv(k,l)
                
    def Forward(self,layer):
        #this function assumes that the input activations are buffered in each PE
        #also assumes weights are buffered for given layer
        #input activations are broadcast to input buffers of PEs within rows        
        #then each PE performs local convolutions
        #output buffers are accumulated within columns and stored as layer output for each row in a K loop

        #TODO K loop here, start fresh for a new output row
        for k in range(0,4):
            #flush input and output buffers
            for i in range(0,4):
                for j in range(0,4):
                    self.PEGrid[i,j].FlushInBuf()
                    self.PEGrid[i,j].FlushOutBuf()

            #loop through each column
            for col in range(0,self.size):
                #broadcast current column and print buffers
                self.RowActBroadcast(0,col,0)
                self.RowActBroadcast(1,col,0)
                self.RowActBroadcast(2,col,0)
                self.RowActBroadcast(3,col,0)
                

                #Perform convolution on first filter for all PEs in first row
                #keep accumulating in the outbut buffer for all channels
                self.AllForwardConv(k,col)

            #once each column has broadcast, reduce in columns
            #save accumulated outpout
            
            for c in range(0,self.size):
                self.ColReduce(k,c)#TODO K loop is row
                self.PEGrid[k,c].outAct[0]=self.PEGrid[k,c].outBuf



    def Backward(self,layer,numKernels):
        #perform backward pass computations and data movement for given layer
        #assumes input activations from forward pass are laoded in memory(.inAct)
        #assumes weights are loaded in memory (.weights)
        #assumes input gradients are loaded in memory (.inGrad)

        #first, make sure that the output buffers have nothing in them
        #Need to loop through all K for the given layer
        for k in range(numKernels):
            colindex = k%self.size
            #first, every row must scatter the current weights
            for r in range(self.size):
                #sets up all the weight buffers of all PEs
                #row,col,index,layer
                self.RowWeightScatter(r,colindex,int(k/4),0)

            #broadcast the current gradient to all PEs
            #sets up the inBuff of all PEs
            self.GradBroadcast(int(k/4),colindex)

            #compute the weight gradient at each PE
            #AND compute and accumulate the backward propogation gradient
            #NOTE 
            self.AllBackwardConv(k,layer)

