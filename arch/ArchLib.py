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

    dact,dw = None,None
    #Hack to make it work with gradient of only one 'channel

    C,H,W=act.shape #NOTE: only one activation at a time (no N param)
    if grad.ndim==2 and weight.ndim==3:
        OH,OW=grad.shape
        C,HH,WW=weight.shape
    else:
    K,C,HH,WW = weight.shape
    K,OH,OW = grad.shape

    dact = np.zeros_like(act)
    dw = np.zeros_like(weight)

    for i in range(K):
        for j in range(OH):
            for k in range(OW):
                input = act[:,j:j+HH,k:k+WW]
                grad_curr = grad[i:i+1,j:j+1,k:k+1]
                dw[i] += input * grad_curr #TODO original sums in N axis ???
                dact[:,j:j+HH,k:k+WW] += weight[i]*grad_curr

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

    def RowBroadcast(self,row,col,layer):
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
                
    def ColAccum(self,row):
        #accumulate elementwise what is in the output buffer of each PE within each column
        #store accumulated result in the buffer of the specified row
        for col in range(0,self.size):
            for i in range(0,self.size):
                if(i != row):
                    print ("-------------------")
                    print ("row is "+ str(row) + " and col is " + str(col))                            
                    print (self.PEGrid[row,col].outBuf)
                    print (self.PEGrid[i,col].outBuf)                    
                    self.PEGrid[row,col].outBuf=np.add(self.PEGrid[row,col].outBuf,self.PEGrid[i,col].outBuf)
                    print (self.PEGrid[row,col].outBuf)
        
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
                self.RowBroadcast(0,col,0)
                self.RowBroadcast(1,col,0)
                self.RowBroadcast(2,col,0)
                self.RowBroadcast(3,col,0)
                

                #Perform convolution on first filter for all PEs in first row
                #keep accumulating in the outbut buffer for all channels
                self.AllForwardConv(k,col)

            #once each column has broadcast, reduce in columns
            #save accumulated outpout
            
            for c in range(0,self.size):
                self.ColReduce(k,c)#TODO K loop is row
                self.PEGrid[k,c].outAct[0]=self.PEGrid[k,c].outBuf
                        



