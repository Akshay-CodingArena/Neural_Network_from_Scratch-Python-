import pickle
import random
import math
import numpy as np

def pow(v):
    x=2
    j=1
    
    if v>0:
        while j<v:
            x=2*x
            j+=1
    else:  
        while j>v:
            x=x/2
            j-=1
    return x    


def feedfor(net,i):
    f=0
    net.sum1ac[f]=net.inpu[net.z].T
    while(f<=i):
#        print("hidden layer",net.wih[f],"input",net.sum1ac[f])
        net.sum1[f]=np.dot(net.sum1ac[f].T,net.wih[f]).T+net.biases[f]  
        net.sum1ac[f+1]=act(net.sum1[f])
        f+=1
#    print("\nsumlac",net.sum1ac[f])    
    return(net.sum1ac[f])

def act(c):
    m=c*1
    for i in range(m.shape[0]):
        for j in range(m.shape[1]): 
            try:
                m[i,j]=float(int(m[i,j]*100)/100)
                m[i,j]=float(int((1/(1+math.exp(-m[i,j])))*1000)/1000)
            except:
                print("Out of range ",m[i,j])
  #          if m[i,j]>.5:
   #             m[i,j]=1
    #        else:
     #           m[i,j]=0               
    return m  

def dact(x):
    return np.multiply(x,1-x)

def ddact(x):
    y=dact(x)
    return act(y)

class Matrix:
    a=[]
    x=None
    y=None
    def __init__(self,x,y):
        self.a=[]
        self.x=x
        self.y=y
        for i in range(self.x):
            self.a.append([0.1 for i in range(self.y)])
        self.a=np.matrix(self.a)
    
    def randomize(self):
        self.a=np.matrix(self.a)
        for i in range(self.x):
            for j in range(self.y):
                self.a[i,j]=random.uniform(0.3,0.1) 
        print("pass",self.a)        
        return self.a        

class NeuralNet:
    inp=None
    inpu=None
    hidden=None
    out=None
    actout1=None
    who=None
    wih={}
    num=None
    sum1={}
    sum1ac={}
    biases={}
    z=None
    
    def __init__(self,inp,hidden,out,num):
        self.inp=inp
        self.num=num
        self.hidden=hidden
        self.out=out
        i=0
        if i==0:
            self.biases[i]=Matrix(hidden,1).randomize()
            self.wih[i]=Matrix(inp,hidden).randomize()
            i+=1
                
        while i<self.num:
            self.biases[i]=Matrix(hidden,1).randomize()
            self.wih[i]=Matrix(hidden,hidden)
            self.wih[i]=self.wih[i].randomize()
            print("Hidden ",self.wih[i])
            i+=1
        self.biases[i]=Matrix(out,1).randomize()
        self.who=Matrix(hidden,out).randomize()
   #     self.who=Matrix(out,hidden).randomize()
        print("Input",self.inp)
        print("Output",self.who)
        
    def feedforward(self,inp):
        inpu=np.matrix(inp)
        print("iNPUT ARRAY",inpu)
        print("HIDDEN LAYER WEIGHTS",self.wih)
        print("OUTPUT LAYER WEIGHTS",self.who)
        who=self.who
        if self.who.shape[0]==1:
            who=self.who.T
        print("num ",self.num)
        i=0
        j=0
        self.sum1ac[i]=np.matrix(inpu.T)
 #       print("Test krl    wih",self.wih," activate ",self.sum1ac[i])
        
        while(i<self.num):
            self.sum1[i]=np.dot(self.sum1ac[i].T,self.wih[i]).T +self.biases[i]
 #           print("loop multiply",self.wih[i],self.sum1[i])
            self.sum1ac[i+1]=act(self.sum1[i])
 #           print("result",self.sum1ac[i+1])
            i+=1
        out=np.dot(self.sum1ac[i].T,who).T+self.biases[i]
        print("check output",out,"Biases",self.biases)
        outputsig=act(out)
        return outputsig

def update(net,error,lr,z,sum2):
    num=net.num
    k=0
    while k<=num:
        if k==0: 
            dw=np.multiply(dact(feedfor(net,k)),error[num-k])
            dw=np.dot(dw,np.matrix(net.inpu[z]))*lr
            net.wih[k]=net.wih[k]+dw.T
            db=np.multiply(error[num-k],dact(feedfor(net,k)))*lr
            net.biases[k]=net.biases[k]+db
            k=k+1
            continue
        if k<num:
            feed=feedfor(net,k-1)
            dw=np.multiply(dact(feedfor(net,k)),error[num-k])
            dw=np.dot(dw,feed.T)*lr
            db=np.multiply(error[num-k],ddact(feedfor(net,k)))*lr
            net.biases[k]=net.biases[k]+db
            net.wih[k]=net.wih[k]+dw.T
            k+=1
            continue
        if k==num:
            dw=np.multiply(dact(sum2),error[0])
            dw=(np.dot(dw,feedfor(net,net.num-1).T))*lr
            db=np.multiply(error[0],dact(sum2))*lr
            net.biases[num]=net.biases[num]+db
            net.who=net.who+dw.T
            break

def setup():
    y1=int(input("Enter no. of hidden layers"));
    x=int(input("Enter the no. of input nodes "))
    y=int(input("Enter the no. of hidden nodes "))
    z=int(input("Enter the no. of output nodes "))
    network= NeuralNet(x,y,z,y1)
    return network  

def train(net,inp,out):
    net.inpu=np.matrix(inp)
#    print("iNPUT ARRAY",net.inpu)
#    print("HIDDEN LAYER WEIGHTS",net.wih)
#    print("OUTPUT LAYER WEIGHTS",net.who)
#    print("iNPUT NUM",net.num)
    error={}
    db={}
    num=net.num
    lr=0.1
    j=0
    
    while j<100000:
        j+=1
        
        z=random.randint(0,len(net.inpu)-1)
        net.z=z
   
        i=0
      
        net.sum1ac[i]=np.matrix(net.inpu[z].T)


        while(i<=net.num-1):
     #           print("\n valueee of i ",i)
                net.sum1[i]=(np.dot(net.sum1ac[i].T,net.wih[i]).T)+net.biases[i]
    #            print("weight",net.wih[i],"dot( activation )",net.sum1ac[i])
                net.sum1ac[i+1]=act(net.sum1[i])
      #          print("Sum",net.sum1ac[i+1])
                i+=1

        sum2=(np.dot(net.sum1ac[i].T,net.who).T)+net.biases[i]
        outputsig=act(sum2)

        error[0]=out[z].T-outputsig
   #     print("error out",error[0])
        
        de=np.dot(net.who,error[0])
        
        k=net.num-1
        error[num-k]=de
        
        if k>0:        
            while k>=1:
                de=np.dot(net.wih[k],error[num-k])
                k-=1
                error[num-k]=de

        update(net,error,lr,z,outputsig)
#    print("After change hidden weight",net.wih,"\n output weight",net.who,"\n biases",net.biases)

def test(network,inp):
    output=network.feedforward(inp)
    return output

def main():
    network=setup()
    while True:
        dec=int(input("1.Train 2.Predict 3.setup 4.Stop"))
        if dec==1:
           
            gat=int(input("1.OR Gate 2.AND Gate 3.XOR Gate 4.NAND GAte 5.XNOR Gate"))
            if gat==1:
                inp=[[0,0],
                    [1,0],
                    [0,1],
                    [1,1]]
                out=[[0],
                     [1],
                     [1],
                     [1]]
                inp=np.matrix(inp)
                out=np.matrix(out)
                train(network,inp,out)
            elif gat==2:
                
                inp=[[0,0],
                    [1,0],
                    [0,1],
                    [1,1]
                    ]
                out=[[0],
                     [0],
                     [0],
                     [1]
                    ]
                inp=np.matrix(inp)
                out=np.matrix(out)
                train(network,inp,out)
            elif gat==3:
                inp=[[0,0],
                    [1,0],
                    [0,1],
                    [1,1]
                    ]
                    
                out=[[0],
                     [1],
                     [1],
                     [0]
                    ]
                inp=np.matrix(inp)
                out=np.matrix(out)
                train(network,inp,out)
            
            elif gat==4:
                inp=[[0,0],
                    [1,0],
                    [0,1],
                    [1,1]
                    ]
                    
                out=[[1],
                     [1],
                     [1],
                     [0]
                    ]
                inp=np.matrix(inp)
                out=np.matrix(out)
                train(network,inp,out)
                
            elif gat==5:
                
                inp=[[0,0],
                    [1,0],
                    [0,1],
                    [1,1]
                    ]
                    
                out=[[1,0],
                     [0,1],
                     [0,1],
                     [1,0]
                    ]
                inp=np.matrix(inp)
                out=np.matrix(out)
                train(network,inp,out)    
                
            elif gat==6:
                dbfile = open('MyNeauralNetTrain', 'rb') 
                data = pickle.load(dbfile) 
                dbfile.close()
                net=train(network,data[0],data[1])
    
            
        elif dec==2:
            inp=input("Enter the input ").split()
            for i in range(len(inp)):
                inp[i]=float(inp[i])
            res=test(network,inp)
            print(res)
        elif dec==3:
            setup();
        elif dec==4:
            break;

main()        
                
