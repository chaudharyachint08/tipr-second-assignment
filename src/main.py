import sys, os, shutil, itertools as it
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt

from skimage import io
from sklearn.metrics import f1_score, accuracy_score, classification_report


import warnings
warnings.filterwarnings("ignore")

try:
    res_stdout
except:
    res_stdout = (sys.stdout if sys.stdout else sys.__stdout__)

verbose = 0

sys.stdout = sys.__stdout__ = open('stdoutbuffer','a',buffering=1)
mpl.use('Agg')

class ActV:
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    def relu(x):
        return np.maximum(0,x)
    def tanh(x):
        return 2*ActV.sigmoid(x)-1
    def swish(x):
        return x*ActV.sigmoid(x)
    def softmax(x):
        x = x-x.max(axis=1,keepdims=True)
        _ = np.exp(x)
        return _/np.sum(_,axis=1,keepdims=True)

class ActD:
    def sigmoid(x):
        _ = ActV.sigmoid( x )
        return _ * (1-_)
    def relu(x):
        '1 for x>=0'
        return (np.sign(x)>=0)
    def tanh(x):
        return 1-(ActV.tanh(x))**2
    def swish(x):
        'y’ = y + σ(x) . (1 – y)'
        _1 = ActV.swish(x)
        _2 = ActV.sigmoid(x)
        return _1 + _2*(1-_1)
    def softmax(x):# Still in doubt, it should be a matrix
        _ = ActV.softmax( x )
        return _ * (1-_)



class CostV:
    def cross_entropy(act, pred):
        pred = np.where(act!=1,pred+np.e,pred) # Handling perfect prediction
        pred = np.where(np.logical_and(act==1,pred==0),pred+10**-8,pred) # Handling imperfect prediction
        return -1*np.mean( act*np.log(pred) ,axis=0,keepdims=True)
    def MSE(act, pred):
        return np.mean( (pred-act)**2 ,axis=0,keepdims=True)
    
class CostD:
    def cross_entropy(act, pred):
        return pred-act
    def MSE(act, pred):
        return 2*(pred-act)

class Metrices:
    def accuracy(act, pred):
            return np.mean((act==pred).all(axis=1))

def one_hot(y):
    return 1*(y==y.max(axis=1,keepdims=True))

def cattooht(Y):
    Y = np.ravel(Y)
    _ = sorted(set(Y))
    tmp = np.zeros((Y.shape[0],len(_)),dtype='int32')
    for i in range(len(Y)):
        tmp[i][_.index(Y[i])] = 1
    return tmp,_



def initWB(IP,OP,function='relu',He=True,mode='gaussian'):
    if He:
        # Xavier & He initialization
        _ = 1/(IP+OP)**0.5
        if function in ('sigmoid','softmax'):
            r, s = 6**0.5, 2**0.5
        elif function=='tanh':
            r, s = 4*6**0.5, 4*2**0.5
        else: # relu or swish function
            r, s = 12**0.5, 2
        r, s = r*_, s*_
    else:
        r, s = 1, 1
    # Generating matrices
    if mode=='uniform':
        return 2*r*np.random.random((IP,OP))-r , 2*r*np.random.random((1,OP))-r
    elif mode=='gaussian':
        return np.random.randn(IP,OP)*s , np.random.randn(1,OP)*s
    else:
        raise Exception('Code should be unreachable')



def RSplit(X,Y,K=10):
    'Random Split Function'
    _ = list(range(X.shape[0]))
    index_set = []
    indxs = set(_)
    batch_size = round(X.shape[0]/K)
    np.random.shuffle(_)
    for k in range(0,X.shape[0],batch_size):
        test = set(_[k:k+batch_size])
        train = indxs - test
        index_set.append((list(train),list(test)))
    return index_set

def SSplit(X,Y,K=10,seed=False):
    'Stratified Split Function'
    if seed:
        np.random.seed(42)
    Y = pd.DataFrame([tuple(y) for y in Y])
    classes = set(Y)
    c2i = {}
    for index,label in Y.iterrows():
        label = label[0]
        if label in c2i:
            c2i[label].add(index)
        else:
            c2i[label] = {index}
    
    # Each class -> list of indices
    for i in c2i:
        c2i[i] = list(c2i[i])
        np.random.shuffle(c2i[i])
    
    # Each class with its set of train, test split indices
    c2is = {}
    for cls in c2i:
        a = int(np.round(len(c2i[cls])/K))
        c2is[cls] = []
        for fold in range(K):
            test_indices  = c2i[cls][a*fold:a*(fold+1)]
            train_indices = c2i[cls][0:a*fold] + c2i[cls][a*(fold+1):]
            c2is[cls].append((train_indices,test_indices))
        np.random.shuffle(c2is[cls])
        
    index_set = []
    for i in range(K):
        train,test = set(),set()
        for cls in c2is:
            _ = c2is[cls][i]
            train.update(set(_[0]))
            test.update (set(_[1]))
        index_set.append((list(train),list(test)))
    return index_set

def BSplit(X,Y,K=10):
    'Biased Split Function'
    indx = sorted(np.arange(X.shape[0]),key = lambda i:list(Y[i]))
    indices = set(indx)
    index_set = []
    step = int(np.ceil(len(indx)/K))
    for i in range(0,len(indx),step):
        test = set(indx[i:i+step])
        train = indices - test
        index_set.append((list(train),list(test)))
    return index_set

def Split(X,Y,K=10,mode='R'):
    if mode=='S':
        return SSplit(X,Y,K)
    elif mode=='B':
        return BSplit(X,Y,K)
    else:
        return RSplit(X,Y,K)



# Ref: https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy
def asStride(arr,sub_shape,stride):
    '''Get a strided sub-matrices view of an ndarray.
    See also skimage.util.shape.view_as_windows()
    '''
    s0,s1 = arr.strides[:2]
    m1,n1 = arr.shape[:2]
    m2,n2 = sub_shape
    view_shape = (1+(m1-m2)//stride[0],1+(n1-n2)//stride[1],m2,n2)+arr.shape[2:]
    strides = (stride[0]*s0,stride[1]*s1,s0,s1)+arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(arr,view_shape,strides=strides)
    return subs

def poolingOverlap(mat,ksize,stride=None,method='max',pad=False):
    '''Overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <stride>: tuple of 2 or None, stride of pooling window.
              If None, same as <ksize> (non-overlapping pooling).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           (n-f)//s+1, n being <mat> size, f being kernel size, s stride.
           if pad, output has size ceil(n/s).

    Return <result>: pooled matrix.
    '''
    m, n = mat.shape[:2]
    ky,kx = ksize
    if stride is None:
        stride = (ky,kx)
    sy,sx = stride

    _ceil = lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny = _ceil(m,sy)
        nx = _ceil(n,sx)
        size = ((ny-1)*sy+ky, (nx-1)*sx+kx) + mat.shape[2:]
        mat_pad = np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        mat_pad=mat[:(m-ky)//sy*sy+ky, :(n-kx)//sx*sx+kx, ...]

    view=asStride(mat_pad,ksize,stride)

    if method=='max':
        result=np.nanmax(view,axis=(2,3))
    else:
        result=np.nanmean(view,axis=(2,3))

    return result


# <h3>Global Dataset store & Dummy set generation</h3>
try:
    datasets
except:
    datasets = {}

argv = list(sys.argv)

if '--train-data' in argv:
    train_path = argv[argv.index('--train-data')+1]
    prm_fldr = 'temporary'
    config   = []
    for st in argv[argv.index('--configuration')+1:]:
        st  = st.strip()
        if st.endswith(']'):
            config.append(int(st.strip('[]')))
            break
        else:
            config.append(int(st.strip('[]')))
            
else:
    prm_fldr = 'models'

test_path     = argv[argv.index('--test-data')+1]
    
name = argv[argv.index('--dataset')+1]
    
def add_dataset(path):
    #path = '/home/clabuser/ACHINT/TIPR/assgn2/tipr-second-assignment-master/data'
    res_path = os.getcwd()
    os.chdir(path)
    fldr = name
    datasets[fldr] = ([],[])
    _ = sorted([x for x in os.listdir() if not x.startswith('.')])
    name_index = {x:_.index(x) for x in _}
    for category in _:
        label = [0]*len(_)
        label[name_index[category]] = 1
        os.chdir(category)
        for sample in os.listdir():
            if not fldr.startswith('.'):
                img_mat = io.imread(sample, as_gray=True)
                if fldr=='Cat-Dog': img_mat = poolingOverlap(img_mat,(4,4))
                img_mat = np.ravel(img_mat)
                datasets[fldr][0].append(img_mat)
                datasets[fldr][1].append(label)
        os.chdir('..')
    datasets[fldr] = tuple(map(np.array,datasets[fldr]))+(_,)
    os.chdir( res_path )
    for i in datasets:
        datasets[i] = np.array(datasets[i][0],dtype='float64'), datasets[i][1], datasets[i][2]




class NN:
    def __init__(self):
        self.Num, self.fun = [], []
        self.IP, self.OP, self.W, self.B, self.delta = {}, {}, {}, {}, {}
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 10**-8

    def data_feed( self, M, L, targets):
        self.raw, self.labels, self.target_names = M, L, targets

    def data_validate( self, M=np.array([]), L=np.array([]) ):
        self.vraw, self.vlabels = M, L

    def add(self,N,f='relu'):
        self.Num.append(N); self.fun.append(f)

    def data_preprocess(self,mode='standard'):
        sp = np.nan_to_num
        try:
            mode = self.preprocess_mode
        except:
            self.preprocess_mode = mode
        if mode=='scale':
            try:
                self.mn, self.mx
            except:
                self.mn, self.mx = self.raw.min(axis=0), self.raw.max(axis=0)
            mx = np.where(self.mx==self.mn,self.mx+1,self.mx)
            self.data  = sp((self.raw - self.mn)/(mx-self.mn))
            try: # If validation data is defined
                self.vdata = sp((self.vraw - self.mn)/(self.mx-self.mn))
            except:
                self.vdata = self.data
        elif mode=='standard':
            try:
                self.mean, self.std
            except:
                self.mean, self.std   = self.raw.mean(axis=0), self.raw.std(axis=0)
            std = np.where(self.std==0,1,self.std)
            self.data = sp((self.raw-self.mean)/std)
            try: # If validation data is defined
                self.vdata  =  sp((self.vraw-self.mean)/std)
            except:
                self.vdata = self.data
        else:
            raise Exception('Code should be unreachable')
    
    def initialize_layers(self,He=True,mode='gaussian'):
        for i in range(len(self.Num)):
            if i==0:
                self.W[i],self.B[i], = initWB(self.data.shape[1],self.Num[i],self.fun[i],He,mode)
            else:
                self.W[i],self.B[i], = initWB(self.Num[i-1],self.Num[i],self.fun[i],He,mode)
                
    def forward_prop(self,predict=False):
        self.IP[0] = self.fdata
        for i in range(len(self.Num)):
            wx_b = np.dot(self.IP[i],self.W[i])+self.B[i]
            if not predict:
                self.OP[i] = wx_b
            _ = eval('ActV.{0}(wx_b)'.format(self.fun[i]))
            self.IP[i+1] = _
            if predict:
                del self.IP[i]
        return self.IP[len(self.Num)]

    def back_prop3(self,Epoch_Count=1,debug=False):
        for i in range(len(self.Num)-1,-1,-1):
            if i==(len(self.Num)-1):
                costD = eval('CostD.{0}(self.flabels,self.IP[len(self.Num)])'.format(self.cost))
                actvD = eval('ActD.{0}(self.OP[i])'.format(self.fun[i]))
                self.delta[i] = costD * actvD
            else:
                costD = np.dot(self.W[i+1],self.delta[i+1].T).T
                actvD = eval('ActD.{0}(self.OP[i])'.format(self.fun[i]))
                self.delta[i] = costD * actvD
            uW = np.dot( self.IP[i].T , self.delta[i] ) / self.IP[i].shape[0]
            uB = np.mean( self.delta[i] ,axis=0, keepdims=True)
            # Eqn 1
            _W1 = (1-self.beta1)*uW/(1-self.beta1**Epoch_Count)
            _B1 = (1-self.beta1)*uB/(1-self.beta1**Epoch_Count)
            # Eqn 2
            _W2 = (1-self.beta2)*uW**2/(1-self.beta2**Epoch_Count)
            _B2 = (1-self.beta2)*uB**2/(1-self.beta2**Epoch_Count)
            # Eqn 3
            self.W[i] -= self.learning_rate*_W1/((_W2+self.eps)**0.5)
            self.B[i] -= self.learning_rate*_B1/((_B2+self.eps)**0.5)
            if np.isnan(self.W[i]).any() or np.isnan(self.B[i]).any():
                raise Exception('NAN value arises')


    def train(self,epochs=1000,batchsize=30,learning_rate=0.001,              optimizer='myopt',cost='cross_entropy',metric='accuracy',es=(True,0,True),amsgrad=False):
        
        self.cost, self.metric, self.learning_rate = cost, metric, learning_rate
        self.costs, self.mvalues, self.f1m, self.f1M, self.vmvalues = [], [], [], [], []
        if es[0]: prev_entropy = [np.inf]
        # Random value at starting NN
        f = open('continue_next_epoch','w')
        f.close()

        for T in range(epochs):
            if 'continue_next_epoch' not in os.listdir(): break
            init = datetime.now()
            print('Epoch {0:{1}}'.format(T+1,int(np.log10(epochs+1))+1))
            if es[0]: W,B = [deepcopy(self.W)],[deepcopy(self.B)] # Saving Weights for Early Stopping
            splits = int(np.ceil(self.data.shape[0]/batchsize))
            self.index_set = Split(self.data, self.labels, splits ,'R')
            for ln in range(len(self.index_set)):
                train_indx, test_indx = self.index_set[ln]
                self.fdata,self.flabels = self.data[test_indx],self.labels[test_indx]
                self.forward_prop()
                self.back_prop3(T+1)
            # Early Stopping using Validation Set #CHECKPOINT
            if es[0]:
                if es[1]==-1:
                    pass
                else:
                    delta = 0 # Exploring with compromising observed value
                    self.fdata,self.flabels = self.vdata,self.vlabels
                    y_pred = self.forward_prop(predict=True)
                    costV  = eval('CostV.{0}(self.flabels,y_pred)'.format(self.cost))
                    best_entropy, cur_entropy = min(prev_entropy), np.mean(costV)
                    if ( cur_entropy - best_entropy) > delta :
                        if len(prev_entropy)==(es[1]+1):
                            if es[2]: # Restoring Best Weights
                                bst_indx = len(prev_entropy)-prev_entropy[::-1].index(best_entropy) - 1
                                self.W,self.B = W[bst_indx], B[bst_indx]
                            print(']\n',best_entropy,'==>',cur_entropy)
                            break
                        else:
                            prev_entropy.append( cur_entropy )
                            W.append(deepcopy(self.W)); B.append(deepcopy(self.B))
                    else:
                        W,B = [deepcopy(self.W)],[deepcopy(self.B)]
                        prev_entropy = [ cur_entropy ]

        
    def save_model(self,model_store='models'):
        if model_store not in os.listdir():
            os.mkdir(model_store)
        try:
            try:
                shutil.rmtree('{}/{}'.format(model_store,self.name))
            except:
                pass
            finally:
                os.mkdir('{}/{}'.format(model_store,self.name))
                os.chdir('{}/{}'.format(model_store,self.name))
            with open('config','w') as f:
                print(repr(self.Num) ,file=f)
                print(repr(self.fun) ,file=f)
                print(self.preprocess_mode,end = '',file=f)                
            dct = {}
            with open('parameters','wb') as f:
                if self.preprocess_mode == 'standard':
                    dct['mean'], dct['std'] = self.mean, self.std
                elif self.preprocess_mode == 'scale':
                    dct['mn'], dct['mx'] = self.mn, self.mx
                else:
                    raise Exception('Code should be unreachable')
                for i in self.W:
                    dct['W{}'.format(i)] = self.W[i]
                    dct['B{}'.format(i)] = self.B[i]
                np.savez(f,**dct)
        except Exception as exc:
            pass
        finally:
            os.chdir('../..')
    
    def load_model(self,model_store = 'models'):
        if model_store.split('/')[-1] not in os.listdir( model_store.split('/')[0] if len(model_store.split('/'))==2 else '.' ):
            raise Exception("{} directory does not Exist".format(model_store))
        try:
            os.chdir('{}/{}'.format(model_store,self.name))
            with open('config') as f:
                self.Num = eval(f.readline().strip())
                self.fun = eval(f.readline().strip())
                self.preprocess_mode = f.readline().strip()
            with open('parameters','rb') as f:
                npzfile = np.load(f)
                if self.preprocess_mode == 'standard':
                    self.mean, self.std = npzfile['mean'], npzfile['std']
                elif self.preprocess_mode == 'scale':
                    self.mn, self.mx = npzfile['mn'], npzfile['mx']
                else:
                    raise Exception('Code should be unreachable')
                for i in range(len(self.Num)):
                    self.W[i] = npzfile['W{}'.format(i)]
                    self.B[i] = npzfile['B{}'.format(i)]
        except Exception as exc:
            print('EXCEPOTION IN LOADING MODEL')
        finally:
            os.chdir('../..')
    

rate = {'MNIST':0.001,'Cat-Dog':0.0001}
batchsize = {'MNIST':1000,'Cat-Dog':200}
epochs = {'MNIST':100,'Cat-Dog':200}
    
net = NN()
net.name = name
if '--train-data' in argv:
    train_path = argv[argv.index('--train-data')+1]
    add_dataset(train_path)
    X,Y,targets = datasets[name]
    prm_fldr = 'temporary'
    net.data_feed(X,Y,targets)
    net.data_preprocess()
    #config = argv[argv.index('--configuration')+1] 
    if type(config[0])==tuple:
        Num,fun = [x for (x,y) in config], [y for (x,y) in config]
    else:
        Num,fun = [x for x in config], (['relu']*(len(config)-1)) + ['tanh',]
    net.Num = Num
    net.fun = fun
    net.add(Y.shape[1],'softmax')
    net.initialize_layers()
    net.train( epochs[name], batchsize[name], rate[name], 'myopt', 'cross_entropy', 'accuracy', (False,-1,False), False  )
    net.save_model(prm_fldr)
else:
    prm_fldr = '../models'

net.load_model(prm_fldr)
add_dataset(test_path)    
X,Y,targets = datasets[name]
net.data_feed(X,Y,targets)
net.data_preprocess(net.preprocess_mode)
net.fdata = net.data
y_pred = one_hot( net.forward_prop(predict=True) )
y_act = Y


sys.stdout = sys.__stdout__ =  res_stdout

print( 'Test Accuracy :: {:.4f}%'.format(accuracy_score(y_pred,y_act)*100))
print( 'Test Macro F1-score :: {:.4f}%'.format(f1_score(y_pred,y_act,average='macro')*100))
print( 'Test Micro F1-score :: {:.4f}%'.format(f1_score(y_pred,y_act,average='micro')*100))
