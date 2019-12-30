import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))      

def sig_deriv(x):
    array = []
    for i in x:
        array.append(sigmoid(i) * (1 - sigmoid(i)))
        
    return np.array(array)

inn,layers = np.loadtxt("./xor.txt",delimiter=",").astype(int),np.loadtxt("./config_Rede.txt",delimiter=",").astype(int)

X,y = inn[:,:-1], np.ones((len(inn),1))
y[:,0] = inn[:,-1]

np.random.seed(123)

weights_j = 2 * np.random.random((layers[0],len(X[0]))) -1

weights_k = 2 * np.random.random((layers[1],layers[0]+1)) - 1

epochs = 10000
alpha = 0.2

for epoch in range(epochs+1):
    if epoch%1000 == 0:
            print("Epoch: ",epoch)
    for i,x in enumerate(X):
        #forward: hidden layer
        j_in = np.dot(x,weights_j.T)
        input_k = sigmoid(j_in) 
        input_k = np.append(np.ones(1),input_k)
        
        #forward: output layer
        k_in = np.dot(input_k,weights_k.T)
        y2 = sigmoid(k_in) 
        
        #forward: error
        error = y[i] - y2
        
        if epoch%1000 == 0:
            print("y2",x[1:],": ",y2)
            
            
        #back: delta_k
        delta_k = error * sig_deriv(k_in)
        delta_k = delta_k.reshape(len(delta_k),1)
        
        #back: delta_j
        delta_j = delta_k * (weights_k[:,1:]) * sig_deriv(j_in)
        
        
        #back: Delta_wk and Delta_bk
        delta_wjk = alpha * delta_k * input_k[1:]
        delta_bk  = delta_k * alpha
        
        
        #back: Delta_wj and Delta_bj
        delta_wij = alpha * np.dot(delta_j.T , x[1:].reshape(1,2))
        delta_bj  = delta_j * alpha
        
        
        #update: wk and bk
        weights_k[:,1:] += delta_wjk
        weights_k[:,0] += delta_bk.reshape(-1)
        
        #update: wj and bj
        weights_j[:,1:] += delta_wij
        weights_j[:,0]  += delta_bj.reshape(-1)
        
    if epoch%1000 ==0:
        print("\n")