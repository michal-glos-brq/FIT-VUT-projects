import numpy as np
import pickle
import torch
from torch.nn import Linear, MSELoss, Softmax, Flatten, Sigmoid, LeakyReLU, Sequential
# Sadly, you will not see progress
#from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

class GCN(torch.nn.Module):
    def __init__(self, input_len, output_len, bias=False, avg=False, agg_post=False):
        super(GCN, self).__init__()
        # Linear for each vertice
        self.linear = Linear(input_len, output_len, bias=bias)
        # Save some parameters for forward propagation
        self.avg, self.post = avg, agg_post

    def forward(self, data):
        # Unpack data
        vertices, g_mat = data[0], data[1]
        # Aggregate neighbours
        vertices = torch.bmm(g_mat, vertices)
        # Use average of neighbouring vertices
        if self.avg:
            vertices /= torch.sum(g_mat, dim=1).reshape((-1,34,1))
        # Feed vertices into linear layer
        vertices = self.linear(vertices)
        return vertices

class Estimator(torch.nn.Module):
    def __init__(self, gcn=[8,6,4,2], linear=[40,24,12], activ_gcn=LeakyReLU, activ_lin=Sigmoid,
                 bias=False, lin_bias=False):
        super(Estimator, self).__init__()
        # Define GCNS
        self._gcn = []
        for i in range(1, len(gcn)):
            _bias = bias if isinstance(bias, bool) else bias[i-1]
            self._gcn.append(Sequential(
                GCN(gcn[i-1], gcn[i], bias=_bias),
                activ_gcn()
            ))
        self.gcn = torch.nn.ModuleList(self._gcn)
        # "constant" for reshaping data in order to feed it into linear layers
        self.gcn_transform = gcn[-1] * 34
        # Define linear layers input and output sizes
        linear = [gcn[-1]*34] + linear + [4]
        
        self.linear_in = Sequential(
            Flatten()
        )
        
        self._linears = []
        for i in range(1, len(linear)-1):
            self._linears.append(Sequential(
                Linear(linear[i-1], linear[i], bias=lin_bias),
                activ_lin()
            ))
        self.linears = torch.nn.ModuleList(self._linears)
        
        self.out = Sequential(
            Linear(linear[-2], linear[-1], bias=lin_bias),
            Softmax(1)
        )
    
    def forward(self, vertices, graph):
        for gcn in self.gcn:
            vertices = gcn([vertices, graph])
        vertices = vertices.reshape((-1,self.gcn_transform))
        vertices = self.linear_in(vertices)
        for linear in self.linears:
            vertices = linear(vertices)
        vertices = self.out(vertices)
        return vertices

def train(net, loss_fn, optimizer, batch, epochs, vert_size=4, scheduler=None):
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(0,len(trainX_mat),batch): # tqdm(range(0,len(trainX_mat),batch), ncols=100, desc=f"Learning epoch {epoch+1}."):  # :( No progress info
            vertices = trainX_vert[i:i+batch].reshape((-1,34,vert_size))
            graph = trainX_mat[i:i+batch].reshape((-1,34,34))
            optimizer.zero_grad()
            outputs = net(vertices, graph)
            loss = loss_fn(outputs, trainY[i:i+batch].reshape((-1,4)))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        net.eval()
        evaluate = loss_fn(net(testX_vert.reshape((-1,34,vert_size)), testX_mat.reshape((-1,34,34))), testY.reshape(-1,4))
        scheduler.step(evaluate)
        print('%d: loss: %.6f' % (epoch + 1, running_loss / (len(trainX)/batch)), "eval: "+ str(evaluate.item()))
        running_loss = 0.0
        net.train()


if __name__ == "__main__":
    ## Open pickled dataset
    with open('data_processing/dataset.p', 'rb') as f:
        _data = pickle.load(f)
    # Use 90% for training, the rost for test
    treshold = int(0.9*len(_data))

    # Load the data into batches
    data = np.array(_data)
    np.random.shuffle(data)
    train_data = data[:treshold]
    test_data = data[treshold:]
    trainX, trainY = train_data.transpose()
    testX, testY = test_data.transpose()

    # For now, choose only graph matrix
    trainX_dic = np.array([t[2] for t in trainX])
    testX_dic = np.array([t[2] for t in testX])
    trainX_own = np.array([t[1] for t in trainX])
    testX_own = np.array([t[1] for t in testX])
    trainX_mat = np.array([t[0] for t in trainX])
    testX_mat = np.array([t[0] for t in testX])
    trainY = np.array([np.array(y) for y in trainY]).astype(np.float32)
    testY = np.array([np.array(y) for y in testY]).astype(np.float32)
    # Create from scalar dice a vector with dice on position of player id - 1
    trainX_dic_ = trainX_dic.reshape((-1,34,1)) * trainX_own
    testX_dic_ = testX_dic.reshape((-1,34,1)) * testX_own
    testX_vert = testX_dic_
    trainX_vert = trainX_dic_

    # Create torch datasets
    trainX_mat = torch.FloatTensor(trainX_mat)
    trainX_vert = torch.FloatTensor(trainX_vert)
    trainY = torch.FloatTensor(trainY)
    testX_mat = torch.FloatTensor(testX_mat)
    testX_vert = torch.FloatTensor(testX_vert)
    testY = torch.FloatTensor(testY)

    vert_size=4
    net1 = Estimator(gcn=[4,12,20,28], linear=[1088, 256], bias=False, lin_bias=False)
    criterion1 = MSELoss()
    optimizer1 = Adam(net1.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer1, 'min')
    train(net1, criterion1, optimizer1, 32, 10, vert_size=vert_size, scheduler=scheduler)
    # Save the model
    torch.save(net1.state_dict(), 'estimator')
