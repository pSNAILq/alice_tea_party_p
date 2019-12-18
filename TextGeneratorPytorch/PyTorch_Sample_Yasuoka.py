import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm 
%matplotlib inline


np.random.seed(0)
data_num = 2000
test_data_num = 100
min_length = 2
max_length = 100
fixed_length = 100
min_num = 0
max_num = 3
def generate_data(n):
    x_data = []
    y_data = []
    for i in range(n):
        length = np.random.randint(min_length, max_length + 1)
        dt = np.zeros(fixed_length)
        dt[:length] = np.random.randint(min_num, max_num + 1, size=length)
        x_data.append(list(dt))
        y_data.append(sum(dt))
    
    # for PyTorch
    x_data = torch.from_numpy(np.array(x_data)).float()
    y_data = torch.from_numpy(np.array(y_data)).float().view(-1, 1)
    return x_data, y_data
X_train, y_train = generate_data(data_num)
train = data.TensorDataset(X_train, y_train)
train_loader = data.DataLoader(train, batch_size=100, shuffle=True)

class Predictor(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Predictor, self).__init__()
        self.rnn = nn.RNN(n_input, n_hidden, num_layers=1, batch_first=True)
        self.out = nn.Linear(n_hidden, n_output)
        
    def forward(self, x, h=None):
        output, hp = self.rnn(x.unsqueeze(1), h)
        output = self.out(output.squeeze(1))
        return output, hp
    model = Predictor(fixed_length, 64, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.005)
n_epoch = 100
lloss = []
for epoch in tqdm(range(n_epoch)):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        bX, by = data
        bX, by = Variable(bX), Variable(by)
        optimizer.zero_grad()
        output, _ = model(bX)
        loss = criterion(output, by)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.data[0]
        
    lloss.append(running_loss)