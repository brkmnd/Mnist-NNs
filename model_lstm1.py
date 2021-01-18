import torch as ts
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import run_model

model_name = "model_lstm1"
model_file_name = model_name + ".ptm"

n_epochs = 5
l_rate = 0.01

class MnistLSTM1(nn.Module):
    def __init__(self,input_dim,hidden_dim,labels_dim):
        super(MnistLSTM1,self).__init__()

        self.lstm = nn.LSTM(input_dim,hidden_dim)

        self.lin1 = nn.Linear(hidden_dim,labels_dim)

    def forward(self,img):
        lstm_out,_ = self.lstm(img.view(1,1,-1))
        out = self.lin1(lstm_out.view(1,-1))
        return F.log_softmax(out,dim=1)

model = MnistLSTM1(784,2048,10)

#model.load_state_dict(ts.load(model_file_name))

run_model.train(model,l_rate,n_epochs,optim.Adam)

run_model.test(model_name,model,n_epochs)

ts.save(model.state_dict(),model_file_name)

