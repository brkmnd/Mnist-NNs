import torch as ts
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import run_model

model_name = "model_nn2"
model_file_name = model_name + ".ptm"

n_epochs = 10
l_rate = 0.000001

class MnistNN2(nn.Module):
    def __init__(self,input_dim,inter_dim,labels_dim):
        super(MnistNN2,self).__init__()

        self.lin1 = nn.Linear(input_dim,inter_dim)

        self.lin2 = nn.Linear(inter_dim,labels_dim)

    def forward(self,img):
        out = self.lin1(img.view(1,-1))
        out = F.relu(out)
        out = self.lin2(out)
        return F.log_softmax(out,dim=1)

model = MnistNN2(784,1024,10)

model.load_state_dict(ts.load(model_file_name))

run_model.train(model,l_rate,n_epochs,optim.Adam)

run_model.test(model_name,model,n_epochs)

ts.save(model.state_dict(),model_file_name)

