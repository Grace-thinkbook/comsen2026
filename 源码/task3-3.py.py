import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt

b=64
c=5
d=0.001

tf=transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.1307,),(0.3081,))
])

tr=datasets.MNIST("./mnist_data",train=True,download=True,transform=tf)
te=datasets.MNIST("./mnist_data",train=False,download=True,transform=tf)
tl=torch.utils.data.DataLoader(tr,batch_size=b,shuffle=True)
tel=torch.utils.data.DataLoader(te,batch_size=b,shuffle=False)

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=nn.Linear(28*28,256)
        self.l2=nn.Linear(256,128)
        self.l3=nn.Linear(128,10)
    def forward(self,x):
        x=x.view(-1,28*28)
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=self.l3(x)
        return x

m=M()
o=torch.optim.Adam(m.parameters(),lr=d)

#e=[],los=[],tra=[],tea=[]
s = []

for i in range(c):
    m.train()
    l0=0
    a0=0
    for x,y in tl:
        z=m(x)
        f=F.cross_entropy(z,y)
        o.zero_grad()
        f.backward()
        o.step()
        
        l0+=f.item()
        p=z.argmax(dim=1,keepdim=True)
        a0+=(p==y.view_as(p)).sum().item()
    
    l1 = l0 / len(tl)
    a1 = a0 / len(tl.dataset)
    s.append({"epoch":i,"loss":l1,"acc":a1})
    print(f"epoch:{i}, loss:{l1:.4f}, acc:{a1:.4f}")

d = pd.DataFrame(s)
d.to_csv("train_log.csv", index=False)
print("success")
d.plot(x="epoch", y=["loss","acc"], title="train curve")
plt.show()