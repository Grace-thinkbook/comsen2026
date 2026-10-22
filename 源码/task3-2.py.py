import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

np.random.se(0)
u=np.linspace(-3,3,50)
v=np.linspace(-3,3,50)
uu,vv=np.meshgrid(u,v)
z=2*uu**2+1.5*uu*vv+3*np.sin(uu)+0.5*np.cos(2*vv)+np.exp(-0.5*uu)+0.8*vv

n=80
p=np.random.uniform(-3,3,n)
q=np.random.uniform(-3,3,n)
r=2*p**2+1.5*p*q+3*np.sin(p)+0.5*np.cos(2*q)+np.exp(-0.5*p)+0.8*q
s=np.random.normal(0,0.3,n)
t=r+s

x=torch.tensor(p,dtype=torch.float32).reshape(-1,1)
y=torch.tensor(q,dtype=torch.float32).reshape(-1,1)
w=torch.tensor(t,dtype=torch.float32).reshape(-1,1)

a=torch.tensor(1.0,requires_grad=True)
b=torch.tensor(1.0,requires_grad=True)
c=torch.tensor(1.0,requires_grad=True)
d=torch.tensor(1.0,requires_grad=True)
e=torch.tensor(-0.2,requires_grad=True)
f=torch.tensor(1.0,requires_grad=True)
g=torch.tensor(2000.0,requires_grad=True)

opt=torch.optim.Adam([a,b,c,d,e,f,g],lr=0.002)
ep=20000

for i in range(ep):
    pred=a*x**2+b*x*y+c*torch.sin(x)+d*torch.cos(2*y)+torch.exp(e*x)+f*y+g
    loss=torch.mean((pred-w)**2)
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        e.clamp_(-2,0)

u1=torch.tensor(u.reshape(-1,1),dtype=torch.float32)
v1=torch.tensor(v.reshape(-1,1),dtype=torch.float32)
surf=a*u1**2+b*u1*v1+c*torch.sin(u1)+d*torch.cos(2*v1)+torch.exp(e*u1)+f*v1+g
surfn=surf.detach().numpy().reshape(u.shape)
sp=a*x**2+b*x*y+c*torch.sin(x)+d*torch.cos(2*x)+torch.exp(e*x)+f*x+g
spn=sp.detach().numpy().flatten()
m=np.mean(t)

print(a.item(),b.item(),c.item(),d.item(),e.item(),f.item(),g.item())
plt.savefig("task2.png", dpi=300)