#!/usr/bin/env python
# coding: utf-8

# ## Convert ONNX

# In[1]:


# torch.__version__ : '1.13.0+cu117'
# np.__version__ : '1.23.5'
# onnxruntime.__version__ : '1.13.1'
size_h = 1000
size_w = 1000


# In[2]:


import torch
import torch.nn as nn
import numpy as np
from utils.file_utils import *

from arch.FastDVDNet import FastDVDNet

model = FastDVDNet(
    in_frames=5
)


# In[3]:


state = torch.load("./models/model_best.pth.tar")
from collections import OrderedDict
new_state_dict = OrderedDict()
for k,x in state['state_dict'].items():
    new_state_dict[k.replace("module.","")] = state['state_dict'][k]
model.load_state_dict(new_state_dict)
model.eval()


# In[ ]:





# In[4]:


frames = torch.zeros([1,5,3,size_h,size_w], dtype=torch.float)
noise_map = torch.zeros([1,3,size_h,size_w], dtype=torch.float)

a0 = torch.cat([frames[:,0:3,...].view(1, -1, size_h, size_w), noise_map], dim=1)
b0 = frames[:,1,...]
c0 = model.block1(a0, b0)

a1 = torch.cat([frames[:,1:4,...].view(1, -1, size_h, size_w), noise_map], dim=1)
b1 = frames[:,2,...]
c1 = model.block1(a1, b1)

a2 = torch.cat([frames[:,2:5,...].view(1, -1, size_h, size_w), noise_map], dim=1)
b2 = frames[:,3,...]
c2 = model.block1(a2, b2)

d = torch.cat([c0, c1, c2, noise_map], dim=1)
e = frames[:, 2, ...]
output = model.block2(d, e)


# In[ ]:





# In[5]:


output.shape


# In[6]:


torch.onnx.export(
    model.block1, 
    args = (a0, b0),
    f = "block1.onnx", 
    verbose=True, 
    input_names=["data", "ref"])


# In[7]:


torch.onnx.export(
    model.block2, 
    args = (d,e),
    f = "block2.onnx", 
    verbose=True, 
    input_names=["data", "ref"])


# ## Test

# In[8]:


import onnxruntime as ort
ort_session_1 = ort.InferenceSession("block1.onnx")
ort_session_2 = ort.InferenceSession("block2.onnx")


# In[9]:


[x.name for x in ort_session_2.get_outputs()]


# In[10]:


get_ipython().system(' mkdir ../imgs/test4')
get_ipython().system(' ffmpeg -i ../imgs/test4.gif -vsync 0 ../imgs/test4/%5d.png')
get_ipython().system('ls ../imgs/test4/')


# In[11]:


import cv2
from glob import glob
files = sorted(glob("../imgs/test4/*.png"))
images = [cv2.imread(x) for x in files]
images = [cv2.cvtColor(x,cv2.COLOR_BGR2RGB) for x in images]
images = [np.clip(x + 20 * np.random.randn(x.size).reshape(x.shape), 0, 255).astype(int) for x in images]


# In[12]:


frames = np.array(images)[np.newaxis,:5]
orig_h = frames.shape[-3]
orig_w = frames.shape[-2]
frames = frames.astype(float)/255.
# orig_r_mean = frames[:,:,:,:,0].mean()
# orig_r_std = frames[:,:,:,:,0].std()
# orig_g_mean = frames[:,:,:,:,0].mean()
# orig_g_std = frames[:,:,:,:,0].std()
# orig_b_mean = frames[:,:,:,:,0].mean()
# orig_b_std = frames[:,:,:,:,0].std()
# frames[:,:,:,:,0] = (frames[:,:,:,:,0]-orig_r_mean)/(orig_r_std)
# frames[:,:,:,:,1] = (frames[:,:,:,:,1]-orig_g_mean)/(orig_g_std)
# frames[:,:,:,:,2] = (frames[:,:,:,:,2]-orig_b_mean)/(orig_b_std)


# In[13]:


frames = np.swapaxes(frames, -1, -3)
frames = np.swapaxes(frames, -1, -2)
frames = np.lib.pad(frames, ((0,0),(0,0),(0,0),(0,size_h-orig_h),(0,size_w-orig_w)), 'constant', constant_values=(0))


# In[ ]:





# In[14]:


frames = frames.astype(np.float32)
noise_map = (np.ones([1,3,size_h,size_w]).astype(np.float32)) * 0.1

a0 = np.concatenate([frames[:,0:3,...].reshape(1, 9, size_h, size_w), noise_map], axis=1)
b0 = frames[:,1,...]
c0 = ort_session_1.run(output_names = ['110'], input_feed={"data":a0, "ref":b0})[0]

a1 = np.concatenate([frames[:,1:4,...].reshape(1, 9, size_h, size_w), noise_map], axis=1)
b1 = frames[:,2,...]
c1 = ort_session_1.run(output_names = ['110'], input_feed={"data":a1, "ref":b1})[0]

a2 = np.concatenate([frames[:,2:5,...].reshape(1, 9, size_h, size_w), noise_map], axis=1)
b2 = frames[:,3,...]
c2 = ort_session_1.run(output_names = ['110'], input_feed={"data":a2, "ref":b2})[0]

d = np.concatenate([c0, c1, c2, noise_map], axis=1)
e = frames[:, 2, ...]
output = ort_session_2.run(output_names = ['110'], input_feed={"data":d, "ref":e})[0]
output = output[0]
output = output.swapaxes(0,2)
output = output.swapaxes(0,1)


# In[15]:


out_img = output[:orig_h, :orig_w, :].copy()
out_img = out_img.clip(0,1) * 255.99
# out_img = np.clip((output[:orig_h, :orig_w, :]+0.5)*255*np.sqrt(0.5), 0, 255)
# out_img[:,:,0] = out_img[:,:,0] * orig_r_std + orig_r_mean
# out_img[:,:,1] = out_img[:,:,1] * orig_g_std + orig_g_mean
# out_img[:,:,2] = out_img[:,:,2] * orig_b_std + orig_b_mean
out_img = out_img.astype(int)


# In[16]:


import matplotlib.pyplot as plt
plt.imshow(images[2])
plt.show()
plt.imshow(out_img)
plt.show()


# In[ ]:





# In[ ]:




