#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Image compression by SVD
# for color image
# 2017, 2018 Tsuyoshi Okubo
# 2019, modified by TO
# 2020, modified by TO


# By using the low rank approximation through SVD, perform data compression of a color image. 
# 
# You can change sample image by modifying file open "sample_color.jpg".
# 
# Also, you can set the rank of approximation by varying "chi".
# 
# Let's see, how the image changes when you change the rank.

# In[2]:


## import libraries
from PIL import Image ## Python Imaging Library
import numpy as np ## numpy
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[3]:


## Rank for the low rank approximation
chi = 50


# In[4]:


img = Image.open("./sample_color.jpg") ## load image
#img = Image.open("./sample_color2.jpg") ## load image
#img.show(title="Original") ## show image
img.save("./img_original.png") ## save image


# In[5]:


array = np.array(img, dtype=float) ## convert to ndarray
print("Array shape:" + repr(array.shape)) ## output array shape


# In[6]:


array_truncated = np.zeros(array.shape)


# In[7]:


## svd for each color
for i in range(3):
    u,s,vt = np.linalg.svd(array[:,:,i],full_matrices=False) ## svd 
    
    #truncation
    u = u[:,:chi]
    vt = vt[:chi,:]
    s = s[:chi]

    array_truncated[:,:,i] = np.dot(np.dot(u,np.diag(s)),vt) ## make truncated array
    
normalized_distance = np.sqrt(np.sum((array-array_truncated)**2))/np.sqrt(np.sum(array**2))
print("Low rank approximation by SVD with chi=" +repr(chi))
print("Normalized distance:" +repr(normalized_distance)) ## print normalized distance


# In[8]:


img_truncated = Image.fromarray(np.uint8(np.clip(array_truncated,0,255))) ## convert to RGB
#img_truncated.show(title="Truncated") ## show image
img_truncated.save("./img_truncated.png") ## save compressed image
#img_truncated.save("./img_truncated.jpg") ## save compressed image in jpg


# In[9]:


plt.figure(figsize=(array.shape[1]*0.01,array.shape[0]*0.01))
plt.axis("off")
plt.title("Original")
plt.imshow(img)

plt.figure(figsize=(array_truncated.shape[1]*0.01,array_truncated.shape[0]*0.01))
plt.axis("off")
plt.title("Compressed")
plt.imshow(img_truncated)

plt.show()

