#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Image compression for multi images by HOSVD 
# 2017,2018 Tsuyoshi Okubo
# 2019, October, modified by TO
# 2020, October, modified by TO


# By using the low rank approximation through HOSVD, perform data compression of multiple gray scale images. 
# 
# You can change input images by modifying the "input_dir" variable.
# You also need to specify the file type as "file_type" (default is file_type=bmp). 
# 
# We assume filenames are "1.bmp", "2.bmp", ...
# (When you set a different file_type, such as file_type="jpg", the filenames become, "1.jpg", "2.jpg", ...)
# Please set proper value for "n_image" depending the number of images.
# 
# In the ".sample/" directory, there are 10 images taken from ORL Database of Faces, AT&T Laboratories Cambridge.
# 
# Also, you can set the rank of approximation by varying "chi" and "chi_p", which correspond the rank for images and the rank for # of images.
# 
# The output images (after data compression) are saved into "./outputs/" directory. You can change the output directorpy by modifying the "output_dir" variable.
# 
# Let's see, how the images change when you change the ranks.

# In[2]:


## import libraries
from PIL import Image ## Python Imaging Library
import numpy as np ## numpy
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[3]:


n_image = 10 ## set number of images
input_dir="./samples/" ## input directory
output_dir="./outputs/" ## output directory
file_type="bmp" ## file format

chi = 30 ## rank for images
chi_p = 9 ## rank for # of images


# In[4]:


array=[]
for i in range(1,n_image+1):
    img = Image.open(input_dir+repr(i)+"."+file_type) ## load bmp image    
    #img = Image.open("./samples/"+repr(i)+".jpg") ## load jpg image    
    array.append(np.array(img,dtype=float)) ## put into ndarray
    ## array.append(np.array(img.convert("L",dtype=float))) ## put into ndarray (for color images)
array=np.array(array).transpose(1,2,0) 
print("Array shape:" +repr(array.shape)) ## print array shape


# In[5]:


array_truncated = np.zeros(array.shape)

print("HOSVD: chi=" +repr(chi) + ", chi_p=" + repr(chi_p))


# In[6]:


## row
matrix = np.reshape(array,(array.shape[0],array.shape[1]*array.shape[2]))
u,s,vt = np.linalg.svd(matrix[:,:],full_matrices=False) ## svd 
    
#truncation
u1 = u[:,:chi]

## column
matrix = np.reshape(np.transpose(array,(1,0,2)),(array.shape[1],array.shape[0]*array.shape[2]))
u,s,vt = np.linalg.svd(matrix[:,:],full_matrices=False) ## svd 
    
#truncation
u2 = u[:,:chi]

## layer
matrix = np.reshape(np.transpose(array,(2,0,1)),(array.shape[2],array.shape[0]*array.shape[1]))
u,s,vt = np.linalg.svd(matrix[:,:],full_matrices=False) ## svd 
    
#truncation
u3 = u[:,:chi_p]


# In[7]:


## make projectors
p1 = np.dot(u1,(u1.conj()).T)
p2 = np.dot(u2,(u2.conj()).T)
p3 = np.dot(u3,(u3.conj()).T)


# In[8]:


## make truncated array
array_truncated = np.tensordot(np.tensordot(np.tensordot(array,p1,axes=(0,1)),p2,axes=(0,1)),p3,axes=(0,1))
normalized_distance = np.sqrt(np.sum((array-array_truncated)**2))/np.sqrt(np.sum(array**2))
print("Low rank approximation by HOSVD with chi= " +repr(chi)+ ", chi_p= "+repr(chi_p))
print("Normalized distance:" +repr(normalized_distance)) ## print normalized distance


# ## Approximated images
# In the following we show several images before and after the low rank approximation.
# 
# By changing 
# * images
# 
# you can select images to be shown.

# In[9]:


images = [0,1,2,3,4] ## indices of images to be shown.


# In[10]:


plt.figure(figsize=(array.shape[1]*len(images)*0.02,array.shape[0]*0.02))
plt.suptitle("Original objects")
for i in range(len(images)):
    img = Image.fromarray(np.uint8(np.clip(array[:,:,images[i]],0,255))) ## convert to each image
    plt.subplot(1,len(images), i+1)
    plt.axis("off")
    plt.imshow(img,cmap='gray')


# In[11]:


plt.figure(figsize=(array.shape[1]*len(images)*0.02,array.shape[0]*0.02))
plt.suptitle("Compressed objects")
for i in range(len(images)):
    img_truncated = Image.fromarray(np.uint8(np.clip(array_truncated[:,:,images[i]],0,255))) ## convert to each image
    plt.subplot(1,len(images), i+1)
    plt.axis("off")   
    plt.imshow(img_truncated,cmap='gray')


# In[12]:


plt.show()


# In[13]:


## save compressed images
import os
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
for i in range(1,n_image+1):    
    img_truncated = Image.fromarray(np.uint8(np.clip(array_truncated[:,:,i-1],0,255))) ## convert to each image
    #img_truncated.save(output_dir+repr(i)+".bmp") ## save compressed image
    img_truncated.save(output_dir+repr(i)+"."+file_type) ## save compressed image


# In[ ]:




