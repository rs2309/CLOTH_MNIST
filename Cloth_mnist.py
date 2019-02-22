#!/usr/bin/env python
# coding: utf-8

# In[29]:


import tensorflow as tf
import numpy as np
from  tensorflow.keras import layers as ly
import keras


# In[ ]:





# In[30]:


(train_im,train_l),(test_im,test_l)=tf.keras.datasets.fashion_mnist.load_data()


# In[ ]:





# In[31]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_im[0,:,:])


# In[32]:


'''
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(train_im[i])  
plt.colorbar()
plt.grid()
plt.show()
'''


# In[33]:


train_im=train_im/255.0
test_im=test_im/255.0


# In[34]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_im[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_l[i]])
plt.show()


# In[82]:


model=tf.keras.Sequential()


# In[83]:


model.add(ly.Flatten(input_shape=(28,28)))
model.add(ly.Dense(128, activation=tf.nn.relu,use_bias=True))
model.add(ly.Dense(10,activation=tf.nn.softmax,use_bias=True))


# In[94]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#y_train = np.zeros((train_l.shape[0], 10))
#y_train[np.arange(train_l.shape[0]), train_l] = 1


# In[98]:


model.fit(train_im,train_l,epochs=20,batch_size=250,shuffle=True)


# In[99]:


predicts=model.predict(test_im)
print(test_im[0,:,:].shape)
test_im[0,:,:].reshape(1,28,28).shape


# In[100]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_im[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[np.argmax(predicts[i,:],axis=0)])
plt.show()


# In[101]:


model.summary()


# In[104]:


model.save('cloth_mnist.hd5')


# In[ ]:




