#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install ultralytics')


# In[2]:


from ultralytics import YOLO


# In[3]:


model = YOLO("yolo11n-cls.pt")


# In[8]:


results = model.train(data="/Users/anushkamishra/Desktop/INFOYSIS TRAINING/Classification model/Classification dataset", epochs=12, imgsz=640, batch=8)


# In[5]:


model = YOLO("runs/classify/train6/weights/best.pt")


# In[6]:


from ultralytics import YOLO
model = YOLO("/Users/anushkamishra/runs/classify/train7/weights/best.pt")


# In[7]:


results = model.predict(source="/Users/anushkamishra/Desktop/INFOYSIS TRAINING/Classification model/car.jpeg")
if results:
    results[0].show()
else:
    print("error")


# In[10]:


results = model.predict(source="/Users/anushkamishra/Desktop/INFOYSIS TRAINING/Classification model/aadhar_cardv3.jpeg")
if results:
    results[0].show()
else:
    print("error")


# In[ ]:




