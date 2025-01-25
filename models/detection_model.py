#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ultralytics import YOLO


# In[2]:


model = YOLO("yolo11n.pt")


# In[3]:


#results = model.train(data="/Users/anushkamishra/Desktop/INFOYSIS TRAINING/detection_model/data.yaml", epochs=25, imgsz=640)


# In[4]:


model = YOLO("/Users/anushkamishra/runs/detect/train3/weights/best.pt")


# In[5]:


metrics = model.val()
metrics.box.map
metrics.box.map50
metrics.box.map75
metrics.box.maps


# In[6]:


results = model("/Users/anushkamishra/Desktop/INFOYSIS TRAINING/detection_model/dataset/dataset/dataset/images/train/6.jpg")


# In[7]:


metrics = model.val()
print(metrics.box.map)


# In[ ]:




