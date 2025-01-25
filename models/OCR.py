#!/usr/bin/env python
# coding: utf-8

# In[10]:


#get_ipython().system('pip install easyocr')


# In[11]:


from ultralytics import YOLO
import easyocr
import cv2


# In[12]:


model = YOLO("/Users/anushkamishra/runs/detect/train3/weights/best.pt")


# In[13]:


reader = easyocr.Reader(['en'])


# In[14]:


image_path = "/Users/anushkamishra/Desktop/INFOYSIS TRAINING/detection_model/dataset/dataset/dataset/images/train/6.jpg" 
results = model(image_path)


# In[15]:


image = cv2.imread(image_path)
extracted_data = {}


# In[18]:


mage = cv2.imread(image_path)
extracted_data = {}
for result in results[0].boxes.data.tolist():  # results[0].boxes.data contains bounding box details
    x1, y1, x2, y2, confidence, class_id = map(int, result[:6])
    field_class = model.names[class_id]  # Get class name (e.g., 'Name', 'UID', 'Address')

    # Crop the detected region
    cropped_roi = image[y1:y2, x1:x2]

    # Convert cropped ROI to grayscale for OCR
    gray_roi = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)

    # Use EasyOCR to extract text
    text = reader.readtext(gray_roi, detail=0)  # detail=0 returns only the text

    # Save the text to the extracted_data dictionary
    extracted_data[field_class] = ' '.join(text)


# In[19]:


print("Extracted Data:", extracted_data)


# In[ ]:




