"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries
from PIL import Image
from torchvision import models, transforms
import torch
import streamlit as st
import numpy as np


from inference import predict
from inference import predict

# set title of app
st.title("Breast Cancer Classification Application")
st.write("")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = "png")




if file_up is not None:
    model = torch.load('tumor_resnet50.pt' , map_location=torch.device('cpu'))
    image = Image.open(file_up)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    st.write("Just a second ...")
    pred_label, pred_confidence =  predict(file_up,model)

    
    st.write("Prediction: ", pred_label)
    st.write("Confidence Score: ", pred_confidence)
    
    #print( np.argmax(model_ft(image_loader(data_transforms, image)).detach().numpy()))
    #print(model_ft(image_loader(data_transforms, image)).detach().numpy())
