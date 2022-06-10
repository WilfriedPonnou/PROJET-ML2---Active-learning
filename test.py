import streamlit as st
from PIL import Image
import random
import numpy as np
import pandas as pd
import pymodal
from matplotlib import pyplot as plt


st.set_page_config(layout="wide")
def dataset_preprocessing(df):
    #removing NAN or handling nan
    #Categorical to Number
    #Scaling
def labelize_data(X): 
    question = 'cette donnée correspond-elle à une tumeur cancereuse(1) ou non(à):\n {}'.format(X)
    y = input(question)
    
    return np.array([int(y)])

def load_image(image_file):
    img = Image.open(image_file)
    return img

choice= st.sidebar.radio('Select one:', ['Letters dataset active learning', 'My image dataset active learning','Normal dataset active learning'])


if choice == 'My image dataset active learning':
    
    with st.expander("1)Test Part, expand me 1st",expanded=True):
        with st.form(key='my_form'):
            classes = st.text_input('List the possible classes seperated with a ","')
            nb_of_images = st.text_input('Number of images')
            st.form_submit_button("Let's go")
        
    classes=classes.split(",")
    
    #st.write(classes,nb_of_images)
    st.write("Expand this part in 2nd to labelize test data, then unexpand it when you are done")
    with st.expander("2)Test lablezing",expanded=False):
        if nb_of_images!='' and int(nb_of_images)>10:
            st.write("Let's make it 70/30")
            
            test_image_size=int(int(nb_of_images)*0.3)
            
            st.write("Let's labellize ",test_image_size," images")
            
            st.subheader("Upload your "+str(test_image_size)+" test images, all at once")

            test_image_list=[]
            
            uploaded_files= st.file_uploader("Upload Images",type=["png","jpg","jpeg"], accept_multiple_files = True)
            
            keys = random.sample(range(1000, 9999), test_image_size)
            
            if uploaded_files is not None:
                
                for i,image_file in enumerate(uploaded_files):
                    image = load_image(image_file)
                    st.image(image, width=250)
                    img_array = np.array(image)
                    image_label=st.selectbox('Pick', classes, key=f"label_{i}")
                    test_image_list.append((image_label,image))
  
                st.write(test_image_list)
            else:
                st.write(" Hahaha, nice joke --', please now upload your files")

        else:
            st.write("please upload more images")
#    with st.expander("3)Active learning:",expanded=False):
elif choice == 'Normal dataset active learning':
    with st.expander("Your dataset, assuming that label is in last column" ,expanded=False):
        uploaded_files= st.file_uploader("Upload Dataset",type=["csv"], accept_multiple_files = True)
        if uploaded_files is not None:
            df=pd.read_csv(uploaded_files)
            df_feature = df[["size","p53_concentration"]]

