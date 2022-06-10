import streamlit as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from PIL import Image
import random
import numpy as np
import pandas as pd
import pymodal
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling


st.set_page_config(layout="wide")
def dataset_preprocessing(df):

    # Clean useless data
    df.replace("", float("NaN"), inplace=True)
    df.dropna(inplace=True)

    transformed_df = df

    # Get columns that are not usable
    filtered_columns = transformed_df.dtypes[df.dtypes == np.object]
    list_of_columns = list(filtered_columns.index)

    # Transform those columns into usable one
    transformed_df[list_of_columns] = transformed_df[list_of_columns].apply(lambda col:pd.Categorical(col).codes)

    # Scale the two dataframes
    scaler = MinMaxScaler()

    df = scaler.transform(df)
    transformed_df = scaler.transform(transformed_df)

    #return the two dataframes
    return df, transformed_df

def compare_label(df):
    nb_col = len(df.columns) - 2
    list_of_columns = list(df.index)
    for i in range(nb_col):
        column = list_of_columns[i]
        last_column  = df.iloc[: , -1]
        
        plt.figure(i)
        sns.countplot(x=df[column], data=last_column)


def labelize_data(request_value,quest): 
    question = str(quest)+'\n {}'.format(request_value)
    st.write(question)
    i=np.random.randint(100)
    y = st.text_input("Your answer",key=f"label_{i}")
    y_return=str(y)
    return np.array([int(y_return)])

def training_loop(X_train,X_test,y_test,quest):
    for i in range(10):
        
        # récupération de la donnée à labeliser
        id_tolabel, x_to_label = learner.query(X_train) 
        
        y = labelize_data(x_to_label,quest)
        print(type(x_to_label))
        # ajout des données à l'ensemble de train et train
        learner.teach(x_to_label, y)
        
        #suppression de la donnée labelisée pour ne pas l'avoir à nouveau
        X_train = pd.DataFrame(np.delete(X_train, id_tolabel, 0))
        
        #todo calculer le score du modelµ
   
        print(learner.score(X_test, y_test))

choice= st.sidebar.radio('Select one:', ['Normal dataset active learning'])

if choice == 'Normal dataset active learning':
    
    with st.expander("Your dataset, assuming that label is in last column" ,expanded=True):
        
        uploaded_files = st.file_uploader("Upload Dataset", type=["csv"], accept_multiple_files = False)
        
        if uploaded_files is not None:
            global df
            df = pd.read_csv(uploaded_files)
            global df_feature
            df_feature = df.drop(df.columns[-1],axis=1)
            st.write(df.head())
            st.write(df_feature.head())
    
    with st.expander("Labelling Function" ,expanded=False):
        
        quest = st.text_input("Label Question")
        
        if quest:
            st.write(quest)
        
            X_train, X_test, y_train, y_test = train_test_split(df_feature, df[df.columns[-1]].values)

            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            model = LogisticRegression()
            learner = ActiveLearner(
            estimator = model,
            query_strategy=uncertainty_sampling,
            X_training=X_train, y_training=y_train
            )
            training_loop(X_train,X_test,y_test,quest)