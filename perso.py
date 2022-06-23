import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IPython import display
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from streamlit.state.session_state import SessionState


def dataset_preprocessing(uploaded_files):
    
    # Import dataframe
    df = pd.read_csv(uploaded_files)
    
    # Clean useless data
    df.replace("", float("NaN"), inplace=True)
    df.dropna(inplace=True)

    # Get the name of all columns
    all_columns = list(df.columns.values)

    # Create the work dataframe
    transformed_df = df

    # Get columns that are not usable
    filtered_columns = transformed_df.dtypes[df.dtypes == np.object]
    list_of_columns = list(filtered_columns.index)

    # Transform those columns into usable one
    transformed_df[list_of_columns] = transformed_df[list_of_columns].apply(lambda col:pd.Categorical(col).codes)

    # Scale the two dataframes
    scaler = MinMaxScaler()

    df = scaler.fit_transform(df)
    transformed_df = scaler.fit_transform(transformed_df)

    #return the two dataframes
    return [pd.DataFrame(df,columns= all_columns), pd.DataFrame(transformed_df,columns= all_columns)]

list_df = dataset_preprocessing('data.csv')



raw = list_df[0]
processed = list_df[1]

print(raw.head(10))