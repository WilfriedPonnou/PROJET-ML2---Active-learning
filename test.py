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


st.set_page_config(layout="wide")
def dataset_preprocessing(uploaded_files):
    df = pd.read_csv(uploaded_files)
    # Clean useless data
    df.replace("", float("NaN"), inplace=True)
    df.dropna(inplace=True)

    transformed_df = df

    # Get columns that are not usable
    filteredColumns = transformed_df.dtypes[df.dtypes == np.object]
    list_of_columns = list(filteredColumns.index)

    # Transform those columns into usable one
    transformed_df[list_of_columns] = transformed_df[list_of_columns].apply(lambda col:pd.Categorical(col).codes)

    # Scale the two dataframes
    scaler = MinMaxScaler()

    df = scaler.fit_transform(df)
    transformed_df = scaler.fit_transform(transformed_df)

    #return the two dataframes
    return df, transformed_df


def labelize_data(request_value,quest): 
    question = str(quest)+'\n {}'.format(request_value)
    st.write(question)
    i=np.random.randint(1000)


def training_loop(X_train,X_test,y_test,learner,quest):
    input_holder=st.empty()
    quest_holder=st.empty()
    x_holder=st.empty()
    plot_holder=st.empty()
    button_finish=st.button("Finish active learning")
    i=0
    if 'accuracy_score' not in st.session_state:
        st.session_state['accuracy_score'] =[0]
    if 'nb_queries' not in st.session_state:
        st.session_state['nb_queries'] =[1]
    while button_finish!=True:
        fig, ax = plt.subplots()
        ax.plot(st.session_state['nb_queries'], st.session_state['accuracy_score'])
        plot_holder.pyplot(fig)

        X_train = pd.DataFrame(X_train)
        id_tolabel, x_to_label = learner.query(X_train) 
        quest_holder.write(quest)
        x_holder.write(df_transformed.iloc[id_tolabel])
        y = input_holder.text_input("Your answer",key=f"label_{i}")
        if y is not None:
            y_return=np.array([int(str(y))])
            # récupération de la donnée à labeliser

            # ajout des données à l'ensemble de train et train
            learner.teach(x_to_label, y_return)

            #suppression de la donnée labelisée pour ne pas l'avoir à nouveau

            X_train.drop_duplicates(inplace=True)
            X_train.drop(id_tolabel,axis=0,inplace=True)
            st.write(X_train.shape)
                        #todo calculer le score du modelµ
            
            st.session_state['accuracy_score'].append(learner.score(X_test, y_test))
            st.write(learner.score(X_test, y_test))
            st.write(st.session_state['nb_queries'])
            st.write(st.session_state['accuracy_score'])
            #fig, ax = plt.subplots()
            #ax.plot(st.session_state['nb_queries'], st.session_state['accuracy_score'])
            #plot_holder.pyplot(fig)
            input_holder.empty()
            x_holder.empty()
            st.write(st.session_state['nb_queries'][-1]+1)
            st.session_state['nb_queries'].append(st.session_state['nb_queries'][-1]+1)
            i+=1

    return learner

choice= st.sidebar.radio('Select one:', ['Normal dataset active learning'])

if choice == 'Normal dataset active learning':
    
    with st.expander("Your dataset, assuming that label is in last column" ,expanded=True):
        
        uploaded_files = st.file_uploader("Upload Dataset", type=["csv"], accept_multiple_files = False)
        
        if uploaded_files is not None:
            global df,df_transformed
            df,df_transformed=dataset_preprocessing(uploaded_files)
            df = pd.DataFrame(df)
            df_transformed=pd.DataFrame(df_transformed)
            global df_feature
            df_feature = df_transformed.drop(df_transformed.columns[-1],axis=1)
            st.write(df.head())
            st.write(df_feature.head())
    
    with st.expander("Labelling Function" ,expanded=False):
        
        quest = st.text_input("Label Question")
        
        if quest!="":
        
            X_train, X_test, y_train, y_test = train_test_split(df_feature, df_transformed[df_transformed.columns[-1]].values)

            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            model = LogisticRegression()
            learner = ActiveLearner(
                estimator = model,
                query_strategy=uncertainty_sampling,
                X_training=X_train, y_training=y_train
            )
            training_loop(X_train,X_test,y_test,learner,quest)