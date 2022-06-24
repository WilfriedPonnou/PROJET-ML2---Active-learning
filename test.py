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
from modAL.uncertainty import *
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score

@st.cache
def dataset_preprocessing(uploaded_files):
    
    # Import dataframe
    df = pd.read_csv(uploaded_files)
    
    # Clean useless data
    df.replace("", float("NaN"), inplace=True)
    df.dropna(inplace=True)
    #st.write(df)
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

    transformed_df = scaler.fit_transform(transformed_df)

    #return the two dataframes
    return [df, pd.DataFrame(transformed_df,columns= all_columns)]


st.set_page_config(layout="wide")
choice= st.sidebar.radio('Select one:', ['Normal dataset active learning','Plots for correlation'])
with st.expander("Your dataset, assuming that label is in last column" ,expanded=True):
        
    uploaded_files = st.file_uploader("Upload Dataset", type=["csv"], accept_multiple_files = False)
        
    if uploaded_files is not None:
        global df,df_transformed
        df,df_transformed=dataset_preprocessing(uploaded_files)
        df = pd.DataFrame(df)
        df_transformed=pd.DataFrame(df_transformed)
        global df_feature
        df_feature = df.drop(columns=df.columns[-1],axis=1)
        global df_user
        uploaded_files.seek(0)
        df_user=pd.read_csv(uploaded_files)
        st.write(df_user.head())
        st.write(df_transformed.head())
        df_user=df_user.drop(columns=df_user.columns[-1],axis=1)



if choice == 'Normal dataset active learning':
    

    
    with st.expander("Labelling Function" ,expanded=False):
        
        quest = st.text_input("Label Question")
        
        if quest!="":
        
            X_train, X_test, y_train, y_test = train_test_split(df_feature, df_transformed[df_transformed.columns[-1]].values)
            qery_strat=st.selectbox("Query strategy:",(uncertainty_sampling,entropy_sampling,margin_sampling))
            if qery_strat:
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                model = LogisticRegression()
                learner = ActiveLearner(
                    estimator = model,
                    query_strategy=qery_strat,
                    X_training=X_train, y_training=y_train
                )
                button_termniaed=st.button("terminate")
                #df_transformed.drop(columns=df_transformed.columns[-1], axis=1, inplace=True)
                if 'accuracy_score' not in st.session_state:
                    st.session_state['accuracy_score'] =[0]
                if 'nb_queries' not in st.session_state:
                    st.session_state['nb_queries'] =[0]
                if 'loss' not in st.session_state:
                    st.session_state['loss']=[0]
                if "f1_score" not in st.session_state:
                    st.session_state['f1_score']=[0]
                X_train = pd.DataFrame(X_train)

                if 'x_train' not in st.session_state:
                    st.session_state['x_train']=X_train
                if 'i' not in st.session_state:
                    st.session_state['i']=0
                input_holder=st.empty()
                quest_holder=st.empty()
                x_holder=st.empty()

                while button_termniaed==False:



                    id_tolabel, x_to_label = learner.query(st.session_state['x_train']) 
                    quest_holder.write(quest)
                    x_holder.write(df.iloc[id_tolabel])

                    y =input_holder.text_input("Your answer",key=f"label_{st.session_state['i']}")
                    if not y:
                        st.warning('Please input a label')
                        st.stop()

                    y_return=np.array([int(y)])
                    learner.teach(x_to_label, y_return)
                    st.session_state['x_train'].drop_duplicates(inplace=True)
                    st.session_state['x_train'].drop(id_tolabel,axis=0,inplace=True)
                    st.session_state['accuracy_score'].append(learner.score(X_test, y_test))
                    st.session_state['loss'].append(log_loss(y_test,learner.predict(X_test)))
                    st.session_state['nb_queries'].append(st.session_state['nb_queries'][-1]+1)
                    st.session_state['f1_score'].append(f1_score(y_test,learner.predict(X_test)))
                    c1,c2,c3=st.columns(3)
                    with c1:
                        fig, ax = plt.subplots()
                        ax.plot(st.session_state['nb_queries'], st.session_state['accuracy_score'])
                        ax.set_xlabel('nb_queries')
                        ax.set_ylabel('accuracy_score')
                        st.pyplot(fig)
                    with c2:
                        fig1, ax1 = plt.subplots()
                        ax1.plot(st.session_state['nb_queries'], st.session_state['loss'])
                        ax1.set_xlabel('nb_queries')
                        ax1.set_ylabel('loss')
                        st.pyplot(fig1)
                    with c3:
                        fig2, ax2 = plt.subplots()
                        ax2.plot(st.session_state['nb_queries'], st.session_state['f1_score'])
                        ax2.set_xlabel('nb_queries')
                        ax2.set_ylabel('f1_score')
                        st.pyplot(fig2)    
                    st.write("Dataframe shape: ", st.session_state['x_train'].shape)
                    st.write("New Score: ", learner.score(X_test, y_test))
                    st.write("New Loss: ", log_loss(y_test,learner.predict(X_test)))
                    st.write("New f1_score: ", f1_score(y_test,learner.predict(X_test)))
                    st.write("Iteration number: ", st.session_state['i'])
                    #st.write(st.session_state['nb_queries'])
                    #st.write(st.session_state['accuracy_score'])
                    #st.write(st.session_state['loss'])
                    
                    st.session_state['i']+=1
                    if button_termniaed==False:
                        input_holder.empty()
                        quest_holder.empty()
                        x_holder.empty()


                input_holder.empty()
                quest_holder.empty()
                x_holder.empty()
                best_accuracy=max(st.session_state['accuracy_score'][2:])
                best_f1_score=max(st.session_state['f1_score'][2:])
                y_pred=learner.predict(X_test)
                best_loss=min(st.session_state['loss'][2:])
                iterations=st.session_state['i']
                st.write("The best accuracy obtained is:", best_accuracy)
                st.write("It was achieved at the ",[index for index, item in enumerate(st.session_state['accuracy_score']) if item == best_accuracy]," th iterations")
                st.write("The best loss obtained is ",best_loss)
                st.write("It was achieved at the ",[index for index, item in enumerate(st.session_state['loss']) if item == best_loss]," th iterations")
                st.write("The best f1_score ",best_f1_score)
                st.write("It was achieved at the ",[index for index, item in enumerate(st.session_state['f1_score']) if item == best_f1_score]," th iterations")
                st.write("Number of iterations: ",iterations)


if choice == 'Plots for correlation':
    for i in range (len(df.columns)-1):
        fig, ax = plt.subplots()
        ax.scatter(df.iloc[:,i], df.iloc[:,-1])
        ax.set_xlabel(str(df.columns[i]))
        ax.set_ylabel(str(df.columns[-1]))
        st.pyplot(fig)

