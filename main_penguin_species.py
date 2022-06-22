import streamlit as st 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import pickle
from streamlit_lottie import st_lottie
import requests
st.set_page_config(page_title='Penguin Classifier',layout='centered',)

def load_lottie(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

url = 'https://assets10.lottiefiles.com/datafiles/fZa46kiFXcqYseC/data.json'
animation_data = load_lottie(url)
st_lottie(animation_data,height=200)



st.title('Penguin Classifier')
st.write("This app uses 6 inputs to predict the species of penguin using a model built on the Palmer's Penguin's dataset. Use\
the form below to get started!")

###################################
st.subheader("Enter Password")
password_guess = st.text_input('What is the Password?')
if password_guess != st.secrets['password']:
    st.stop()
####################################

# Read In Models and Classes
model = open('model_training_script/random_forest/random_forest_model.pickle','rb')
classes = open('model_training_script/random_forest/classes_map.pickle','rb')

clf = pickle.load(model)
class_map = pickle.load(classes)
features = ['bill_length','bill_depth','flipper_length','body_mass',
            'island_biscoe','island_dream','island_torgerson',
            'sex_female','sex_male']
# Save Memory
model.close()
classes.close()

################################################################
# Display Model Information and Unique Classes
st.info(f"Here is the model used for this projecct demo:{clf}")
st.info(f"Unique Classes For Predictions {list(class_map)}")
# User Inputs

st.markdown("User Inputs")
with st.form("User Inputs",clear_on_submit=True):

    island = st.selectbox('Penguin Island', options=['Biscoe', 'Dream', 'Torgerson'])
    sex = st.selectbox('Sex', options=['Female', 'Male'])
    bill_length = st.number_input('Bill Length (mm)', min_value=0)
    bill_depth = st.number_input('Bill Depth (mm)', min_value=0)
    flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
    body_mass = st.number_input('Body Mass (g)', min_value=0)

    submit = st.form_submit_button()

    if submit:
        # Injest user inputs and transform to model accepted format
        island_biscoe, island_dream, island_torgerson = 0, 0, 0
        sex_female, sex_male = 0, 0

        if island == 'Biscoe':
            island_biscoe = 1
        elif island == 'Dream':
            island_dream = 1
        elif island == 'Torgerson':
            island_torgerson = 1

        if sex == 'Female':
            sex_female = 1
        elif sex == 'Male':
            sex_male = 1



        prediction = clf.predict([
            [bill_length,bill_depth,flipper_length,body_mass,
            island_biscoe,island_dream,island_torgerson,
            sex_female,sex_male]    
        ])
        text_prediction = class_map[prediction][0]
        st.write(f'We predict your penguin is of the {text_prediction} species')

        prediction_proba = clf.predict_proba([
            [bill_length,bill_depth,flipper_length,body_mass,
            island_biscoe,island_dream,island_torgerson,
            sex_female,sex_male]    
        ])
        prediction_proba = np.round(prediction_proba,2)

        st.subheader('User inputs are {}'.format([island, sex, bill_length,bill_depth, flipper_length, body_mass]))
        col1, col2, col3 = st.columns(3)
        col1.metric(f"{class_map[0]}", f"{prediction_proba[0][0]*100}%")
        col2.metric(f"{class_map[1]}", f"{prediction_proba[0][1]*100}%")
        col3.metric(f"{class_map[2]}", f"{prediction_proba[0][2]*100}%")



        fig, ax = plt.subplots(figsize=(5,5))
        sns.barplot(y = clf.feature_importances_, x= features, ax = ax)
        plt.title('Which features are the most important for species prediction?')
        plt.xticks(rotation=45)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        st.pyplot(fig)