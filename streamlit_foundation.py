import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import numpy as np
import pandas as pd
import csv
import streamlit as st
from PIL import Image

st.write("""
# Concrete Compressive Strength Prediction
This app predicts the **Unconfied Compressive Strength (UCS) of Geopolymer Stabilized Clayey Soil**!
""")
st.write('---')
image=Image.open(r'foundation.jpg')
st.image(image, use_column_width=True)

data = pd.read_csv(r"foundation1.csv")

req_col_names = ["B", "D", "LoverB", "angle","unit_weight","qu"]
curr_col_names = list(data.columns)

mapper = {}
for i, name in enumerate(curr_col_names):
    mapper[name] = req_col_names[i]

data = data.rename(columns=mapper)
st.subheader('data information')
data.head()
data.isna().sum()
corr = data.corr()
st.dataframe(data)
X = data.iloc[:,:-1]         # Features - All columns but last
y = data.iloc[:,-1]          # Target - Last Column
print(X)

st.sidebar.header('Specify Input Parameters')
def get_input_features():
    B = st.sidebar.slider('B(m)', 0.030,3.016,0.050)
    D = st.sidebar.slider('D(m)',0.000,0.890,0.500)
    LoverB = st.sidebar.slider('L/B', 1.000,6.000,3.000)
    angle = st.sidebar.slider('angle (degree)', 31.950,45.700,33.000)
    unit_weight  = st.sidebar.slider('unit weight (kN/m3)', 9.850,20.800,20.600)





    data_user = {'B(m)': B,
            'D(m)': D,
            'L/B': LoverB,
            'angle (degree)': angle,
            'unit weight (kN/m3)': unit_weight,

    }
    features = pd.DataFrame(data_user, index=[0])
    return features

df = get_input_features()
# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')




# Reads in saved classification model
import pickle
load_clf = pickle.load(open('GBRT_model.pkl', 'rb'))
st.header('Prediction of UCS (Mpa)')

# Apply model to make predictions
prediction = load_clf.predict(df)
st.write(prediction)
st.write('---')
