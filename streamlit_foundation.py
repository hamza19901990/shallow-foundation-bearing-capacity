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

req_col_names = ["B(m)", "D(m)", "L/B", "angle (degree)","unit weight (kN/m3)","qu"]
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

import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Sample data (replace with your own data)
# X, y = your_features, your_labels
data = pd.read_csv(r"foundation1.csv")
X = data.iloc[:,:-1]         # Features - All columns but last
y = data.iloc[:,-1]
# Split the data

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.15,random_state =0)
# Initialize and train the AdaBoostRegressor
model = GradientBoostingRegressor(learning_rate=0.01, n_estimators=600, max_depth=3.0)
model.fit(X_train, y_train)

# Save the model to a pickle file
with open('GBRT_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'ada_boost_model.pkl'")
st.sidebar.header('Specify Input Parameters')
"d10", "d50", "d60", "e"
def get_input_features():
    B(m) = st.sidebar.slider('B(m)', 0.030,3.016,0.050)
    D(m) = st.sidebar.slider('D(m)',0.000,0.890,0.500)
    L/B = st.sidebar.slider('L/B', 1.000,6.000,3.000)
    angle (degree) = st.sidebar.slider('angle (degree)', 31.950,45.700,33.000)
    unit weight (kN/m3) = st.sidebar.slider('unit weight (kN/m3)', 9.850,20.800,20.600)





    data_user = {'B(m)': B(m),
            'D(m)': D(m),
            'L/B': L/B,
            'angle (degree)': angle (degree),
            'unit weight (kN/m3)': unit weight (kN/m3),

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
