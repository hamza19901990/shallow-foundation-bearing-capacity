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
image=Image.open(r'Unconfined-Compressive-Strength-Test-Apparatus (1).jpg')
st.image(image, use_column_width=True)

data = pd.read_csv(r"soil permability.csv")

req_col_names = ["d10", "d50", "d60", "e","k (m/s)"]
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

from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Sample data (replace with your own data)
# X, y = your_features, your_labels

# Split the data

# Initialize and train the AdaBoostRegressor
model = AdaBoostRegressor(learning_rate=0.5, n_estimators=100)
model.fit(X_train, y_train)


print("Model saved as 'ada_boost_model.pkl'")
st.sidebar.header('Specify Input Parameters')
"d10", "d50", "d60", "e"
def get_input_features():
    d10 = st.sidebar.slider('d10', 0.01,0.91,0.05)
    d50 = st.sidebar.slider('d50',0.02,12.00,1.00)
    d60 = st.sidebar.slider('d60', 0.03,19.00,13.00)
    e = st.sidebar.slider('air void', 0.10,0.94,0.85)


    data_user = {'d10': d10,
            'd50': d50,
            'd60': d60,
             'air void': e,

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
load_clf = pickle.load(open('ada_boost_model.pkl', 'rb'))
st.header('Prediction of UCS (Mpa)')

# Apply model to make predictions
prediction = load_clf.predict(df)
st.write(prediction)
st.write('---')
