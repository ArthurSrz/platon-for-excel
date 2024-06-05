import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile
import os
from openai import OpenAI
import pandas as pd
import re
import json

# Page title
st.set_page_config(page_title='ML Model Building', page_icon='🤖')
st.title('🤖 Sage comme Platon')
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key = 45)
if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=False)











with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app allow users to build a machine learning (ML) model in an end-to-end workflow. Particularly, this encompasses data upload, data pre-processing, ML model building and post-model analysis.')

  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, go to the sidebar and 1. Select a data set and 2. Adjust the model parameters by adjusting the various slider widgets. As a result, this would initiate the ML model building process, display the model results as well as allowing users to download the generated models and accompanying data.')

  st.markdown('**Under the hood**')
  st.markdown('Data sets:')
  st.code('''- Drug solubility data set
  ''', language='markdown')
  
  st.markdown('Libraries used:')
  st.code('''- Pandas for data wrangling
- Scikit-learn for building a machine learning model
- Altair for chart creation
- Streamlit for user interface
  ''', language='markdown')


