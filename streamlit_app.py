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
import io
import xlsxwriter

# Page title
st.set_page_config(page_title='ML Model Building', page_icon='🤖')
st.title('🤖 Sage comme Platon')
uploaded_file = st.file_uploader("Upload an Excel file",type=["xlsx"], key = 45)
if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, index_col=False)
        
        # Replace NaN values with None to match JSON format expectations
        df = df.where(pd.notnull(df), None)

        # Convert the DataFrame to a list of dictionaries (JSON format)
        json_data = df.to_dict(orient='records')

        # Convert the list of dictionaries to JSON string with escaped double quotes
        input_data = json.dumps(json_data, ensure_ascii=False)

        # Print the JSON string
        print(input_data)

        os.environ["OPENAI_API_KEY"] = st.secrets["API_KEY"]
        
        
        client = OpenAI()

        response = client.chat.completions.create(
          model="ft:gpt-3.5-turbo-0125:personal:sage-comme-platon:9UcSNmSh",
          messages=[
            {
              "role": "system",
              "content": [
                {
                  "type": "text",
                  "text": "I will give you a structured JSON with waste management metrics. I need you to add a libelle. Choose among this list of libelle : Total déchets MOA, OM MOA x - flux de déchets réceptionnés UVE, TVI-DID MOA x - flux de déchets réceptionnés UVE, Total Refus CS MOA - flux de déchets réceptionnés UVE, Total OM Apporteurs tiers - flux de déchets réceptionnés UVE, DAE-DIB Apporteurs tiers x - flux de déchets réceptionnés UVE, Total DASRI Apporteurs tiers - flux de déchets réceptionnés UVE, Total Boues MOA - flux de déchets réceptionnés UVE, Total déchets réceptionnés, Part déchets MOA, Total déchets assimilables HPCI, Part déchets HPCI. Also add the probability that the libelle you attached is right. "
                }
              ]
            },
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": input_data
                }
              ]
            }
          ],
          temperature=0.51,
          max_tokens=2048,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
        response = response.choices[0].message.content
        print(response)

                # Replace NaN with null
        data_string = re.sub(r'\bNaN\b', 'null', response)

        # Unescape the string
        data_string = data_string.replace('\\"', '"')

        # Convert the string to JSON
        data_json = json.loads(data_string)

        # Print the JSON to verify
        print(json.dumps(data_json,ensure_ascii=False, indent=4))
        df_final = pd.DataFrame(data_json)
        output = io.BytesIO()
        df_final.to_excel(output, index=False, engine='xlsxwriter')
        
        # Download button
        
        st.download_button(
            label="Download Processed Excel",
            data=output,
            file_name='processed_file.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )






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


