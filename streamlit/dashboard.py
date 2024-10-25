import pandas as pd
import streamlit as st
import requests
import ast
import numpy as np
import json
from pydantic import create_model
import time
import matplotlib.pyplot as plt




FAST_API = 'http://127.0.0.1:8000'
MIN_SK_ID_CURR = 100001
MAX_SK_ID_CURR = 456255
st.set_page_config(page_title='Credit approval', page_icon='🤞', layout="wide", initial_sidebar_state="auto")


@st.cache_data
def load_input_information():
    """Load the input information CSV and cache it."""
    return pd.read_csv("data/input_information.csv", index_col=0)

input_information = load_input_information()
field_types = input_information['Dtype'].apply(lambda x : eval(f'({x}, ...)')).to_dict()
ModelEntries = create_model('ModelEntries', **field_types)

def main():
    
    # Streamlit tabs
    tab_names = ['Credit Approval', ' Particular decision factors', 'General decision factors']
    credit_approval_tab, local_feature_importance_tab, global_feature_importabce_tab = st.tabs(tab_names)    

    # credit ID form
    row_0 = credit_approval_tab.columns([1, 2])
    row_1 = credit_approval_tab.columns([1])  
    

    if 'sk_id_curr0' in st.session_state:
        sk_id_curr0 = st.session_state['sk_id_curr0']        
    else:
        sk_id_curr0 = np.random.randint(MIN_SK_ID_CURR, MAX_SK_ID_CURR)
        st.session_state['sk_id_curr0'] = sk_id_curr0
    credit_id_form = row_0[0].form("load_credit_form")
    
    sk_id_curr = credit_id_form.number_input(
        label='SK_ID_CURR', 
        min_value=MIN_SK_ID_CURR,                        
        max_value=MAX_SK_ID_CURR,
        value=sk_id_curr0,
        step=1,
        help='ID of loan',
        )
    load_application = credit_id_form.form_submit_button('Load Application')

    # Loan application form
    loan_request_detail_form = credit_approval_tab.form("Loan request Details")
    row_2 = loan_request_detail_form.columns([1,1,1])  
    submit_credit = row_2[1].form_submit_button('Submit Credit Request')
    
    if load_application:        
        loan_request_data_0 = ModelEntries(**get_credit_application(sk_id_curr))
        st.session_state['loan_request_data_0'] = loan_request_data_0    

    if 'loan_request_data_0' in st.session_state:
        loan_request_data_0 = st.session_state['loan_request_data_0']
        loan_submission = load_credit_request_form(loan_request_detail_form, input_information, loan_request_data_0 )
        
    # evaluate credit
    if submit_credit:
        response = request_model(FAST_API, 'validate_client/deployment_v1', loan_submission)
        display_credit_request_response([row_0[1], row_1[0]], response)


@st.cache_data
def load_full_application_data():
    """Load and cache the full application dataset."""
    X_train = pd.read_csv('data/application_train.csv', index_col='SK_ID_CURR').drop(columns=['TARGET'])
    X_test = pd.read_csv('data/application_test.csv', index_col='SK_ID_CURR')
    X_full = pd.concat([X_train, X_test], axis=0).sort_index()    
    return X_full

def display_credit_request_response(container: st.container, response):
    if response.status_code==200:
        prediction = json.loads(response.text)    
        predict_proba = prediction["default_probability"]
        validation_threshold = prediction['validation_threshold']
        show_scoring_gauge(container, predict_proba, validation_threshold)
    else:
        container.error(response.text)

def request_model(model_uri, request, data):
    response = requests.post( f"{model_uri}/{request}/", json=data,)
    return response


def get_credit_application(sk_id_curr: int)-> ModelEntries:
    X_full = load_full_application_data()
    return X_full.loc[sk_id_curr,:].replace({np.nan:None}).to_dict()
    
def load_credit_request_form(form: st.container, inputs: pd.DataFrame, credit_application_0: ModelEntries):
    form_output: dict[str: str|float|int|None] = {}
    credit_application_0 = credit_application_0.dict()
    
    columns = form.columns([1,1,1]) 
    for i, feature in enumerate(inputs.index):
        
        value = credit_application_0[feature]
        if value is None:
            value = 'NA'
                
        if inputs.loc[feature,'Dtype'].__contains__('str'):        
                     
            options = ast.literal_eval(inputs.loc[feature,'categories'].replace('nan', "'NA'"))
            form_output[feature] = columns[1].selectbox(
                label=inputs.loc[feature, 'Column'],
                options=options,
                index=options.index(value),
                help=inputs.loc[feature, 'Description']
                )
        elif inputs.loc[feature,'Dtype'].__contains__('None') :
            form_output[feature] = columns[0].text_input(
                label=inputs.loc[feature, 'Column'],                         
                value=value,
                help=inputs.loc[feature, 'Description']
                ) 
        elif (inputs.loc[feature,'Dtype'].__contains__('int') ) and (inputs.loc[feature,'n_unique']==2):            
            form_output[feature] = columns[2].toggle(            
                label=inputs.loc[feature, 'Column'],
                value=int(value)==1,
                help=inputs.loc[feature, 'Description']
                )    
        elif  (inputs.loc[feature,'Dtype'].__contains__('int') ) :
            form_output[feature] = columns[0].number_input(
                label=inputs.loc[feature, 'Column'],                         
                value=int(value),
                step=1,
                help=inputs.loc[feature, 'Description'],
                )              
        else:
            form_output[feature] = columns[0].number_input(
                label=inputs.loc[feature, 'Column'],                         
                value=float(value),
                format=inputs.loc[feature, 'format'],
                help=inputs.loc[feature, 'Description'],
                )  
        if form_output[feature] == '' or form_output[feature] == 'NA':
            form_output[feature] = None
    return form_output  

def  show_scoring_gauge(container:list[st.container], predict_proba:float, validation_threshold) :
   

    if predict_proba <= validation_threshold:
        container[0].title('Credit approved:')
        container[0].pyplot(create_gauge_figure(1-predict_proba,1-validation_threshold ))
        container[1].success(f"Score {(1-predict_proba)*100:0.1f} $\ge$ Threshold {(1-validation_threshold)*100:0.1f}", icon="✅")
        st.balloons()
    else:
        container[0].title('Credit denied:')        
        container[0].pyplot(create_gauge_figure(1-predict_proba,1-validation_threshold ))
        container[1].error(f"Score {(1-predict_proba)*100:0.1f} < Threshold {(1-validation_threshold)*100:0.1f}", icon="🛑")        
    

def create_gauge_figure(percent_full, needle):

    
    
    fig, ax = plt.subplots(figsize=(15, 0.3))
    ax.set_facecolor('#F0F2F6')
    fig.patch.set_alpha(0)
    # Plot a horizontal bar to simulate a progress bar
    ax.barh([0], [percent_full], color='#1C83E1', height=0.3, label=f'Score: {percent_full:.2%}')
    ax.axvline(x=needle, color='black', linestyle='--', linewidth=1, label=f'Threshold: {needle:.2%}')
   # Add labels for clarity
    ax.set_xlim(0, 1)
    ax.tick_params(left=False, bottom=False)
    ax.set_yticks([])  # Remove y-axis ticks for a cleaner look
    ax.set_xticks(np.linspace(0,1,11))
    ax.set_xticklabels([f'{x:d}' for x in np.linspace(0,100,11, dtype=int)],color=(0.3, 0.3, 0.3))
    ax.set_xlabel('Score', color='gray')

    ax.grid()
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(False)
    return fig
    

if __name__ == '__main__':
    main()
