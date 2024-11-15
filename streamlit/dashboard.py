import os
from typing import Callable, Any
import pandas as pd
import streamlit as st
import requests
import ast
import numpy as np
import json
from pydantic import create_model
import matplotlib.pyplot as plt
import seaborn as sns
from shap.plots import waterfall, beeswarm
from shap import Explanation
from streamlit_shap import st_shap
from projet07.model_evaluation import plot_feature_importances
from dotenv import load_dotenv

load_dotenv()
FAST_API = 'http://127.0.0.1:8000'
FAST_API = os.getenv('FAST_API')

MIN_SK_ID_CURR = int(os.getenv('MIN_SK_ID_CURR')) 
MAX_SK_ID_CURR = int(os.getenv('MAX_SK_ID_CURR')) 

input_information = "data/input_information.csv"

st.set_page_config(page_title='Credit approval', page_icon='🤞', layout="wide", initial_sidebar_state="collapsed")

sns.set_theme()

@st.cache_data
def load_input_information():
    """Load the input information CSV and cache it."""
    return pd.read_csv(input_information, index_col=0)

input_information = load_input_information()
field_types = input_information['Dtype'].apply(lambda x : eval(f'({x}, ...)')).to_dict()
ModelEntries = create_model('ModelEntries', **field_types)

def main():
    
    # check available models
    available_models = get_model_names(FAST_API)
    model_name_v = st.sidebar.selectbox('model', available_models)
    
    # Streamlit tabs
    tab_names = ['Credit Approval', ' Particular decision factors', 'General decision factors']
    credit_approval_tab, local_feature_importance_tab, global_feature_importance_tab = st.tabs(tab_names)    

    # credit ID form
    row_0 = credit_approval_tab.columns([1, 2])
    row_1 = credit_approval_tab.columns([1])  
    
    if 'sk_id_curr0' in st.session_state:
        sk_id_curr0 = st.session_state['sk_id_curr0']        
    else:
        sk_id_curr0 = np.random.randint(MIN_SK_ID_CURR, MAX_SK_ID_CURR)
        st.session_state['sk_id_curr0'] = sk_id_curr0
    if 'credit_analized' not in st.session_state:
        st.session_state['credit_analized'] = False
        
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
        st.session_state['credit_analized'] = False

    if 'loan_request_data_0' in st.session_state:
        loan_request_data_0 = st.session_state['loan_request_data_0']
        loan_submission = load_credit_request_form(loan_request_detail_form, input_information, loan_request_data_0 )
    

    # Local feature importance form
    get_local_explanation, n_features_to_show = load_feature_form(local_feature_importance_tab,'local')

    # # Local feature importance form
    get_global_explanation, n_global_features_to_show, cum_imp_cut = load_feature_form(global_feature_importance_tab, 'global')


    # evaluate credit
    if submit_credit or st.session_state['credit_analized']:
        response = post_request_model(FAST_API, f'{model_name_v}/validate_client', loan_submission)
        if response is not None:
            display_credit_request_response([row_0[1], row_1[0]], response)
            st.session_state['credit_analized'] = True
            
    # get_local_explanation
    if get_local_explanation:
        local_shap_plots(local_feature_importance_tab, loan_submission, model_name_v, n_features_to_show)

    # get_local_explanation
    if get_global_explanation:
        global_features_plots(global_feature_importance_tab, model_name_v, n_global_features_to_show, cum_imp_cut)


def global_features_plots(container: st.container, model_name_v:str, n_features_to_show:int, cum_imp_cut:float): # type: ignore
    
    global_shaps = post_request_model(FAST_API, f'{model_name_v}/shap_value_attributes', None)
    global_importance_dict = post_query_model(FAST_API, f'{model_name_v}/get_global_feature_importance', {"cum_importance_cut": cum_imp_cut})
    feature_importances_domain=pd.DataFrame(global_importance_dict['feature_importances_domain']).reset_index().drop(columns=['index'], inplace=False)
    most_important_features = global_importance_dict['most_important_features']
    
    if global_shaps is not None:
        
        global_shap_values = get_shap_values(global_shaps)

        #Inversion if values sign to signify approval score instead of default prediction
        global_shap_values.values = -global_shap_values.values
        global_shap_values.base_values = -global_shap_values.base_values

        n_features = global_shap_values.shape[1]
        st.session_state['n_features'] = n_features
        n_features_to_show = min(n_features, n_features_to_show )

        with container:            
            fig, axes = plt.subplots(1, 2, figsize=(n_features_to_show*0.5, 30))
            ax = plt.subplot(1, 2, 1)      
            plot_feature_importances(feature_importances_domain, most_important_features, n_features_to_show, ax)
      
            ax = plt.subplot(1, 2, 2)    
            shap_order = global_shap_values.abs.mean(0)
            feat_order = feature_importances_domain.sort_index(ascending=False).reset_index()[['feature']].reset_index().set_index('feature')
            feat_order = feat_order
            shap_order.values = feat_order.loc[global_shap_values.feature_names,'index'].values            
            beeswarm(global_shap_values, max_display=n_features_to_show, order=shap_order, s=2)
            ax.set_yticklabels(['']*n_features_to_show)
            ax.set_ylim([-1, n_features_to_show])
            ax.set_xticklabels([])            
            ax.set_xlabel('Feature effect', fontsize=8)
            plt.tight_layout()            
            container.pyplot(fig, use_container_width=True)

def load_feature_form(container, feature_type:str):
    feat_importance_row_0 = container.columns([1,1,1])
    n_features_to_show_form = feat_importance_row_0[1].form(f'Number of {feature_type} features to show features to show')
    cum_imp_cut = None
    with n_features_to_show_form:
        if feature_type=='local':
            columns = st.columns([3,1], vertical_alignment='bottom')
        elif feature_type=='global':
            columns = st.columns([3,3,2], vertical_alignment='bottom')   
            cum_imp_cut = columns[1].number_input("Cumulative importance cut:",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.95)     
        else:
            raise('feature_type not handled')

        if 'n_features' in st.session_state:
            n_features = st.session_state['n_features']
        else:
            n_features = None

        n_features_to_show = columns[0].number_input("Number of explaining factors to display:",
                        min_value=2,
                        max_value=n_features,
                        value= 10)
        get_explanation = columns[-1].form_submit_button(f'Show {feature_type} feature explanation')


        if cum_imp_cut is None:
            return get_explanation, n_features_to_show
        else:
            return get_explanation, n_features_to_show, cum_imp_cut
    


def get_shap_values(explanation_attrs: dict[str, Any])->Explanation:
    """Reconstitutes shap values from the corresponding attributes transfered.
    Args:
        explanation_attrs (list[dict[str, Any]]): attributes to reconstitute and exmplanation

    Returns:
        shap_values
    """

    recons_attrs = {attr : (np.array(value) if isinstance(value, list) else value)
                         for attr, value in explanation_attrs.items() 
                   }
  
    return Explanation(**recons_attrs)

def local_shap_plots(container: st.container, loan_submission, model_name_v, n_features_to_show): # type: ignore
    
    local_shaps = post_request_model(FAST_API, f'{model_name_v}/shap_value_attributes', loan_submission)
    global_shaps = post_request_model(FAST_API, f'{model_name_v}/shap_value_attributes', None)

    if local_shaps is not None:
        shap_values = get_shap_values(local_shaps)
        global_shap_values = get_shap_values(global_shaps)

        #Inversion if values sign to signify approval score instead of default prediction
        shap_values.values = -shap_values.values
        shap_values.base_values = -shap_values.base_values        
        
        global_shap_values.values = -global_shap_values.values
        global_shap_values.base_values = -global_shap_values.base_values

        n_features = global_shap_values.shape[1]
        st.session_state['n_features'] = n_features
        n_features_to_show = min(n_features, n_features_to_show )
        with container:            
            fig, axes = plt.subplots(1, 2, figsize=(n_features_to_show, 8))
            ax = plt.subplot(1, 2, 1)                
            waterfall(shap_values[0], max_display=n_features_to_show, show=False)
            ax.set_ylim([-1, n_features_to_show])

            ax = plt.subplot(1, 2, 2)    
            beeswarm(global_shap_values, max_display=n_features_to_show, order=shap_values.abs.mean(0), s=2)
            ax.set_yticklabels([])
            ax.set_ylim([-1, n_features_to_show])
            
            st_shap( fig, height=n_features_to_show*50 , width=1500)
            
            
    
def process_server_response_decorator(func:Callable):
    def modified_function(*args, **kwargs):
        try:
            response = func(*args, **kwargs)
            if response.status_code==200:
                content = json.loads(response.text)    
                return content           
            else:
                show_server_error(response)
        except requests.ConnectionError as e:
            show_exception_error(e)
        
    return modified_function
        

@st.dialog("Server Error")
def show_server_error(response):
    error_codes={422: 'Input Validation Error',
                 404: 'Requested url unavailable'}
    
    status_code = response.status_code 
    if status_code in error_codes.keys():
        st.write(f'ERROR: {status_code} : {error_codes[status_code]}')
    else:
        st.write(f'ERROR: {status_code} : detail unknown')
    if st.button('OK'):
        st.rerun()


@st.dialog("Server Error")
def show_exception_error(e:Exception):
    st.write(f'ERROR: {e.__dict__} : detail unknown')
    if st.button('OK'):
        st.rerun()

@process_server_response_decorator
def post_request_model(api_uri, request, data):
    headers = {"Content-Type": "application/json"}    
    response = requests.post( f"{api_uri}/{request}/", json=data, headers=headers)
    return response

@process_server_response_decorator
def post_query_model(api_uri, request, param):
    headers = {"Content-Type": "application/json"}    
    response = requests.post( f"{api_uri}/{request}/", params=param, headers=headers)
    return response


@process_server_response_decorator
def get_model_names(api_uri):
    headers = {"Content-Type": "application/json"}
    response = requests.get( f"{api_uri}/available_model_name_version/", headers=headers)
    return response

def get_credit_application(sk_id_curr: int)-> ModelEntries: # type: ignore
    input_data = post_query_model(FAST_API,'/get_credit_application', {'SK_ID_CURR': sk_id_curr})    
    print(input_data)
    return input_data
    # return X_full.loc[sk_id_curr,:].replace({np.nan:None}).to_dict()
    
def load_credit_request_form(form: st.container, inputs: pd.DataFrame, credit_application_0: ModelEntries): # type: ignore
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

def  display_credit_request_response(container: st.container, prediction:dict[str, bool|list[bool]]): # type: ignore
    predict_proba = prediction["default_probability"][0]
    validation_threshold = prediction['validation_threshold']   

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
