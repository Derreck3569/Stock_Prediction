import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer


from sklearn.pipeline import Pipeline
import shap

from joblib import dump
from joblib import load



# Setup & Path Configuration
warnings.simplefilter("ignore")

# Fix path for Streamlit Cloud (ensure 'src' is findable)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

#from src.feature_utils import extract_features
#from src.Custom_Classes import DropHighMissingCols, TransactionFeatureEngineer, DropHighCorrelation

file_path = os.path.join(project_root, 'Poftfolio/X_train.csv')

dataset = pd.read_csv(file_path)
# ── FIX 1: keep robust Unnamed-column drop (your original 'Unnamed: 0' drop
#           crashes if the column isn't there) ──────────────────────────────
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

# Access the secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# AWS Session Management
@st.cache_resource # Use this to avoid downloading the file every time the page refreshes
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# Data & Model Configuration

MODEL_INFO = {
    "endpoint"  : aws_endpoint,
    # ── FIX 2: explainer filename matches what notebook uploads to S3 ────────
    "explainer" : "explainer_fraud.shap",
    # ── FIX 3: pipeline tar filename matches what notebook uploads to S3 ─────
    "pipeline"  : "fine_tuned_pipeline.tar.gz",
    "keys"      : ['V292','V291','id_35','V295'],
    "inputs"    : [{"name": k, "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01} for k in ['V292','V291','id_35','V295']]
}


def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename=MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key= f"{key}/{os.path.basename(filename)}")
        # Extract the .joblib file from the .tar.gz
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        # ── FIX 4: tar contains .pkl (notebook bundles finalized_fraud_model.pkl)
        joblib_file = [f for f in tar.getnames() if f.endswith('.pkl')][0]


    # Load the full pipeline
    return joblib.load(f"{joblib_file}")

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    local_path = local_path

    # Only download if it doesn't exist locally to save time
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)

    with open(local_path, "rb") as f:
        return load(f)
        #return shap.Explainer.load(f)

# Prediction Logic
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )
    
    try:
        if isinstance(input_df, dict):
            input_df = pd.DataFrame(input_df)
        # Pass a Python list-of-dicts; JSONSerializer handles serialisation
        payload = input_df.to_dict(orient="records")
        
        raw_pred = predictor.predict(payload)

        # ── FIX 7: inference.py now returns {"prediction":[...], "probability":[...]}
        pred_val = raw_pred["prediction"][0]
        prob     = raw_pred["probability"][0]
        mapping  = {0: "Legitimate", 1: "Fraud"}
        return mapping.get(pred_val), prob, 200
    except Exception as e:
        return f"Error: {str(e)}", None, 500

# Local Explainability
def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(session, aws_bucket, posixpath.join('fraud-detection', explainer_name),os.path.join(tempfile.gettempdir(), explainer_name))

    best_pipeline = load_pipeline(session, aws_bucket, 'fraud-detection/sklearn-pipeline-deployment')

    # ── FIX 8: walk the fraud pipeline's actual preprocessing steps by name.
    #          Original sliced steps[:-2] which only worked for the loan
    #          pipeline. Fraud pipeline steps:
    #          time_fe → high_missing → low_var → preprocessor → kbest → sampler → model
    if isinstance(input_df, dict):
        input_df = pd.DataFrame(input_df)

    pre_steps = ['time_fe', 'high_missing', 'low_var', 'preprocessor', 'kbest']
    Xt = input_df.copy()
    for step in pre_steps:
        if step in best_pipeline.named_steps:
            Xt = best_pipeline.named_steps[step].transform(Xt)

    # ── FIX 9: use 'kbest' (the actual feature-selection step name) ──────────
    feature_names = best_pipeline.named_steps['kbest'].get_feature_names_out()
    input_df_transformed = pd.DataFrame(Xt, columns=feature_names)

    shap_values = explainer(input_df_transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 1], show=False)  # class 1 = fraud
    st.pyplot(fig)
    plt.close(fig)
    top_feature = pd.Series(shap_values[0, :, 1].values, index=shap_values[0, :, 1].feature_names).abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")


# Streamlit UI
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.form("pred_form"):
    st.subheader(f"Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'], max_value=inp['max'], value=inp['default'], step=inp['step']
            )

    submitted = st.form_submit_button("Run Prediction")

# ── FIX 10: build a proper one-row DataFrame from the X_train sample,
#           overwrite with user inputs. Original used to_dict() on a 1-row
#           DataFrame which produced {col: {0: val}} (broken shape).
original = dataset.iloc[0:1].copy()
for k, v in user_inputs.items():
    if k in original.columns:
        original[k] = v

if submitted:

    # ── FIX 11: unpack 3-tuple now (label, probability, status) ──────────────
    res, prob, status = call_model_api(original)
    if status == 200:
        col1, col2 = st.columns(2)
        col1.metric("Prediction Result", res)
        col2.metric("Fraud Probability", f"{prob:.2%}")
        display_explanation(original, session, aws_bucket)
    else:
        st.error(res)
