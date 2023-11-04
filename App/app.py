   
import streamlit as st
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from utils.functions import preprocess, sentiment_analysis, map_sentiment_score_to_rating
from home import render_home

# Create a sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("👉 **Choose an option from the sidebar**")
page = st.sidebar.radio("Go to", ["Home"])

# Load the models and tokenizers (you can use your model paths)
model_paths = {
    "RoBERTa": "petermutwiri/NLP_Capstone",
    "TinyBERT": "petermutwiri/Tiny_Bert_Cupstone"
}

# Create a dictionary to store models and tokenizers
models = {}
tokenizers = {}

for model_name, model_path in model_paths.items():
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    models[model_name] = model
    tokenizers[model_name] = tokenizer

# Create a dropdown to select the model
selected_model = st.selectbox("Select a model", list(models.keys()))

# Get the model and tokenizer for the selected model
model = models[selected_model]
tokenizer = tokenizers[selected_model]

# Use the selected page function to display the content
if page == "Home":
    render_home(model, tokenizer)
