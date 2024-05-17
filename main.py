import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,pipeline
from scipy.special import softmax
import streamlit as st

# Load pipelines
@st.cache_resource
def load_summarization_pipeline():
    return pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn", framework="pt")

summarize_pipe = load_summarization_pipeline()

# summarized text generation
def summarize_text(article,chunk_size=2048, summary_length=80, overlap=100):
    # Initialize an empty list to store the summarized chunks
    summarized_chunks = []

    # Split the article into chunks
    chunks = [article[i:i+chunk_size] for i in range(0, len(article), chunk_size)]

    # Summarize each chunk
    for chunk in chunks:
        # Generate summary for the current chunk
        summary = summarize_pipe(chunk, max_length=summary_length, min_length=20, do_sample=False)[0]['summary_text']
        summarized_chunks.append(summary)
    
    # Concatenate the summarized chunks to form the final summary
    final_summary = ' '.join(summarized_chunks)
    return final_summary
    

#
@st.cache_resource 
def load_model_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


def get_sentiment_score(text, model, tokenizer):
    encoded_text = tokenizer(text, return_tensors="pt",)
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        "Negative": scores[0], 
        "Neutral": scores[1], 
        "Positive": scores[2]}
    
    return scores_dict

def get_sentiment_label(scores_dict):
    return max(scores_dict, key=scores_dict.get)

# Load data
# Streamlit code
# Main Streamlit app

st.title("Text Summarization and Sentiment Analysis")

# Text input
text = st.text_area("Enter your Text to summarize and analyze:")

if st.button("Generate Summary"):
    # Generate and display summary
    summary_output = summarize_text(text)
    st.session_state.summary_output = summary_output

if "summary_output" in st.session_state:
    st.subheader("Summary")
    st.write(st.session_state.summary_output)

    if st.button("Analyze"):
        # Load sentiment classification model and tokenizer
        tokenizer, sentiment_model = load_model_tokenizer("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

        # Get sentiment score
        polarity_scores = get_sentiment_score(st.session_state.summary_output, sentiment_model, tokenizer)

        # Get sentiment label
        sentiment_label = get_sentiment_label(polarity_scores)

        # Display sentiment analysis result
        st.subheader("Sentiment Analysis")
        st.markdown(f"""
        **Polarity Scores:**
        - Negative: {polarity_scores["Negative"]}
        - Neutral: {polarity_scores["Neutral"]}
        - Positive: {polarity_scores["Positive"]}
        """)
        
        st.markdown(f"**Sentiment Label:** {sentiment_label}")

