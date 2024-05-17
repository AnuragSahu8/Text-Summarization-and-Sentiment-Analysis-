1. first create an environment for python. 
2. install the necassary library given in the requirements.txt file  
    use pip cmd (pip install -r requirements.txt).

3. Run the Streamlit app by writting the streamlit run main.py cmd in the terminal.
4. paste your article,text  in the text area.

I used Transformer model for this task because these models are easy to use and having the higher accuracy perform well for the text data.

Model used :
    Summarization ("facebook/bart-large-cnn")
    Sentiment Analysis ("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")


Following are the steps required during the taks -:

1. use pipeline for using the transformer model.
2. converted the text into tokens passed it to the model get the output ids.
3. generated the summarized text for the sentence.
4. passed it to the sentiment model for  Sentiment analysis.
5. predict the Sentiment label.

Application Link : https://c2plpvn3svonj2guygdy8l.streamlit.app/
