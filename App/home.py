import streamlit as st
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from utils.functions import preprocess, sentiment_analysis, map_sentiment_score_to_rating


def render_home(model, tokenizer):
    st.title("Movie Review App")
    st.write("Welcome to our Movie Review App powered by the state-of-the-art RoBERTa and TinyBERT models with an impressive accuracy score of 0.93 and 0.83 respectively. Get ready to dive into the world of cinema and discover the sentiments behind your favorite movies. Whether it's a thrilling 9 or a heartwarming 3, our app not only predicts the sentiment but also rates the movie on a scale of 1 to 10. Express your thoughts, press 'Analyze,' and uncover the emotional depth of your movie review")
    st.image("Assets/movie_review.png", caption="", use_column_width=True)
    
    # Create a list to store comments
    comments = []
    
    
    
    # Input text area for the user to enter a review
    input_text = st.text_area("Write your movie review here...")

    # Output area for displaying sentiment
    if st.button("Analyze Review"):
        if input_text:
            # Perform sentiment analysis using the loaded model
            scores = sentiment_analysis(input_text, tokenizer, model)

            # Display sentiment scores
            st.text("Sentiment Scores:")
            for label, score in scores.items():
                st.text(f"{label}: {score:.2f}")

            # Determine the overall sentiment label
            sentiment_label = max(scores, key=scores.get)

            # Map sentiment labels to human-readable forms
            sentiment_mapping = {
                "Negative": "Negative",
                "Positive": "Positive"
            }
            sentiment_readable = sentiment_mapping.get(sentiment_label)

            # Display the sentiment label
            st.text(f"Sentiment: {sentiment_readable}")

            
            rating = map_sentiment_score_to_rating(scores[sentiment_label])

            # Convert the rating to an integer
            rating = int(rating)

            st.text(f"Rating: {rating}")            

    # Button to Clear the input text
    if st.button("Clear Input"):
        input_text = ""

    # Input area for adding comments
    new_comment = st.text_area("Add a comment:", "")
    if st.button("Submit Comment"):
        if new_comment:
            comments.append(new_comment)

        # Display the comments
        st.subheader("Comments")
        for comment in comments:
            st.write(comment)
