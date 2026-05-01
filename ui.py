import streamlit as st
from src.models.predict import Predictor


@st.cache_resource
def load_predictor():
    return Predictor(
        model_path="artifacts/sentiment_model.keras",
        tokenizer_path="artifacts/tokenizer.pkl",
        max_len_path="artifacts/max_len.pkl",
        label_path="artifacts/label_classes.pkl"
    )


predictor = load_predictor()


st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="📊",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align: center;'>Sentiment Analyzer</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: grey;'>Analyze text sentiment using a deep learning model.</p>",
    unsafe_allow_html=True
)

st.markdown("""
### Project Overview
- Model: Artificial Neural Network (ANN) with Embedding Layer  
- Task: Multi-class Sentiment Classification  
- Classes: Positive, Negative, Neutral, Irrelevant  
""")


st.markdown("---")

user_input = st.text_area("Enter text to analyze", height=150)

predict_clicked = st.button("Analyze Sentiment")

if predict_clicked:

    if user_input.strip() == "":
        st.warning("Please enter text to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            label, confidence, probs = predictor.predict(user_input)

        st.metric("Predicted Sentiment", label)
        st.metric("Confidence", f"{confidence:.2f}%")

        st.bar_chart(probs)

        st.markdown("---")

        # Prediction Card
        st.markdown("### Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Predicted Sentiment", value=label)

        with col2:
            st.metric(label="Confidence", value=f"{confidence:.2f}%")

        st.progress(int(confidence))

    
        if confidence >= 80:
            st.success("High confidence prediction.")
        elif confidence >= 60:
            st.info("Moderate confidence prediction.")
        else:
            st.warning("Low confidence. The input may be ambiguous.")

    
        st.markdown("### Interpretation")

        if label.lower() == "positive":
            st.write("The text expresses satisfaction, praise, or positive emotion.")
        elif label.lower() == "negative":
            st.write("The text expresses dissatisfaction, criticism, or negative emotion.")
        elif label.lower() == "neutral":
            st.write("The text does not contain strong emotional signals.")
        else:
            st.write("The text may not be directly relevant to the defined sentiment categories.")


        st.markdown("---")


        st.markdown("## Model Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric("Training Accuracy", "96.5%")
        col2.metric("Validation Accuracy", "96.1%")
        col3.metric("Validation Loss", "0.19")

        st.markdown(
                """
                    **Generalization Insight:**  
                    The small gap between training accuracy (96.5%) and validation accuracy (96.1%) 
                    indicates that the model generalizes well and does not significantly overfit.
                """
        )

        st.caption(
                "Metrics were calculated on a held-out validation dataset during training."
        )


        st.markdown("---")
        
        with st.expander("Model Details & Technical Explanation"):
            st.write("""
                **Architecture**
                - Embedding Layer
                - Dense Hidden Layers
                - Softmax Output Layer

                **Preprocessing**
                    - Lowercasing
                    - Punctuation removal
                    - Tokenization
                    - Sequence padding

                **Training**
                    - Loss Function: Categorical Crossentropy
                    - Optimizer: Adam
                    - Evaluation Metrics: Accuracy

                The model outputs probability scores for each sentiment class.
                The highest probability determines the final classification.
            """)