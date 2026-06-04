import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN

# Load the IMDB dataset
## word index
word_index = imdb.get_word_index()
## loading the data
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn_model.h5')


# Step 2: Helper Functions

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in encoded_review]
    )

# Function to preprocess user input
VOCAB_SIZE = 10000
UNKNOWN_TOKEN = 2

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = []
    for word in words:
        index = word_index.get(word, None)
        if index is None or index >= VOCAB_SIZE:
            encoded_review.append(UNKNOWN_TOKEN + 3)
        else:
            encoded_review.append(index + 3)

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )
    return padded_review

## prediction function
def predict_sentiment(riview):
    processed_input = preprocess_text(riview)
    prediction = model.predict(processed_input)
    
    sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'
    return sentiment, prediction[0][0]

## streamlit app
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go

# Set page config for attractive theme
st.set_page_config(
    page_title="🎬 Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for attractive theme
st.markdown("""
    <style>
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --accent-color: #f093fb;
        --background-color: #0f0c29;
        --text-color: #ffffff;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Header styling */
    .stMarkdown h1, h2, h3 {
        color: #667eea;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Input boxes */
    .stTextArea textarea {
        border: 2px solid #667eea !important;
        border-radius: 10px !important;
        background-color: #1a1a2e !important;
        color: #ffffff !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 30px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
    
    /* Cards/Containers */
    .stContainer {
        background: rgba(26, 26, 46, 0.8);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Metric styling */
    .stMetric {
        background: rgba(26, 26, 46, 0.8);
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# Title with gradient effect
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("## 🎬")
with col2:
    st.markdown("# Sentiment Analysis - RNN Model")

st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Analyze", "📊 Visualization", "ℹ️ About"])

# Tab 1: Analysis
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Enter Your Review")
        st.markdown("*Write a movie review to get sentiment prediction*")
        
        user_review = st.text_area(
            "Movie Review:",
            height=150,
            placeholder="Enter your movie review here...",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### ✨ Example Reviews")
        st.info("**Positive:**\n\nThis movie was absolutely amazing! I loved every minute of it.")
        st.warning("**Negative:**\n\nTerrible plot, bad acting. Complete waste of time.")
    
    st.markdown("---")
    
    # Predict button
    if st.button("🎯 Predict Sentiment", use_container_width=True, key="predict_btn"):
        if user_review.strip():
            with st.spinner("🔄 Analyzing..."):
                sentiment, confidence = predict_sentiment(user_review)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if sentiment == "Positive":
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                padding: 20px; border-radius: 10px; text-align: center;">
                        <h3 style="color: white; margin: 0;">😊 Sentiment</h3>
                        <h2 style="color: #00ff00; margin: 10px 0; font-size: 2em;">POSITIVE</h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%);
                                padding: 20px; border-radius: 10px; text-align: center;">
                        <h3 style="color: white; margin: 0;">😞 Sentiment</h3>
                        <h2 style="color: #ff4444; margin: 10px 0; font-size: 2em;">NEGATIVE</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 20px; border-radius: 10px; text-align: center;">
                    <h3 style="color: white; margin: 0;">🎯 Confidence</h3>
                    <h2 style="color: #ffd700; margin: 10px 0; font-size: 2em;">{confidence*100:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                prob_negative = (1 - confidence) * 100
                prob_positive = confidence * 100
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Positive', 'Negative'],
                        y=[prob_positive, prob_negative],
                        marker=dict(
                            color=['#00ff00', '#ff4444'],
                            opacity=0.8
                        ),
                        text=[f'{prob_positive:.1f}%', f'{prob_negative:.1f}%'],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Probability Distribution",
                    xaxis_title="Sentiment",
                    yaxis_title="Probability (%)",
                    hovermode='x unified',
                    plot_bgcolor='rgba(26, 26, 46, 0.8)',
                    paper_bgcolor='rgba(15, 12, 41, 0)',
                    font=dict(color='#ffffff'),
                    height=350,
                    showlegend=False,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Display original review
            st.markdown("---")
            st.markdown("### 📖 Your Review")
            st.info(user_review)
        else:
            st.error("⚠️ Please enter a review to analyze!")

# Tab 2: Visualization
with tab2:
    st.markdown("### 📊 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🤖 Model Type",
            value="Simple RNN",
            delta="Recurrent Neural Network"
        )
    
    with col2:
        st.metric(
            label="📚 Dataset",
            value="IMDB",
            delta="25,000 samples"
        )
    
    with col3:
        st.metric(
            label="🎯 Task",
            value="Sentiment",
            delta="Binary Classification"
        )
    
    with col4:
        st.metric(
            label="🔤 Max Length",
            value="500",
            delta="tokens"
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏗️ Model Architecture")
        st.code("""
Embedding Layer (128 dims)
    ↓
SimpleRNN Layer (32 units)
    ↓
Dense Layer (1 unit)
    ↓
Sigmoid Activation
        """, language="text")
    
    with col2:
        st.markdown("### ⚙️ Model Configuration")
        config_text = """
**Optimizer:** Adam
**Loss:** Binary Crossentropy
**Metrics:** Accuracy
**Batch Size:** 32
**Epochs:** 5
**Embedding Dim:** 128
**RNN Units:** 32
        """
        st.markdown(config_text)

# Tab 3: About
with tab3:
    st.markdown("### ℹ️ About This Application")
    st.markdown("""
    This is a **Sentiment Analysis** application powered by a **Simple RNN** neural network model.
    
    #### 🎯 Features:
    - **Real-time Sentiment Analysis** - Get instant predictions on movie reviews
    - **Confidence Scores** - See how confident the model is about its predictions
    - **Visual Analytics** - Beautiful charts and metrics
    - **Pre-trained Model** - Uses IMDB dataset trained model
    
    #### 🔍 How It Works:
    1. You enter a movie review
    2. The review is preprocessed and tokenized using the IMDB word index
    3. Text is padded to a fixed length (500 tokens)
    4. The RNN model processes the sequence
    5. Output is a probability score (0-1)
    6. Score ≥ 0.5 → Positive Review, < 0.5 → Negative Review
    
    #### 📊 Model Details:
    - **Model Type:** Simple RNN (Recurrent Neural Network)
    - **Dataset:** IMDB Movie Reviews
    - **Classes:** Positive & Negative
    - **Max Review Length:** 500 words
    
    #### 👨‍💻 Built with:
    - TensorFlow/Keras
    - Streamlit
    - Plotly
    - Python
    """)
    
    st.markdown("---")
    st.info("✨ Enjoy analyzing movie reviews with AI!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #667eea; margin-top: 20px;">
    <p>🔬 Sentiment Analysis RNN | Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
