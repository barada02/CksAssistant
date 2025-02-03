import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="Chat with AI",
    page_icon="🤖",
    layout="centered"
)

# Initialize the model
@st.cache_resource
def load_model():
    """
    Load the language model using Hugging Face's pipeline.
    This function is cached to prevent reloading on every run.
    """
    return pipeline(
        "text-generation",
        model="facebook/opt-125m",  # A small model suitable for demonstrations
        torch_dtype="auto",
        device="cpu"
    )

# Custom CSS for better appearance
st.markdown("""
    <style>
    .stChat {
        padding: 10px;
    }
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# App title
st.title("💬 Chat with AI")
st.caption("A simple chat interface using Streamlit and Hugging Face")

# Initialize model
model = load_model()

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Generate response using the model
            response = model(
                prompt,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
            )[0]["generated_text"]
            
            # Clean up the response by removing the input prompt
            response = response.replace(prompt, "").strip()
            
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a sidebar with information
with st.sidebar:
    st.title("About")
    st.markdown("""
    This is a simple chat application built with:
    - Streamlit
    - Hugging Face Transformers
    - Facebook's OPT-125M model
    
    The model used is a smaller version suitable for demonstrations.
    For production use, consider using more powerful models.
    """)
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
