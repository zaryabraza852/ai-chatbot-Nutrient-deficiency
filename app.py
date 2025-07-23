import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
# Set page configuration
st.set_page_config(layout="wide")

# Initialize session state for chat history and detection results
if "messages" not in st.session_state:
    st.session_state.messages = []
if "detection_results" not in st.session_state:
    st.session_state.detection_results = None
if "detected_class" not in st.session_state:
    st.session_state.detected_class = None
if "annotated_image" not in st.session_state:
    st.session_state.annotated_image = None

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO('/home/ikram-ali/Downloads/xavor-work/rice_app/best.pt')

# Initialize LangChain components
@st.cache_resource
def initialize_chat():
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
    )
    
    memory = ConversationBufferMemory()
    
    template = """You are an agricultural expert specializing in rice crop nutrient deficiencies. 
    Provide brief, accurate information about nutrient deficiencies, their causes, symptoms, and treatments.
    Current conversation:
    {history}
    Human: {input}
    Assistant:"""
    
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
    )
    
    return conversation

# Main app title
st.title("🌾 Rice Crop Nutrient Deficiency Detector")

# Create two columns
col1, col2 = st.columns(2)

# Left column for image upload and detection
with col1:
    st.header("🧪 Image Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a rice leaf image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Only perform detection if we don't have results or if it's a new image
        if (st.session_state.detection_results is None or 
            st.session_state.uploaded_file != uploaded_file):
            
            # Store the current uploaded file
            st.session_state.uploaded_file = uploaded_file
            
            # Read and display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image")
            
            # Convert PIL image to numpy array for YOLO
            image_np = np.array(image)
            
            # Load model and perform detection
            model = load_model()
            results = model(image_np)
            
            # Store detection results in session state
            st.session_state.detection_results = results[0]
            st.session_state.detected_class = results[0].names[int(results[0].probs.top1)]
            st.session_state.annotated_image = results[0].plot()
            
            # Reset chat state for new detection
            st.session_state.messages = []
            st.session_state.initial_message = False
            if "conversation" in st.session_state:
                del st.session_state.conversation
        
        # Display stored results
        if st.session_state.detection_results is not None:
            st.image(st.session_state.annotated_image, caption="Detected Deficiency")
            st.subheader(f"Detected Deficiency: {st.session_state.detected_class}")
            
            # Initialize chat if not already done
            if "conversation" not in st.session_state:
                st.session_state.conversation = initialize_chat()
            
            # Generate initial bot message about the deficiency
            if not st.session_state.get("initial_message", False):
                initial_message = st.session_state.conversation.predict(
                    input=f"Explain the {st.session_state.detected_class} deficiency in rice crops, including its causes, symptoms, and treatment methods."
                )
                st.session_state.messages.append({"role": "assistant", "content": initial_message})
                st.session_state.initial_message = True

# Right column for chatbot
with col2:
    st.header("🤖 AgriBot Assistant")
    
    # Create a container for chat messages
    chat_container = st.container()
    
    # Display chat messages in the container
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input - this will automatically stay at the bottom
    if "conversation" in st.session_state:
        if prompt := st.chat_input("Ask about the deficiency..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get bot response
            response = st.session_state.conversation.predict(input=prompt)
            
            # Add bot response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display bot response
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Rerun to update the chat display
            st.rerun() 