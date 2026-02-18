from groq import Groq
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GROQ_API_KEY')
client = Groq(api_key=api_key)

device = torch.device("cpu")

@st.cache_resource
def load_leaf_model():
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 10)

    if os.path.exists("best_model.pth"):
        checkpoint = torch.load("best_model.pth", map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    else:
        st.error("Model file 'best_model.pth' not found!")
        return None

leaf_model = load_leaf_model()

class_names = {
    0: 'Tomato_Bacterial_spot',
    1: 'Tomato_Early_blight',
    2: 'Tomato_Late_blight',
    3: 'Tomato_Leaf_Mold',
    4: 'Tomato_Septoria_leaf_spot',
    5: 'Tomato_Spider_mites_Two_spotted_spider_mite',
    6: 'Tomato__Target_Spot',
    7: 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    8: 'Tomato__Tomato_mosaic_virus',
    9: 'Tomato_healthy'
}

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

pages = st.sidebar.radio("Select the pages for navigation:",[
    "Introduction", 'Analysis and Prediction', 'Approach'
])

if pages == 'Introduction':
    st.title("🌿 Tomato Leaf Disease Classifier")
    st.write("Developed with a robust stack including Python, Pandas, and Streamlit, this platform utilizes Deep Learning to diagnose plant diseases. Beyond simple identification, the system integrates Generative AI (GPT) to provide detailed disease descriptions, symptom analysis, and actionable prevention strategies.")
    if st.button("Click to find out the number of classes"):
        st.write("The total number of classes is: 10")
        st.markdown("""
                    * Tomato Bacterial spot
            * Tomato Early blight
            * Tomato Late blight
            * Tomato Leaf Mold
            * Tomato Septoria leaf spot
            * Tomato Spider mites Two spotted spider mite
            * Tomato Target Spot
            * Tomato Tomato YellowLeaf Curl Virus
            * Tomato Tomato mosaic virus
            * Tomato healthy
                    """
                    )
        
elif pages == 'Analysis and Prediction':
    st.header("Analysis and Prediction")
    st.write("Upload an image of a tomato leaf to identify the disease.")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"]
    )

    def leaf_diseases_info(label):
        response = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[
                {
                    'role': 'system',
                    'content': (
                        "You are a professional plant pathologist. "
                        "A deep learning model has identified a leaf disease "
                        "in a crop. Provide a concise, expert report for a farmer."
                    )
                },
                {
                    'role': 'user',
                    'content': f"""
    Explain the following leaf disease with:
    1. Description
    2. Common symptoms
    3. Preventive or early-detection measures

    Leaf Disease: {label}
    """
                }
            ],
            temperature=0.6,
            max_tokens=1000
        )
        return response.choices[0].message.content

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)

        if st.button('Identify Image'):
            if leaf_model is not None:
                with st.spinner('Analyzing...'):
                    img_tensor = preprocess(image).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = leaf_model(img_tensor)
                        probabilities = torch.nn.functional.softmax(output[0], dim=0)
                        confidence, index = torch.max(probabilities, dim=0)

                    label = class_names[index.item()]
                    percent = confidence.item() * 100

                    st.subheader("Results:")
                    st.success(f"**Prediction:** {label}")
                    st.info(f"**Confidence:** {percent:.2f}%")
                    st.progress(confidence.item())
            else:
                st.warning(
                    "Model is not loaded. Please check if 'best_model.pth' is in the project folder."
                )

            if leaf_model is not None and 'label' in locals():
                # if st.button("Get the suggestion from the AI"):
                    with st.spinner("Generating expert advice..."):
                        advice = leaf_diseases_info(label)
                        st.subheader("🌱 Expert Recommendation")
                        st.write(advice)

elif pages == 'Approach':
    st.header("Approach")
    approach = st.radio(
                            "Select Project Phase:", 
                            [
                                "Data Preparation", 
                                "EDA", 
                                "Data Augmentation", 
                                "Model Selection", 
                                "Training Pipeline", 
                                "Evaluation Metrics", 
                                "Generative AI Integration", 
                                "Deployment"
                            ]           
                        )
    if approach == "Data Preparation":
        st.header("Data Preparation")
        st.markdown("""
                * Download and extract the dataset
                * Organize images into train/ and test/ folders
                * Resize images and normalize pixel values
                   """)
        
    elif approach == 'EDA':
        st.write("Done in the IPYNB file")

    elif approach == 'Data Augmentation':
            st.header("Data Augmentation")
            st.markdown("""
                        * RandomResizedCrop(224, scale=(0.8, 1.0))
                        * RandomHorizontalFlip(p=0.5)
                        * ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)
                        * Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])""")
            
    elif approach == "Model Selection":
        st.header("Model Selection")
        st.markdown("""
                    * **Model Used - Mobilenet - V2**
                    * Replace the Final classification layer
                    * Fine-tune model on the disease dataset """)
        
    elif approach == 'Training Pipeline':
        st.header("Training pipeline")
        st.markdown("""
                    * Loss function: CrossEntropyLoss
                    * Optimizer: Adam
                    * Learning rate scheduling
                    * GPU acceleration (if available) """)
        
    elif approach == 'Evaluation Metrics':
        st.header("Evaluation Metrics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Overall Accuracy", "99.1%")
        with col2:
            st.metric("Weighted F1-Score", "0.9914")
        with col3:
            st.metric("Total Images Tested", "3602")

    elif approach == 'Generative AI Integration':
        st.header("Generative AI")
        st.write("**Model Used**: llama-3.3-70b-versatile")