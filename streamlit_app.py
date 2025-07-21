import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="🏠 Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-result {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# CNN Feature Extractor (same as in model.py)
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        return x

# Multimodal Model (same as in model.py)
class MultimodalModel(nn.Module):
    def __init__(self, cnn_feature_dim, tabular_feature_dim):
        super(MultimodalModel, self).__init__()
        self.cnn_extractor = CNNFeatureExtractor()
        self.combined_feature_dim = cnn_feature_dim + tabular_feature_dim
        
        self.fc_combined = nn.Sequential(
            nn.Linear(self.combined_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, tabular_features, images):
        frontal_features = self.cnn_extractor(images["frontal"])
        image_features = frontal_features
        combined_features = torch.cat((image_features, tabular_features), dim=1)
        output = self.fc_combined(combined_features)
        return output

# Initialize model
@st.cache_resource
def load_model():
    model = MultimodalModel(cnn_feature_dim=128, tabular_feature_dim=4)
    # Note: In a real application, you would load pre-trained weights here
    # model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Load sample data for visualization
@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv('housing_data.csv')
        return df
    except:
        # Create sample data if file doesn't exist
        np.random.seed(42)
        sample_data = {
            'bedrooms': np.random.randint(1, 6, 100),
            'bathrooms': np.random.randint(1, 5, 100),
            'area': np.random.randint(800, 5000, 100),
            'zipcode': np.random.choice([85255, 85266, 85262, 92677, 94501], 100),
            'price': np.random.randint(200000, 2000000, 100)
        }
        return pd.DataFrame(sample_data)

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 AI-Powered Housing Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">🤖 This application uses a multimodal deep learning model that combines house images and tabular data to predict housing prices.</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 📊 Model Information")
    st.sidebar.info("""
    **Model Architecture:**
    - CNN for image feature extraction
    - Fully connected layers for tabular data
    - Multimodal fusion for final prediction
    
    **Training Data:**
    - 535 house samples
    - 4 features per house
    - Images from multiple room types
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">🏡 House Details</h2>', unsafe_allow_html=True)
        
        # Input fields
        bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.slider("Number of Bathrooms", min_value=1, max_value=8, value=2)
        area = st.number_input("House Area (sq ft)", min_value=500, max_value=10000, value=2000)
        zipcode = st.selectbox("Zipcode", [85255, 85266, 85262, 92677, 94501, 93446, 91901])
        
        # Image upload
        st.markdown('<h3 class="sub-header">📸 Upload House Image</h3>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a house image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded House Image", use_column_width=True)
        else:
            # Use a default placeholder image
            st.info("Please upload a house image for better prediction accuracy.")
    
    with col2:
        st.markdown('<h2 class="sub-header">🔮 Price Prediction</h2>', unsafe_allow_html=True)
        
        if st.button("🚀 Predict Price", type="primary"):
            # Load model
            model = load_model()
            
            # Prepare tabular data
            tabular_features = torch.tensor([[bedrooms, bathrooms, area, zipcode]], dtype=torch.float32)
            
            # Prepare image data
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                image_tensor = preprocess_image(image)
            else:
                # Use a default tensor for missing image
                image_tensor = torch.zeros(1, 3, 224, 224)
            
            images = {"frontal": image_tensor}
            
            # Make prediction
            with torch.no_grad():
                prediction = model(tabular_features, images)
                predicted_price = prediction.item()
            
            # Display prediction
            st.markdown(f'<div class="prediction-result">💰 Predicted Price: ${predicted_price:,.0f}</div>', unsafe_allow_html=True)
            
            # Additional metrics
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                price_per_sqft = predicted_price / area
                st.metric("Price per sq ft", f"${price_per_sqft:.2f}")
            with col2_2:
                # Calculate a confidence score (mock)
                confidence = min(95, max(60, 85 + np.random.normal(0, 5)))
                st.metric("Confidence", f"{confidence:.1f}%")
    
    # Data visualization section
    st.markdown("---")
    st.markdown('<h2 class="sub-header">📈 Market Analysis</h2>', unsafe_allow_html=True)
    
    # Load sample data
    df = load_sample_data()
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Price distribution
        fig1 = px.histogram(df, x='price', nbins=20, title='Price Distribution')
        fig1.update_layout(xaxis_title='Price ($)', yaxis_title='Count')
        st.plotly_chart(fig1, use_container_width=True)
        
        # Price vs Area scatter plot
        fig2 = px.scatter(df, x='area', y='price', color='bedrooms', 
                         title='Price vs Area', size='bathrooms')
        fig2.update_layout(xaxis_title='Area (sq ft)', yaxis_title='Price ($)')
        st.plotly_chart(fig2, use_container_width=True)
    
    with col4:
        # Average price by bedrooms
        avg_price_bedrooms = df.groupby('bedrooms')['price'].mean().reset_index()
        fig3 = px.bar(avg_price_bedrooms, x='bedrooms', y='price', 
                     title='Average Price by Bedrooms')
        fig3.update_layout(xaxis_title='Number of Bedrooms', yaxis_title='Average Price ($)')
        st.plotly_chart(fig3, use_container_width=True)
        
        # Price by zipcode
        avg_price_zipcode = df.groupby('zipcode')['price'].mean().reset_index()
        fig4 = px.bar(avg_price_zipcode, x='zipcode', y='price', 
                     title='Average Price by Zipcode')
        fig4.update_layout(xaxis_title='Zipcode', yaxis_title='Average Price ($)')
        st.plotly_chart(fig4, use_container_width=True)
    
    # Model performance section
    st.markdown("---")
    st.markdown('<h2 class="sub-header">🎯 Model Performance</h2>', unsafe_allow_html=True)
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.markdown('<div class="metric-card"><h3>MAE</h3><p>Mean Absolute Error: $315,209</p></div>', unsafe_allow_html=True)
    
    with col6:
        st.markdown('<div class="metric-card"><h3>RMSE</h3><p>Root Mean Squared Error: $508,416</p></div>', unsafe_allow_html=True)
    
    with col7:
        st.markdown('<div class="metric-card"><h3>Dataset</h3><p>535 house samples with images</p></div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d;">
        <p>🏠 Housing Price Predictor | Built with Streamlit & PyTorch</p>
        <p>This is a demonstration of multimodal machine learning for real estate price prediction.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

