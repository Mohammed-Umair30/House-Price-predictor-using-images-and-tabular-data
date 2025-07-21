# House-Price-predictor-using-images-and-tabular-data
# 🏠 AI-Powered Housing Price Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://8501-ixakbkaeslfvs765jqkct-e8e0f43b.manusvm.computer)
[![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated multimodal machine learning application that predicts housing prices by combining **Convolutional Neural Networks (CNNs)** for image feature extraction with traditional tabular data analysis. Built with PyTorch and deployed as an interactive Streamlit web application.

## 🌟 Features

- **🤖 Multimodal AI**: Combines house images and tabular data for accurate price predictions
- **🖼️ Image Processing**: CNN-based feature extraction from house photos
- **📊 Interactive Dashboard**: Beautiful Streamlit interface with real-time predictions
- **📈 Data Visualization**: Comprehensive market analysis with Plotly charts
- **🎯 Performance Metrics**: MAE and RMSE evaluation with confidence scores
- **📱 Responsive Design**: Works seamlessly on desktop and mobile devices

## 🚀 Live Demo

**[🌐 Try the Live Application](https://8501-ixakbkaeslfvs765jqkct-e8e0f43b.manusvm.computer)**


### Main Interface
![Housing Price Predictor Interface](https://via.placeholder.com/800x400/1f77b4/ffffff?text=Housing+Price+Predictor+Interface)

### Data Visualizations
![Market Analysis Charts](https://via.placeholder.com/800x400/2ca02c/ffffff?text=Market+Analysis+Charts)

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   House Images  │    │  Tabular Data    │    │   Predictions   │
│                 │    │                  │    │                 │
│ • Bathroom      │    │ • Bedrooms       │    │ • Price         │
│ • Bedroom       │────│ • Bathrooms      │────│ • Confidence    │
│ • Kitchen       │    │ • Area           │    │ • Price/sq ft   │
│ • Frontal       │    │ • Zipcode        │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ CNN Feature     │    │ Feature          │    │ Regression      │
│ Extractor       │    │ Normalization    │    │ Head            │
│                 │    │                  │    │                 │
│ Conv2D → Pool   │    │ StandardScaler   │    │ FC Layers       │
│ Conv2D → Pool   │────│ MinMaxScaler     │────│ ReLU            │
│ Flatten → FC    │    │                  │    │ Dropout         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, torchvision
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy, scikit-learn
- **Visualization**: Plotly, matplotlib
- **Image Processing**: PIL (Pillow)
- **Deployment**: Streamlit Cloud

## 📦 Installation

### Prerequisites
- Python 3.11+
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/housing-price-predictor.git
   cd housing-price-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   ```bash
   wget https://github.com/emanhamed/Houses-dataset/archive/master.zip
   unzip master.zip
   ```

4. **Preprocess the data**
   ```bash
   python preprocess_data.py
   ```

5. **Train the model** (optional)
   ```bash
   python model.py
   ```

6. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

## 📋 Requirements

```txt
streamlit>=1.46.1
torch>=2.0.0
torchvision>=0.15.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.15.0
Pillow>=10.0.0
```

## 📊 Dataset

The project uses the **Houses Dataset** containing:
- **535 house samples** with corresponding images
- **4 tabular features**: bedrooms, bathrooms, area, zipcode
- **4 image types per house**: bathroom, bedroom, frontal, kitchen
- **Price range**: $200K - $2M

### Data Structure
```
Houses-dataset-master/
├── Houses Dataset/
│   ├── HousesInfo.txt          # Tabular data
│   ├── 1_bathroom.jpg          # House images
│   ├── 1_bedroom.jpg
│   ├── 1_frontal.jpg
│   ├── 1_kitchen.jpg
│   └── ...
└── README.md
```

## 🧠 Model Architecture

### CNN Feature Extractor
```python
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
```

### Multimodal Fusion
```python
class MultimodalModel(nn.Module):
    def __init__(self, cnn_feature_dim, tabular_feature_dim):
        # Combines CNN features with tabular data
        self.combined_feature_dim = cnn_feature_dim + tabular_feature_dim
        self.fc_combined = nn.Sequential(
            nn.Linear(self.combined_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
```

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **MAE** | $315,209 |
| **RMSE** | $508,416 |
| **Dataset Size** | 535 samples |
| **Training Time** | ~2 minutes |

## 🎯 Usage

### Web Interface
1. Visit the [live application](https://8501-ixakbkaeslfvs765jqkct-e8e0f43b.manusvm.computer)
2. Input house details (bedrooms, bathrooms, area, zipcode)
3. Upload a house image (optional)
4. Click "🚀 Predict Price" to get the prediction

### Programmatic Usage
```python
import torch
from model import MultimodalModel
from preprocess_data import preprocess_image

# Load model
model = MultimodalModel(cnn_feature_dim=128, tabular_feature_dim=4)
model.eval()

# Prepare data
tabular_features = torch.tensor([[3, 2, 2000, 85255]], dtype=torch.float32)
image_tensor = preprocess_image(house_image)
images = {"frontal": image_tensor}

# Make prediction
with torch.no_grad():
    prediction = model(tabular_features, images)
    predicted_price = prediction.item()
```

## 📁 Project Structure

```
housing-price-predictor/
├── streamlit_app.py           # Main Streamlit application
├── model.py                   # PyTorch model definitions
├── preprocess_data.py         # Data preprocessing utilities
├── report.md                  # Technical report
├── housing_data.csv           # Processed tabular data
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── Houses-dataset-master/     # Raw dataset
    └── Houses Dataset/
        ├── HousesInfo.txt
        └── *.jpg
```

## 🔮 Future Enhancements

- [ ] **Advanced CNN Architectures**: Implement ResNet, VGG, or EfficientNet
- [ ] **Transfer Learning**: Use pre-trained models for better feature extraction
- [ ] **Data Augmentation**: Increase dataset size with image transformations
- [ ] **Hyperparameter Tuning**: Optimize learning rate, batch size, and architecture
- [ ] **Multi-Image Fusion**: Utilize all 4 room images simultaneously
- [ ] **Geospatial Features**: Incorporate location-based market data
- [ ] **Real-time Data**: Connect to live real estate APIs
- [ ] **Model Interpretability**: Add SHAP or LIME explanations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Houses Dataset by emanhamed](https://github.com/emanhamed/Houses-dataset)
- **Framework**: [Streamlit](https://streamlit.io/) for the amazing web framework
- **Deep Learning**: [PyTorch](https://pytorch.org/) for the neural network implementation
- **Visualization**: [Plotly](https://plotly.com/) for interactive charts

## 📞 Contact

**Mohammad Umair** - hafizumair07.hm@gmail.com -

**Project Link**: https://github.com/Mohammed-Umair30/House-Price-predictor-using-images-and-tabular-data

---

⭐ **Star this repository if you found it helpful!** ⭐

