# Multimodal ML – Housing Price Prediction Using Images + Tabular Data

## 1. Introduction

This project aimed to predict housing prices by leveraging both structured tabular data and house images. The core idea was to combine the rich visual information from images, extracted using Convolutional Neural Networks (CNNs), with traditional numerical features to build a more robust and accurate prediction model.

## 2. Methodology

### 2.1 Data Collection and Preparation

The dataset used for this project is the Houses Dataset from a GitHub repository (emanhamed/Houses-dataset). This dataset contains 535 samples, each with tabular information (number of bedrooms, bathrooms, area, zipcode, and price) and four corresponding images (bathroom, bedroom, frontal, and kitchen).

- **Tabular Data:** The `HousesInfo.txt` file was parsed to extract numerical features and house prices. This data was then saved as `housing_data.csv`.
- **Image Data:** The image files were organized by house ID and room type. A custom PyTorch `Dataset` class (`HouseDataset`) was implemented to handle loading both tabular data and the corresponding images for each house. Missing images were handled by returning a tensor of zeros with the expected shape.

### 2.2 CNN Model Development for Image Feature Extraction

A `CNNFeatureExtractor` module was implemented using PyTorch. This simple CNN consists of two convolutional layers followed by max-pooling and a fully connected layer. The purpose of this module is to extract a fixed-size feature vector from each input image. Images were preprocessed with resizing, conversion to tensors, and normalization.

### 2.3 Multimodal Fusion and Regression Model Implementation

The `MultimodalModel` was designed to combine the extracted image features with the tabular data. For each house, the features from its four images (bathroom, bedroom, frontal, kitchen) were extracted using the `CNNFeatureExtractor` and then concatenated. This concatenated image feature vector was then combined with the tabular features. A regression head, consisting of fully connected layers with ReLU activations, was added to predict the final housing price.

### 2.4 Model Training and Evaluation

The dataset was split into 80% training and 20% testing sets. The model was trained using Mean Squared Error (MSE) as the loss function and the Adam optimizer. The training process was run for a limited number of epochs for demonstration purposes. After training, the model's performance was evaluated on the test set using two common regression metrics:

- **Mean Absolute Error (MAE):** Measures the average magnitude of the errors in a set of predictions, without considering their direction.
- **Root Mean Squared Error (RMSE):** Measures the square root of the average of the squared errors. It gives a relatively high weight to large errors.

## 3. Results

During training, the model's loss decreased over epochs, indicating that it was learning to predict housing prices. The final evaluation on the test set yielded the following metrics:




MAE: 315208.95
RMSE: 508416.46

## 4. Discussion and Future Work

The initial results show that the model is able to learn from both tabular and image data to predict housing prices. However, the MAE and RMSE values are quite high, indicating that there is significant room for improvement. This could be attributed to several factors:

- **Simple CNN Architecture:** The current CNN is a very basic one. Using pre-trained, more complex CNN architectures (e.g., ResNet, VGG) as feature extractors could significantly improve performance. These models have learned rich representations from large image datasets and can be fine-tuned for this specific task.
- **Limited Image Data:** Using only the frontal image for each house simplifies the model but might discard valuable information present in the bathroom, bedroom, and kitchen images. Incorporating features from all four images, perhaps through more sophisticated fusion techniques, could lead to better predictions.
- **Data Normalization/Scaling:** While image normalization is applied, further scaling or normalization of tabular features might be beneficial.
- **Hyperparameter Tuning:** The current learning rate, batch size, and number of epochs are chosen for demonstration. Extensive hyperparameter tuning could lead to better model performance.
- **Dataset Size:** 535 samples might be relatively small for training deep learning models effectively, especially for image-based tasks. Augmenting the dataset or using transfer learning more extensively could help.
- **Feature Engineering:** More advanced feature engineering on the tabular data could also improve results.

Future work would involve exploring these areas to enhance the model's accuracy and robustness. This includes experimenting with different CNN architectures, more advanced fusion strategies, and comprehensive hyperparameter optimization. Additionally, investigating the impact of individual image types on the prediction accuracy could provide valuable insights.


