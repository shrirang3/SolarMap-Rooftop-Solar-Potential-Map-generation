
# SolarMap: Rooftop Solar Potential Map Generation

The project focuses on estimating solar energy potential from segmented rooftops using the Geneva Rooftop Dataset, consisting of high-resolution satellite images. Preprocessing steps, including resizing, normalization, and augmentation, prepared the data for an optimized U-Net model enhanced with attention gates, dense layers, and bilinear upsampling, achieving superior segmentation performance with an IoU of 0.66 and AP of 0.88. The segmented rooftop areas were analyzed using contour detection to calculate their real-world size and determine the number of standard solar panels that could be installed. Using location-specific solar irradiance data and panel efficiency, the daily solar energy generation was estimated, offering actionable insights for sustainable energy planning and urban development.


## Dataset Details


This project utilizes the Geneva Rooftop Dataset for rooftop segmentation tasks. Special thanks to the repository [Photovoltaic-datection](https://github.com/riccardocadei/photovoltaic-detection) for providing valuable resources and inspiration for this project. The dataset contains high-resolution satellite images for rooftop segmentation. The dataset used for this project is referred to as Geneva_Rooftop. It consists of high-resolution aerial images of rooftops in the city of Geneva, Switzerland. The images capture a wide variety of rooftops, including those with different architectural styles and environmental conditions. The dataset is specifically designed for the task of pixel-wise binary semantic segmentation, where the goal is to classify each pixel for rooftop photovoltaic (PV) panel installation.You can access the dataset [Dataset Link](https://drive.google.com/file/d/1qDSLOEWwiuijwJT1v-VBkPzHhHev4CxD/view?usp=drive_link).


## Building the Repo locally:

Install required libraries:

```bash
  cd <project_folder>
```

```bash
  pip install -r requirements.txt
```

Training script script:

```bash
  python optimized_unet_train.py --train_dir ./data/train --val_dir ./data/val --test_dir ./data/test --epochs 50 --batch_size 4 --lr 0.001

```

Predict script:

```bash
  python predict.py --image_path ./input/image.png --model_path ./models/best_unet_model.pth 

```

## Project Flow





#### 1. **Setting Up Data and Models**
**Function**: Initialize the environment by loading the pre-processed Geneva Rooftop Dataset and setting up the U-Net model for training and inference.

- **Dataset Preparation**: Make sure the dataset with high-resolution satellite images and corresponding ground truth masks is available for training.
- **Model Setup**: The U-Net model is initialized, with optimizations including attention gates, dense layers, and bilinear upsampling.
- **Training Configuration**: Set up the training parameters like batch size, learning rate, and epochs using a configuration file or command-line arguments.

#### 2. **Training the Model**
**Function**: Train the U-Net model to segment rooftop areas in satellite images.

**Steps**:
- **Input**: Use the pre-processed dataset consisting of satellite images and their respective segmentation masks (rooftop footprints).
- **Model Training**: Train the U-Net model using the training data, optimizing the loss function (e.g., Dice loss or IoU) to improve segmentation accuracy.
- **Evaluation**: Validate the model on a separate validation dataset and track the performance metrics such as IoU and Dice coefficient.
- **Checkpointing**: Save the best-performing model for future inference.


#### 3. **Calculating Solar Energy Potential**
**Function**: Estimate the solar energy generation potential based on segmented rooftop areas.

**Steps**:
- **Input**: Use the segmented rooftop areas from the previous step and combine them with location-specific solar irradiance data.
- **Calculation**: Multiply the rooftop area by the solar irradiance for the specific location and panel efficiency to estimate potential energy generation.
- **Seasonal Adjustments**: Factor in seasonal variations by adjusting the solar irradiance based on the time of year (e.g., higher irradiance in summer, lower in winter).
- **Output**: Calculate the total energy generation in kWh/day for the given rooftops.

#### 5. **Visualizing Results**
**Function**: Visualize the segmented rooftops and the estimated solar energy potential.

**Steps**:
- **Visual Output**: Display the segmented rooftops on the satellite image with the calculated solar energy generation overlaid.
- **Reports**: Generate a report summarizing the total rooftop area, number of solar panels that can be installed, and the estimated daily energy production.


#### 7. **Advanced Features**
**Extras**:
- **Interactive Mapping**: Integrate a map interface (e.g., using Google Maps API) to select regions of interest and visualize their solar potential in real-time.
- **Energy Forecasting**: Incorporate historical weather and solar irradiance data to forecast energy generation over different seasons or years.
- **Optimization**: Implement advanced optimization techniques, such as using a ResNet backbone for U-Net or experimenting with different architectures like DeepLabV3 for better segmentation results.

#### 8. **Results and Reporting**
Provide detailed results for stakeholders or researchers.

![image](https://github.com/user-attachments/assets/f89bdd46-ddc3-4f28-9bd3-07b62e95a3db)

Above figure illustrates the practical output of the optimized U-Net model for rooftop segmentation, showcasing the progression from the original aerial image to the final binary mask. 
The original image (left) serves as the input to the U-Net model, which has been enhanced with attention gates and dense layers to improve the segmentation accuracy. 
The middle panel, referred to as the probability mask, is a direct output of the model, where each pixel is assigned a probability indicating its likelihood of being part of a rooftop. 


| Model    | Test IoU    | Val IoU   |
| ------------- | ------------- | ------------- |
| U-Net Architecture | 0.5886 | 0.6121 |
| Our Proposed Architecture (U-Net+Attention+Dense layers | 0.667 | 0.6723 |

Above table illustrates the performance comparison between the standard U-Net architecture and the proposed optimized architecture, which incorporates attention gates and dense layers, for both the test and validation datasets.
The standard U-Net achieved an IoU of 0.5886 on the test dataset and 0.6121 on the validation dataset, highlighting its baseline capability in segmenting rooftops from aerial images.




## Authors

- [Shrirang Mahankaliwar](https://www.github.com/shrirang3)
- [Swayamprakash Mahale](https://github.com/Swayamprakash1)


