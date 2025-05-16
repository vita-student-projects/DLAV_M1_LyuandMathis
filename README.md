# Deep Learning Autonomous Vehicle (DLAV) Path Planning Project

# Milestone 1
## Overview
This project involves training a neural network to predict the future trajectory of a vehicle based on dash cam images, the vehicle's past positions, and driving commands. The model serves as a planner that can determine the optimal path for an autonomous vehicle.

## Project Structure

### Data
The dataset consists of:
- **Train set**: 1000 examples
- **Validation set**: Separate examples for evaluation
- **Test set**: Examples without ground truth trajectories for submissions

The dataset contains pickle files with the following keys:
- `camera`: RGB front-view image (H, W, 3)
- `sdc_history_feature`: 21-step historical trajectory, shape (21, 3)
- `sdc_future_feature`: 60-step future trajectory, shape (60, 3) (not available in test)
- `driving_command`: one of `['forward', 'left', 'right']`
- `semantic_label`: Semantic segmentation map (optional)


### Testing
`sdc_future_feature` is removed from the input and instead the model outputs a prediction for `sdc_future_feature`.

 	•	Input: (camera image, trajectory history, driving command)
	•	Output: predicted trajectory of shape (60, 3) → (x, y, heading)



### Model Architecture
(We mostly followed the originally given structure, adding a few elements with the code)
The provided baseline model (`EnhancedPlanner`) now includes:
- One-hot encoding for the driving direction
- Added some complexity to the model :
  - A 3 layer to the base CNN + batchnorm + average pooling
  - A history encoder
  - A command embedding
  - A bigger decoder with two fully connected layers
  - Camera input normalizing

### Training Pipeline
- **Dataset Class**: Handles data loading and preprocessing
- **Logger**: Tracks and displays training metrics
- **Training Loop**: Implements model training with separate losses for position and heading

### Evaluation Metrics
- **ADE (Average Displacement Error)**: Average L2 distance between predicted and ground truth trajectories
- **FDE (Final Displacement Error)**: L2 distance at the final prediction point
- **MSE (Mean square error)**: MSE on the final prediction

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- Matplotlib
- NumPy
- Pandas

### Installation
1. Download the dataset files from the provided Google Drive links
2. Extract the zip files to your working directory

### Training
```python
# Load and prepare the datasets
train_dataset = DrivingDataset(train_files)
val_dataset = DrivingDataset(val_files)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=2)

# Initialize model and optimizer
model = EnhancedPlanner()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train the model
train(model, train_loader, val_loader, optimizer, logger, num_epochs=50)
```

### Creating a Submission
After training, you can generate predictions for the test set and create a submission file:
```python
# Save your trained model
torch.save(model.state_dict(), "phase1_model.pth")
```

## Improving the Model
The baseline model can be improved in many ways:
- Experimenting with different CNN architectures (ResNet, EfficientNet)
- Using attention mechanisms or transformers
- Adding data augmentation techniques
- Implementing learning rate schedulers
- Exploring different loss functions
- Increasing model capacity for complex scenarios

## Visualizing Results
The notebook includes visualization code to display:
- Current camera view
- Past trajectory (gold)
- Ground truth future trajectory (green)
- Predicted future trajectory (red)

## Submission
The final output is a CSV file (`submission_phase1.csv`) containing the predicted x,y coordinates for each test example, which can be submitted to the leaderboard for evaluation.

# Milestone 2

## Overview
This project involves training a neural network to predict the future trajectory of a vehicle based on dash cam images, the vehicle's past positions, and driving commands. The model serves as a planner that can determine the optimal path for an autonomous vehicle.

## Project Structure

### Data
The dataset consists of:
- **Train set**: 1000 examples
- **Validation set**: Separate examples for evaluation
- **Test set**: Examples without ground truth trajectories for submissions

The dataset contains pickle files with the following keys:
- `camera`: RGB front-view image (H, W, 3)
- `sdc_history_feature`: 21-step historical trajectory, shape (21, 3)
- `sdc_future_feature`: 60-step future trajectory, shape (60, 3) (not available in test)
- `driving_command`: one of `['forward', 'left', 'right']`
- `semantic_label`: Semantic segmentation map (optional)


### Testing
`sdc_future_feature` is removed from the input and instead the model outputs a prediction for `sdc_future_feature`.

 	•	Input: (camera image, trajectory history, driving command)
	•	Output: predicted trajectory of shape (60, 3) → (x, y, heading)



### Model Architecture
(We mostly followed the originally given structure, adding a few elements with the code)
The provided baseline model (`EnhancedPlanner`) now includes:
- One-hot encoding for the driving direction
- Added some complexity to the model :
  - A 3 layer to the base CNN + batchnorm + average pooling
  - A history encoder
  - A command embedding
  - A bigger decoder with two fully connected layers
  - Camera input normalizing

### Training Pipeline
- **Dataset Class**: Handles data loading and preprocessing
- **Logger**: Tracks and displays training metrics
- **Training Loop**: Implements model training with separate losses for position and heading

### Evaluation Metrics
- **ADE (Average Displacement Error)**: Average L2 distance between predicted and ground truth trajectories
- **FDE (Final Displacement Error)**: L2 distance at the final prediction point
- **MSE (Mean square error)**: MSE on the final prediction

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- Matplotlib
- NumPy
- Pandas

### Installation
1. Download the dataset files from the provided Google Drive links
2. Extract the zip files to your working directory

### Training
```python
# Load and prepare the datasets
train_dataset = DrivingDataset(train_files)
val_dataset = DrivingDataset(val_files)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=2)

# Initialize model and optimizer
model = EnhancedPlanner()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train the model
train(model, train_loader, val_loader, optimizer, logger, num_epochs=50)
```

### Creating a Submission
After training, you can generate predictions for the test set and create a submission file:
```python
# Save your trained model
torch.save(model.state_dict(), "phase1_model.pth")
```

## Improving the Model
The baseline model can be improved in many ways:
- Experimenting with different CNN architectures (ResNet, EfficientNet)
- Using attention mechanisms or transformers
- Adding data augmentation techniques
- Implementing learning rate schedulers
- Exploring different loss functions
- Increasing model capacity for complex scenarios

## Visualizing Results
The notebook includes visualization code to display:
- Current camera view
- Past trajectory (gold)
- Ground truth future trajectory (green)
- Predicted future trajectory (red)

## Submission
The final output is a CSV file (`submission_phase1.csv`) containing the predicted x,y coordinates for each test example, which can be submitted to the leaderboard for evaluation.
