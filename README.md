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
This project implements a neural network-based trajectory planner for autonomous vehicles. The model predicts future vehicle trajectories based on dash cam images, historical vehicle positions, and driving commands, serving as a path planning component for autonomous driving systems.

## Project Structure

### Data
The dataset consists of:
- **Train set**: 1000 examples
- **Validation set**: Separate examples for evaluation
- **Test set**: Examples without ground truth trajectories for submissions

Each data sample contains:
- `camera`: RGB front-view image (H, W, 3)
- `sdc_history_feature`: 21-step historical trajectory, shape (21, 3)
- `sdc_future_feature`: 60-step future trajectory, shape (60, 3) (not available in test)
- `driving_command`: one of `['forward', 'left', 'right']`
- `semantic_label`: Semantic segmentation map
- `depth`: Depth map information

### Model Architecture

The `DrivingPlanner` model consists of several key components:

1. **Image Encoder**:
   - Pretrained ResNet50 backbone (with frozen early layers, trainable later layers)
   - Custom CNN head with batch normalization and dropout
   - Adaptive average pooling to standardize feature dimensions

2. **Trajectory History Encoder**:
   - Processes vehicle's past trajectory including position, velocity, and acceleration
   - Linear layer with ReLU activation

3. **Command Embedding**:
   - Embeds driving commands (forward, left, right) into a learned representation

4. **Trajectory Decoder**:
   - Multi-layer network with batch normalization and dropout
   - Outputs 60-step future trajectory predictions (x, y, heading)

5. **Auxiliary Task - Depth Prediction** (optional):
   - Upsampling network for depth map prediction
   - Used as an additional training signal to improve feature learning

### Data Processing & Augmentation

The `DrivingDataset` class implements several augmentation techniques:
- Random affine transformations (translation, scaling)
- Color jittering (brightness, contrast, saturation, hue)
- Horizontal flipping with command adaptation
- Gaussian noise addition to trajectories
- Feature augmentation (velocity and acceleration computation)

### Loss Functions

The training incorporates multiple loss components:
- MSE trajectory loss for position prediction
- Smooth L1 loss for heading prediction
- Optional depth prediction loss
- Smoothness constraints:
  - Acceleration smoothness loss (jerk minimization)
  - Curve smoothness loss (to avoid erratic turns)
- Weighted loss for turning scenarios (higher weight for left/right turns)

### Training Pipeline

The training process includes:
- Batch processing with separate train/validation stages
- Learning rate optimization with Adam optimizer
- Comprehensive logging of metrics
- Model checkpointing

### Evaluation Metrics

- **ADE (Average Displacement Error)**: Average L2 distance between predicted and ground truth trajectories
- **FDE (Final Displacement Error)**: L2 distance at the final prediction point
- **MSE (Mean Square Error)**: Overall trajectory prediction error
- **Depth Loss**: For the auxiliary depth prediction task

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- torchvision
- NumPy
- Pandas
- Matplotlib

### Installation
1. Clone this repository
2. Install required packages: `pip install torch torchvision numpy pandas matplotlib`
3. Prepare your data directories: `train`, `val`, and `test_public`

### Training
```python
# Load and prepare datasets
train_dataset = DrivingDataset(train_files)
val_dataset = DrivingDataset(val_files)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=2)

# Initialize model and optimizer
model = DrivingPlanner(use_depth_aux=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
logger = Logger()

# Train the model
train(model, train_loader, val_loader, optimizer, logger, num_epochs=50, lambda_depth=0.05, use_depth_aux=True)

# Save trained model
torch.save(model.state_dict(), "driving_planner_model_depth.pth")
```

### Testing and Submission Generation
```python
# Load test data
test_dataset = DrivingDataset(test_files, test=True)
test_loader = DataLoader(test_dataset, batch_size=250, num_workers=2)

# Generate predictions
model.eval()
with torch.no_grad():
    # Get predictions for all test samples
    # Format results into submission format
    # Save to CSV
```

## Model Performance Highlights

The model achieves strong performance by:
- Using transfer learning with a pretrained ResNet50 backbone
- Incorporating multi-task learning with depth prediction
- Applying data augmentation to improve generalization
- Using trajectory smoothness constraints
- Implementing weighted loss functions for different driving scenarios

## Visualizing Results

The model output can be visualized by plotting:
- Current camera view
- Past trajectory history
- Predicted future trajectory
- Optional depth prediction map

## Submission

The final output is a CSV file (`submission_phase2.csv`) containing the predicted x,y coordinates for each test example, which can be submitted for evaluation.
