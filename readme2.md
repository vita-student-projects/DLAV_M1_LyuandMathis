# Deep Learning Autonomous Vehicle (DLAV) Path Planning Project

## Overview
This project involves training a neural network to predict the future trajectory of a vehicle based on dash cam images, the vehicle's past positions, and driving commands. The model serves as a planner that can determine the optimal path for an autonomous vehicle.

## Project Structure

### Data
The dataset consists of:
- **Train set**: 1000 examples
- **Validation set**: Separate examples for evaluation
- **Test set**: Examples without ground truth trajectories for submissions

Each data sample includes:
- `camera`: Dash cam image of the current vehicle state
- `sdc_history_feature`: Past trajectory points (21 time steps, [x, y, heading])
- `driving_command`: Command instructions ('forward', 'left', 'right')
- `semantic_label`: Semantic segmentation map (optional)
- `sdc_future_feature`: Future trajectory points (60 time steps) - not available in test set

### Model Architecture
The provided baseline model (`EnhancedPlanner`) includes:
- A CNN-based image encoder to process dash cam images
- A trajectory history encoder
- Command embedding module
- Fully connected layers to output the predicted future trajectory

### Training Pipeline
- **Dataset Class**: Handles data loading and preprocessing
- **Logger**: Tracks and displays training metrics
- **Training Loop**: Implements model training with separate losses for position and heading

### Evaluation Metrics
- **ADE (Average Displacement Error)**: Average L2 distance between predicted and ground truth trajectories
- **FDE (Final Displacement Error)**: L2 distance at the final prediction point

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

# Generate predictions for test set
# Run the test prediction code to create submission_phase1.csv
```

## Improving the Model
The baseline model can be improved in many ways:
- Experimenting with different CNN architectures (ResNet, EfficientNet)
- Using attention mechanisms or transformers
- Adding data augmentation techniques
- Implementing learning rate schedulers
- Exploring different loss functions
- Utilizing the semantic segmentation data
- Increasing model capacity for complex scenarios

## Visualizing Results
The notebook includes visualization code to display:
- Current camera view
- Past trajectory (gold)
- Ground truth future trajectory (green)
- Predicted future trajectory (red)

## Submission
The final output is a CSV file (`submission_phase1.csv`) containing the predicted x,y coordinates for each test example, which can be submitted to the leaderboard for evaluation.
