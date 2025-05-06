# DLAV_M1_LyuandMathis
This project implements an EnhancedPlanner model to predict the future trajectory of a self-driving vehicle based on past trajectory, front camera image, and driving command (forward/left/right). The goal is to forecast 60 steps into the future.

# Dataset

The dataset contains pickle files with the following keys:
- `camera`: RGB front-view image (H, W, 3)
- `sdc_history_feature`: 21-step historical trajectory, shape (21, 3)
- `sdc_future_feature`: 60-step future trajectory, shape (60, 3) (not available in test)
- `driving_command`: one of `['forward', 'left', 'right']`

### Testing
`sdc_future_feature` is removed from the input and instead the model outputs a prediction for `sdc_future_feature`.


 	•	Input: (camera image, trajectory history, driving command)
	•	Output: predicted trajectory of shape (60, 3) → (x, y, heading)


# Model Architecture

We mostly followed the originally given structure, adding a few elements with the code. Those changes are : 
- One-hot encoding for the driving direction
- Added some complexity to the model :
  - A 3 layer to the base CNN + batchnorm + average pooling
  - A history encoder
  - A command embedding
  - A bigger decoder with two fully connected layers
  - Camera input normalizing
 

# Results
After training for 50 epochs, our ADE dropped under 2 almost consistently. The trajectory seems decently correct, with a quite good overall direction, although sometimes going faster or slower than the ground truth (the curve being longer/shorter in the same direction).
