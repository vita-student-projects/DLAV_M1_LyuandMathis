# DLAV_M1_LyuandMathis
This project implements an EnhancedPlanner model to predict the future trajectory of a self-driving vehicle based on past trajectory, front camera image, and driving command (forward/left/right). The goal is to forecast 60 steps into the future.
Dataset

The dataset contains pickle files with the following keys:
	•	camera: RGB front-view image (H, W, 3)
	•	sdc_history_feature: 21-step historical trajectory, shape (21, 3)
	•	sdc_future_feature: 60-step future trajectory, shape (60, 3) (not available in test)
	•	driving_command: one of ['forward', 'left', 'right']

 Model Architecture

 	•	Input: (camera image, trajectory history, driving command)
	•	Output: predicted trajectory of shape (60, 3) → (x, y, heading)
