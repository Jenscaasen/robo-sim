import os
import numpy as np
from tensorflow import keras

# Disable GPU and XLA JIT to match run2.py
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

print("files in folder: ")
print(os.listdir('./'))
# Find the first .keras file in the Supervised_learning folder
keras_files = [f for f in os.listdir('./') if f.endswith('.keras')]
if not keras_files:
    raise FileNotFoundError("No .keras file found in Supervised_learning folder")
model_path = keras_files[0]  # Use the first one found

# Load the model
model = keras.models.load_model(model_path)

def predict_servo_positions(pixels):
    """
    Predict servo positions from 6 pixel values.

    Parameters
    ----------
    pixels : list of 6 floats
        Pixel coordinates: [cam1_x, cam1_y, cam2_x, cam2_y, cam3_x, cam3_y]

    Returns
    -------
    list of 5 floats
        Predicted servo positions (angles in radians)
    """
    if len(pixels) != 6:
        raise ValueError("Exactly 6 pixel values required")

    # Convert to numpy array and reshape
    sample = np.array(pixels, dtype=np.float32)[None, :]  # shape (1,6)

    # Apply same scaling as during training
    sample[:, 0::2] /= 639.0  # x coordinates
    sample[:, 1::2] /= 479.0  # y coordinates

    # Predict
    pred_scaled = model.predict(sample, verbose=0)[0]

    # Scale back to original units
    predicted_angles = pred_scaled * 1.5

    return predicted_angles.tolist()