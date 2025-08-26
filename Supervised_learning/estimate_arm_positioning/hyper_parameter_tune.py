import os
import shutil
import glob
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU for this script
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"  # ensure XLA JIT is off

import keras_tuner as kt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# ──────────────────────────────────────────────────────────────
# 1.  Select and load dataset
# ──────────────────────────────────────────────────────────────
def select_dataset():
    """Let user select a dataset file from available options"""
    csv_files = glob.glob("*.csv")

    if not csv_files:
        raise FileNotFoundError("No CSV files found in the current directory")

    print("Available dataset files:")
    for i, filename in enumerate(csv_files, 1):
        print(f"{i}. {filename}")

    while True:
        try:
            choice = int(input("Select dataset (1-{}): ".format(len(csv_files))))
            if 1 <= choice <= len(csv_files):
                return csv_files[choice-1]
            print(f"Please enter a number between 1 and {len(csv_files)}")
        except ValueError:
            print("Please enter a valid number")

# Select dataset and load it
dataset_path = select_dataset()
print(f"\nLoading dataset: {dataset_path}")

na_tokens = ['', ' ', 'NA', 'N/A', 'nan', 'NaN', 'None']
df = pd.read_csv(dataset_path, sep=';', na_values=na_tokens)
pix_cols   = ['cam1_tip_x','cam1_tip_y','cam2_tip_x','cam2_tip_y','cam3_tip_x','cam3_tip_y']
angle_cols = ['shoulder_yaw','shoulder_pitch','elbow_pitch','wrist_roll_1','wrist_yaw']

for col in pix_cols + angle_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=pix_cols + angle_cols).reset_index(drop=True)

# ──────────────────────────────────────────────────────────────
# 2.  Scale
# ──────────────────────────────────────────────────────────────
X = df[pix_cols].to_numpy(dtype=np.float32)
y = df[angle_cols].to_numpy(dtype=np.float32)

# pixel x-coords → divide by 639, pixel y-coords → divide by 479
X[:,0::2] /= 640.0   # columns 0,2,4 (all the x’s)
X[:,1::2] /= 480.0   # columns 1,3,5 (all the y’s)

# angles (-1.6 .. +1.6) → divide by 1.6  →  (-1 .. +1)
y /= 1.6

# ──────────────────────────────────────────────────────────────
# 3.  Train / test split
# ──────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)

# ──────────────────────────────────────────────────────────────
# 4.  Build model
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
# 7.  Ready-to-use predictor
# ──────────────────────────────────────────────────────────────
def predict_angles(cam1_xy, cam2_xy, cam3_xy):
    """
    Parameters
    ----------
    cam1_xy, cam2_xy, cam3_xy : (x,y) pixel tuples from 640×480 images.

    Returns
    -------
    np.ndarray shape (5,)  →  angles in original [-1.6, +1.6] radians
    """
    sample = np.array([
        cam1_xy[0], cam1_xy[1],
        cam2_xy[0], cam2_xy[1],
        cam3_xy[0], cam3_xy[1]], dtype=np.float32)[None, :]   # shape (1,6)

    # same scaling as during training
    sample[:,0::2] /= 640.0
    sample[:,1::2] /= 480.0

    pred_scaled = best_model.predict(sample, verbose=0)[0]
    return pred_scaled * 1.6         # back to original units


def build_model(hp):
    """
    hp (HyperParameters) is an object you can ask for integers,
    floats, choices, booleans, etc.  Each call receives a different
    set of values selected by the tuner.
    """
    # ❶ Tunable layer widths
    units1 = hp.Int('units1',  64, 256, step=32)          # 64,96,…,256
    units2 = hp.Int('units2',  64, 256, step=32)
    units3 = hp.Int('units3',  32, 128, step=32) 
    # ❷ Tunable dropout
    dr_rate = hp.Float('drop_rate',  0.0, 0.5, step=0.1)  # 0.0 … 0.5

    # ❸ Tunable L2 regularisation
    l2_val  = hp.Float('l2', 1e-5, 1e-3, sampling='log')

    # ❹ Tunable learning-rate
    lr = hp.Float('lr', 1e-5, 1e-3, sampling='log')

    model = keras.Sequential([
        layers.Dense(units1, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(l2_val),
                    input_shape=(6,)),
        layers.BatchNormalization(),
        layers.Dropout(dr_rate),

        layers.Dense(units2, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(l2_val)),
        layers.BatchNormalization(),
        layers.Dropout(dr_rate),

        layers.Dense(units3, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(l2_val)),
        layers.BatchNormalization(),

        layers.Dense(32, activation='relu'),   # kept fixed
        layers.Dense(5, activation='tanh')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss='mae',
                metrics=['mae'])
    return model



max_trials    = 40     # how many different models to try
executions    = 1      # repetitions per trial (robustness)
directory     = "kt_inverse_kin"    # where results go

# Remove existing tuner directory to start fresh
if os.path.exists(directory):
    shutil.rmtree(directory)
    print(f"Removed existing tuner directory: {directory}")

tuner = kt.RandomSearch(
        build_model,
        objective   = 'val_loss',
        max_trials  = max_trials,
        executions_per_trial = executions,
        directory   = directory,
        project_name= "mae_search",
        overwrite=True)  # Force fresh start

tuner.search_space_summary()   # optional: show what will be searched 

callbacks = [
    keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-5)
]

tuner.search(X_train, y_train,
             validation_split = 0.15,
             epochs           = 300,
             batch_size       = 32,
             callbacks        = callbacks,
             verbose=1) 


best_hp    = tuner.get_best_hyperparameters(1)[0]
best_model = tuner.get_best_models(1)[0]

print("\nBest hyper-parameters found:")
for p in best_hp.values:
    print(f"  {p}: {best_hp.get(p)}")

# Get validation loss from the best model
val_loss = best_model.evaluate(X_test, y_test, verbose=0)[0]

# Extract dataset name (without .csv extension)
dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

# Create safe filename combining dataset name and val_loss
safe_dataset_name = re.sub(r'[^\w\-_]', '_', dataset_name)
val_loss_str = "{:.2f}".format(val_loss).replace('.', '_')
filename = f"best_inverse_kinematics_{safe_dataset_name}_val_loss_{val_loss_str}.keras"

# Save the model with the new filename
best_model.save(filename)

# Make the model globally available for the predict_angles function
model = best_model

print(f"\nModel saved as '{filename}'")
print(f"You can now use predict_angles() function for predictions.")
print(f"Validation loss: {val_loss:.4f}")