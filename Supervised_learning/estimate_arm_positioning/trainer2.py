import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU for this script
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"  # ensure XLA JIT is off

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# ──────────────────────────────────────────────────────────────
# 1.  Load and clean  (this is the code you already ran)
# ──────────────────────────────────────────────────────────────
na_tokens = ['', ' ', 'NA', 'N/A', 'nan', 'NaN', 'None']
df = pd.read_csv('dataset_20250821_130514.csv', sep=';', na_values=na_tokens)
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
model = keras.Sequential([
    layers.Dense(128, activation='relu',
                 kernel_regularizer=keras.regularizers.l2(1e-4),
                 input_shape=(6,)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(128, activation='relu',
                 kernel_regularizer=keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(64, activation='relu',
                 kernel_regularizer=keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    
    layers.Dense(32, activation='relu'),
    layers.Dense(5, activation='tanh')
])

model.compile(optimizer=keras.optimizers.Adam(3e-4),
              loss='mae',
              metrics=['mae'])

callbacks = [
    keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-5)
]

# ──────────────────────────────────────────────────────────────
# 5.  Train
# ──────────────────────────────────────────────────────────────
early = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True)

lr_drop = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.5,
                                            patience=10,
                                            min_lr=1e-5)
ckpt = keras.callbacks.ModelCheckpoint(
        filepath='best_inverse_kinematics.keras',   # new TF-2.x format
        monitor='val_loss',
        save_best_only=True,        # only keep the best epoch
        save_weights_only=False)    # entire model, not just weights


history = model.fit(X_train, y_train,
                    epochs=300,
                    batch_size=32,
                    validation_split=0.15,
                    callbacks=[early, ckpt, lr_drop],
                    verbose=1)

# ──────────────────────────────────────────────────────────────
# 6.  Evaluate
# ──────────────────────────────────────────────────────────────
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f'\nTest MAE (scaled) : {mae:.4f}')
print(f'Test MAE (radians): {mae*1.5:.4f}')


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

    pred_scaled = model.predict(sample, verbose=0)[0]
    return pred_scaled * 1.6         # back to original units

# Example usage
ex_angles = predict_angles((607,301), (238,66), (461,144))
print("\nPredicted angles (rad):", ex_angles)