import pandas as pd, numpy as np
import joblib

csv_path = "dataset_20250821_115734.csv"
columns   = ["shoulder_yaw","shoulder_pitch","elbow_pitch",
              "wrist_pitch","wrist_roll",
              "cam1_tip_x","cam1_tip_y",
              "cam2_tip_x","cam2_tip_y",
              "cam3_tip_x","cam3_tip_y"]

df = pd.read_csv(csv_path, sep=';')       # because your file is ';'-separated
df.columns = columns

X = df[["shoulder_yaw","shoulder_pitch","elbow_pitch",
        "wrist_pitch","wrist_roll"]].values.astype(np.float32)

Y = df[["cam1_tip_x","cam1_tip_y",
        "cam2_tip_x","cam2_tip_y",
        "cam3_tip_x","cam3_tip_y"]].values.astype(np.float32)

# Check for NaN values and handle them
print("Original data shape:", X.shape)
print("NaN values in X:", np.isnan(X).sum())
print("NaN values in Y:", np.isnan(Y).sum())

# Remove rows with NaN values
valid_rows = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
X = X[valid_rows]
Y = Y[valid_rows]

print("After removing NaN rows:")
print("Data shape:", X.shape)
print("NaN values in X:", np.isnan(X).sum())
print("NaN values in Y:", np.isnan(Y).sum())

# ➋   Standardise (zero mean, unit std).  VERY important for NNs
X_mean, X_std = X.mean(0), X.std(0)
Y_mean, Y_std = Y.mean(0), Y.std(0)

Xn = (X - X_mean) / X_std
Yn = (Y - Y_mean) / Y_std



from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

X_train, X_test, Y_train, Y_test = train_test_split(
        Xn, Yn, test_size=0.15, random_state=0)

mlp = MLPRegressor(hidden_layer_sizes=(256,256),
                   activation='tanh',
                   max_iter=4000,
                   learning_rate_init=1e-3,
                   verbose=True)
mlp.fit(X_train, Y_train)

pred = mlp.predict(X_test)
pixel_mae  = mean_absolute_error(Y_test * Y_std + Y_mean,
                                 pred    * Y_std + Y_mean)
print("mean abs pixel error per view:", pixel_mae)

# Save the trained model
model_filename = "arm_position_estimator_model.pkl"
joblib.dump(mlp, model_filename)
print(f"Model saved to {model_filename}")

# Also save the standardization parameters for later use
scaler_params = {
    'X_mean': X_mean,
    'X_std': X_std,
    'Y_mean': Y_mean,
    'Y_std': Y_std
}
scaler_filename = "scaler_params.pkl"
joblib.dump(scaler_params, scaler_filename)
print(f"Scaler parameters saved to {scaler_filename}")