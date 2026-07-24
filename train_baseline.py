print("--- SCRIPT STARTED SUCCESSFULLY ---")
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

print("TensorFlow Version:", tf.__version__)

# 1. Generate Synthetic Edge Sensor Data
def generate_sensor_data(num_samples=5000):
    np.random.seed(42)
    
    # Simulate realistic micro-climate readings
    temperature = np.random.uniform(15, 45, num_samples)     # 15°C to 45°C
    humidity = np.random.uniform(20, 90, num_samples)        # 20% to 90% RH
    soil_moisture = np.random.uniform(10, 80, num_samples)   # 10% to 80% capacity
    
    # Define automation logic: Open valve (1) if soil is dry and it's hot/dry air
    # Otherwise, keep valve closed (0)
    valve_status = np.zeros(num_samples)
    for i in range(num_samples):
        if soil_moisture[i] < 30 or (soil_moisture[i] < 45 and temperature[i] > 35):
            valve_status[i] = 1
            
    # Structure into a DataFrame
    df = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'soil_moisture': soil_moisture,
        'valve_status': valve_status
    })
    return df

# Load and split data
data = generate_sensor_data()
X = data[['temperature', 'humidity', 'soil_moisture']].values
y = data['valve_status'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build a Resource-Constrained "TinyML" Architecture
# We keep the layers small so it can eventually fit in microcontroller SRAM
model = models.Sequential([
    layers.Input(shape=(3,)),                 # 3 Sensor inputs
    layers.Dense(12, activation='relu'),      # Hidden layer 1
    layers.Dense(8, activation='relu'),       # Hidden layer 2
    layers.Dense(1, activation='sigmoid')     # Binary Output (0 or 1)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 3. Train the Model
print("\n--- Training Baseline Model ---")
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Evaluate performance on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nBaseline Test Accuracy: {accuracy * 100:.2f}%")

# Save the baseline model
model.save("baseline_model.h5")
print("Baseline model saved successfully as 'baseline_model.h5'")