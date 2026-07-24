import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import os

print("--- STARTING OPTIMIZATION PIPELINE ---")

# 1. Rebuild the exact same TinyML model structure directly in memory
# This avoids the Keras version deserialization error entirely
def build_clean_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(3,)),                 
        tf.keras.layers.Dense(12, activation='relu'),      
        tf.keras.layers.Dense(8, activation='relu'),       
        tf.keras.layers.Dense(1, activation='sigmoid')     
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create a fresh instance of our model
baseline_model = build_clean_model()

# 2. Prepare Representative Data for Quantization
np.random.seed(42)
representative_data = np.random.uniform(15, 80, (100, 3)).astype(np.float32)

def representative_data_gen():
    for i in range(100):
        yield [representative_data[i:i+1]]

# --- TECHNIQUE A: POST-TRAINING INT8 QUANTIZATION ---
converter = tf.lite.TFLiteConverter.from_keras_model(baseline_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

quantized_tflite = converter.convert()

with open("quantized_model.tflite", "wb") as f:
    f.write(quantized_tflite)

# --- TECHNIQUE B: WEIGHT PRUNING (SPARSITY) ---
# We build a fresh model to apply pruning cleanly
pruning_model_base = build_clean_model()

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, begin_step=0, frequency=100)
}

pruned_model = prune_low_magnitude(pruning_model_base, **pruning_params)

# Standard TFLite baseline converter for comparison
converter_baseline = tf.lite.TFLiteConverter.from_keras_model(baseline_model)
baseline_tflite = converter_baseline.convert()
with open("baseline_model.tflite", "wb") as f:
    f.write(baseline_tflite)

# Pruned converter
converter_pruned = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
pruned_tflite = converter_pruned.convert()
with open("pruned_model.tflite", "wb") as f:
    f.write(pruned_tflite)

print("\n--- OPTIMIZATION COMPLETE ---")

# --- DATA COLLECTION FOR YOUR PAPER ---
def get_file_size(file_path):
    return os.path.getsize(file_path) / 1024  # Size in KB

print(f"Standard TFLite Model Size: {get_file_size('baseline_model.tflite'):.2f} KB")
print(f"Quantized Model Size:     {get_file_size('quantized_model.tflite'):.2f} KB")
print(f"Pruned Model Size:        {get_file_size('pruned_model.tflite'):.2f} KB")