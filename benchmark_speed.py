import numpy as np
import tensorflow as tf
import time

# Create a dummy sensor reading: [Temperature, Humidity, Soil Moisture]
test_input = np.array([[30.0, 50.0, 25.0]], dtype=np.float32)
# Scale input for the quantized integer model
test_input_int8 = np.array([[30, 50, 25]], dtype=np.int8)

def benchmark_tflite(model_path, is_quantized=False):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Warm up the engine
    interpreter.set_tensor(input_details['index'], test_input_int8 if is_quantized else test_input)
    interpreter.invoke()
    
    # Measure 1000 runs
    start_time = time.perf_counter_ns()
    for _ in range(1000):
        interpreter.set_tensor(input_details['index'], test_input_int8 if is_quantized else test_input)
        interpreter.invoke()
    end_time = time.perf_counter_ns()
    
    return (end_time - start_time) / 1000  # Average time per inference in nanoseconds

print("--- BENCHMARKING INFERENCE SPEED ---")
base_speed = benchmark_tflite("baseline_model.tflite", is_quantized=False)
quant_speed = benchmark_tflite("quantized_model.tflite", is_quantized=True)

print(f"Standard Model Speed: {base_speed:.2f} ns per inference")
print(f"Quantized Model Speed: {quant_speed:.2f} ns per inference")
print(f"Speedup Factor:        {base_speed / quant_speed:.2f}x faster")