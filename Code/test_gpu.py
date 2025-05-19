import tensorflow as tf

# Create a simple matrix multiplication operation
with tf.device('/GPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)

print("TensorFlow is running on:", tf.config.list_physical_devices('GPU'))
