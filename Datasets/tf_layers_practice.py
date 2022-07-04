import tensorflow as tf

name = 'UEmbed'
dtype = tf.float32
shape = [5, 3]

ret = tf.get_variable(name=name, dtype=dtype, shape=shape,
			initializer=tf.keras.initializers.GlorotNormal(),
			trainable=True)

print(ret)