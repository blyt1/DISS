import tensorflow as tf


def create_model(input_size, output_size):
    """
    Create a simple feedforward neural network model.
    Arguments:
    - input_size: The size of the input features.
    - output_size: The size of the output.
    Returns:
    - model: The created model.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=64, activation='relu', input_shape=(input_size,)))
    model.add(tf.keras.layers.Dense(units=output_size, activation='sigmoid'))
    return model


def train_model(X, Y, input_size, output_size, learning_rate, num_epochs):
    """
    Train the model using labeled data.
    Arguments:
    - X: The input features.
    - Y: The true labels.
    - input_size: The size of the input features.
    - output_size: The size of the output.
    - learning_rate: The learning rate for the optimizer.
    - num_epochs: The number of training epochs.
    Returns:
    - model: The trained model.
    - losses: A list of the average losses for each epoch.
    """
    model = create_model(input_size, output_size)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    losses = []

    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            logits = model(X, training=True)
            loss_value = tf.reduce_mean(tf.losses.binary_crossentropy(Y, logits))

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        losses.append(loss_value.numpy())

    return model, losses


