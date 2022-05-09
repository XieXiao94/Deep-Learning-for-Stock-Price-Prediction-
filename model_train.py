import tensorflow as tf

batch_size = 32


physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

def model_train(model, X_train, y_train, X_val, y_val,EP_NUM):

    # callback = tf.keras.callbacks.ModelCheckpoint('Transformer+TimeEmbedding_new.hdf5',
    #                                               monitor='val_loss',
    #                                               save_best_only=True, verbose=1)
    with tf.device('/GPU:0'):
        history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=EP_NUM,
                        # callbacks=[callback],
                        validation_data=(X_val, y_val))

        model.save('Models/my_model.h5')
    return history