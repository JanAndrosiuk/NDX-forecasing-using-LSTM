from preprocessing import *
from splitWindows import *
from buildModel import *
from tqdm import tqdm
import numpy as np
from tensorflow.keras.callbacks import History


def model_fit_predict():
    """
    Training example was implemented according to machine-learning-mastery forum
    The function takes data from the dictionary returned from splitWindows.create_windows function
    https://machinelearningmastery.com/stateful-stateless-lstm-time-series-forecasting-python/
    :param X: input matrix
    :param y: target vector
    :param test_input: test input matrix
    :return: np.array of predictions
    """
    # Load preprocessed dataframe
    df = transform_target()

    # Generate windows, and load train input data, train target data, and test input data
    windows_dict = create_windows(
        df[features].values,  # input matrix
        df['Close'].values.reshape(df.shape[0], -1),
        df['Date'].values,
        df['target'].values
    )
    X, y, test_input = windows_dict['X'], windows_dict['y'], windows_dict['X_test']
    del df

    # Predictions are stored in a list
    predictions = []

    with tqdm(total=X.shape[0], desc="Training model, saving predictions") as progress_bar:

        # Save model History in order to check error data
        history = History()

        # build model framework
        current_model = model_builder(X)

        # Make predictions for each window
        for i in range(X.shape[0]):

            # TRAIN (FIT) model for each epoch
            # history = current_model.fit(
            #     input_X[i], target_X[i],
            #     epochs=_epochs, batch_size=batch,
            #     verbose=0, shuffle=False, validation_split=0.1,
            #     callbacks=[history]
            # )
            for e in range(epochs):
                current_model.fit(
                    X[i], y[i],
                    epochs=1, batch_size=batch,
                    verbose=0, shuffle=False,
                    callbacks=[history]
                )
                current_model.reset_states()

            # PREDICT and save results
            predictions.append(
                current_model.predict(test_input[i], batch_size=batch, verbose=0)
            )

            progress_bar.update(1)

    return np.asarray(predictions)
