import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_lstm_model(n_features, lookback=48, horizon=24, units=64, dropout=0.2, lr=1e-3, loss_type="huber"):
    """
    Construye y compila el modelo LSTM compacto para predicción multisalida de 24 horas.
    """
    inp = keras.Input(shape=(lookback, n_features))
    x = layers.LSTM(units)(inp)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(horizon)(x)
    
    # Selección de función de pérdida
    if loss_type == "pinball":
        # Pérdida asimétrica (tau = 0.7): penaliza más la subestimación de picos de contaminación
        def pinball_loss(tau=0.7):
            def loss(y_true, y_pred):
                e = y_true - y_pred
                return tf.reduce_mean(tf.maximum(tau * e, (tau - 1) * e))
            return loss
        selected_loss = pinball_loss(tau=0.7)
    else:
        # Huber Loss: robusta frente a ruido atípico en los sensores
        selected_loss = keras.losses.Huber(delta=1.0)
        
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=selected_loss,
        metrics=["mae"],
    )
    return model