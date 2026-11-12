import numpy as np
from tensorflow import keras
from ingesta import cargar_y_limpiar_iboca
from procesamiento import preparar_dataset, crear_ventanas_deslizantes, LOOKBACK, HORIZON
from modelo import build_lstm_model

def ejecutar_entrenamiento():
    print("[3/5] Iniciando proceso de entrenamiento...")
    
    # 1. Cargar y procesar
    df_long = cargar_y_limpiar_iboca()
    df_dataset = preparar_dataset(df_long)
    
    # 2. Selección de características de entrada
    feature_cols = ['pm10_log', 'pm10_imputado', 'hora_sin', 'hora_cos', 'dow_sin', 'dow_cos']
    X, y = crear_ventanas_deslizantes(df_dataset, feature_cols)
    
    if len(X) == 0:
        raise ValueError("No se generaron ventanas válidas. Revisa los datos de entrada.")
        
    # 3. División cronológica (80% Entrenamiento, 20% Validación)
    split_idx = int(len(X) * 0.8)
    X_tr, X_va = X[:split_idx], X[split_idx:]
    y_tr, y_va = y[:split_idx], y[split_idx:]
    
    # 4. Crear modelo y Callbacks
    model = build_lstm_model(n_features=X.shape[2], lookback=LOOKBACK, horizon=HORIZON)
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)
    ]
    
    # 5. Ajustar modelo
    print("[4/5] Entrenando la red LSTM...")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=50,
        batch_size=64,
        shuffle=True, # Permitido barajar dentro de la partición de entrenamiento
        callbacks=callbacks
    )
    
    # Guardar modelo entrenado
    model.save("modelo_lstm_calidad_aire.h5")
    print(" -> Modelo guardado como 'modelo_lstm_calidad_aire.h5'.")
    return model, df_dataset, feature_cols

if __name__ == "__main__":
    ejecutar_entrenamiento()