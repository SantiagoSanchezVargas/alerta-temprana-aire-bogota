import pandas as pd
import numpy as np

LOOKBACK = 48  # Ventana de entrada de 48 horas (2 ciclos diarios completos)
HORIZON = 24   # Horizonte de predicción directa de 24 horas

def preparar_dataset(df_long):
    """
    Rellena huecos menores a 3 horas, marca valores imputados 
    y genera características temporales cíclicas (hora y día de la semana).
    """
    print("[2/5] Procesando datos, aplicando interpolación y variables cíclicas...")
    df_procesado_list = []
    
    for estacion, grupo in df_long.groupby('estacion'):
        grupo = grupo.set_index('timestamp').asfreq('h')
        grupo['estacion'] = estacion
        
        # Marcar imputación y rellenar únicamente huecos <= 3 horas
        grupo['pm10_imputado'] = grupo['pm10'].isna().astype(int)
        grupo['pm10'] = grupo['pm10'].interpolate(method='linear', limit=3, limit_direction='both')
        
        # Transformación log1p sobre el objetivo para atenuar asimetría de picos
        grupo['pm10_log'] = np.log1p(grupo['pm10'])
        
        # Variables cíclicas temporales
        horas = grupo.index.hour
        dows = grupo.index.dayofweek
        grupo['hora_sin'] = np.sin(2 * np.pi * horas / 24)
        grupo['hora_cos'] = np.cos(2 * np.pi * horas / 24)
        grupo['dow_sin'] = np.sin(2 * np.pi * dows / 7)
        grupo['dow_cos'] = np.cos(2 * np.pi * dows / 7)
        
        df_procesado_list.append(grupo.reset_index())
        
    return pd.concat(df_procesado_list, ignore_index=True)

def crear_ventanas_deslizantes(df, feature_cols, target_col='pm10_log'):
    """
    Genera tensores X (ventanas pasadas) e y (secuencia futura directa).
    Excluye ventanas con huecos largos no imputados.
    """
    X_list, y_list = [], []
    
    for estacion, grupo in df.groupby('estacion'):
        grupo = grupo.sort_values('timestamp').reset_index(drop=True)
        values = grupo[feature_cols].to_numpy(dtype="float32")
        target = grupo[target_col].to_numpy(dtype="float32")
        
        for t in range(LOOKBACK, len(grupo) - HORIZON + 1):
            x_win = values[t - LOOKBACK : t]
            y_win = target[t : t + HORIZON]
            
            # Descartar ventanas si existe algún NaN no imputado
            if np.isnan(x_win).any() or np.isnan(y_win).any():
                continue
                
            X_list.append(x_win)
            y_list.append(y_win)
            
    print(f" -> Ventanas generadas correctamente: {len(X_list)} muestras.")
    return np.array(X_list), np.array(y_list)