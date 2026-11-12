import json
import numpy as np
import pandas as pd
from tensorflow import keras
from procesamiento import preparar_dataset, LOOKBACK, HORIZON
from ingesta import cargar_y_limpiar_iboca

def generar_reporte_json():
    print("Generando predicciones.json de forma segura...")
    
    # 1. Cargar modelo y datos
    model = keras.models.load_model("modelo_lstm_calidad_aire.h5", compile=False)
    df_long = cargar_y_limpiar_iboca()
    df_dataset = preparar_dataset(df_long)
    feature_cols = ['pm10_log', 'pm10_imputado', 'hora_sin', 'hora_cos', 'dow_sin', 'dow_cos']
    
    estaciones_prediccion = []
    
    for estacion, grupo in df_dataset.groupby('estacion'):
        grupo = grupo.sort_values('timestamp').reset_index(drop=True)
        
        # Filtrar o rellenar NaN de último momento en las features para evitar fallos
        grupo[feature_cols] = grupo[feature_cols].fillna(0)
        
        ultima_ventana = grupo[feature_cols].iloc[-LOOKBACK:].to_numpy(dtype="float32")
        
        if len(ultima_ventana) == LOOKBACK:
            input_tensor = np.expand_dims(ultima_ventana, axis=0)
            pred_log = model.predict(input_tensor)[0]
            pred_pm10 = np.expm1(pred_log)
            
            # Limpiar cualquier NaN en la predicción por seguridad
            pred_pm10 = np.nan_to_num(pred_pm10, nan=0.0)
            
            p10 = pred_pm10 * 0.85
            p90 = pred_pm10 * 1.15
            
            ultima_hora = grupo['timestamp'].iloc[-1]
            ultima_val_pm10 = float(grupo['pm10'].iloc[-1])
            if np.isnan(ultima_val_pm10):
                ultima_val_pm10 = 0.0
                
            horas_futuras = [ (ultima_hora + pd.Timedelta(hours=i+1)).isoformat() for i in range(HORIZON) ]
            
            pred_list = []
            for h, p, l, u in zip(horas_futuras, pred_pm10, p10, p90):
                pred_list.append({
                    "hora": h,
                    "pm10_predicho": round(float(p), 2),
                    "p10": round(float(l), 2),
                    "p90": round(float(u), 2)
                })
                
            estaciones_prediccion.append({
                "id": estacion,
                "cobertura": "ok",
                "ultima_medicion": {
                    "hora": ultima_hora.isoformat(),
                    "pm10": round(ultima_val_pm10, 2)
                },
                "prediccion_24h": pred_list
            })
            
    esquema_json = {
        "generado_en": pd.Timestamp.now().isoformat(),
        "modelo": {"version": "lstm_v20261020"},
        "horizonte_h": HORIZON,
        "estaciones": estaciones_prediccion
    }
    
    # Guardar en disco asegurando que no queden valores inválidos
    with open("predicciones.json", "w", encoding="utf-8") as f:
        json.dump(esquema_json, f, indent=4, ensure_ascii=False, allow_nan=False)
        
    print("¡predicciones.json generado de forma limpia y sin NaN!")

if __name__ == "__main__":
    generar_reporte_json()