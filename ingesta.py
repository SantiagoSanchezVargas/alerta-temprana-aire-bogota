import pandas as pd
import numpy as np

def cargar_y_limpiar_iboca(excel_path="reporteIboca (1).xls"):
    """
    Lee el archivo Excel de IBOCA, extrae la concentración horaria cruda 
    y transforma el dataset a formato largo (una fila por timestamp y estación).
    """
    print("[1/5] Cargando y limpiando reporte de IBOCA...")
    tables = pd.read_html(excel_path)
    df_raw = tables[1]
    
    # Estandarización de nombres de columnas
    df_raw.columns = [
        'fecha_hora', 'bolivia_conc', 'bolivia_nowcast', 'bolivia_iboca', 
        'ciudad_bolivar_conc', 'ciudad_bolivar_nowcast', 'ciudad_bolivar_iboca'
    ]
    
    # Eliminar filas de resumen del reporte (AVG, Num, Datos [%])
    df_clean = df_raw.iloc[:-3].copy()
    df_clean['fecha_hora'] = pd.to_datetime(df_clean['fecha_hora'])
    df_clean.replace('Sin data', np.nan, inplace=True)
    
    # REGLA DEL DOCUMENTO: Usar SOLO la concentración cruda de PM10/PM2.5
    df_bolivia = df_clean[['fecha_hora', 'bolivia_conc']].rename(
        columns={'fecha_hora': 'timestamp', 'bolivia_conc': 'pm10'}
    )
    df_bolivia['estacion'] = 'bolivia'
    df_bolivia['pm10'] = pd.to_numeric(df_bolivia['pm10'])
    
    df_cb = df_clean[['fecha_hora', 'ciudad_bolivar_conc']].rename(
        columns={'fecha_hora': 'timestamp', 'ciudad_bolivar_conc': 'pm10'}
    )
    df_cb['estacion'] = 'ciudad_bolivar'
    df_cb['pm10'] = pd.to_numeric(df_cb['pm10'])
    
    # Unificación en formato largo
    df_long = pd.concat([df_bolivia, df_cb], ignore_index=True)
    df_long = df_long.sort_values(['estacion', 'timestamp']).reset_index(drop=True)
    
    print(f" -> Carga completada: {len(df_long)} registros en formato largo.")
    return df_long

if __name__ == "__main__":
    df = cargar_y_limpiar_iboca()
    print(df.head())