from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer
import pandas as pd

# Escalado Min-Max: transforma los valores para que estén dentro del rango [0,1]
def min_max(m):
    scaler = MinMaxScaler()  # Inicializa el escalador Min-Max
    m_scaled = scaler.fit_transform(m)  # Ajusta y transforma los datos
    return pd.DataFrame(m_scaled, columns=m.columns)  # Devuelve un DataFrame con las mismas columnas

# Escalado Z-Score (StandardScaler): centra los datos en media 0 y desviación estándar 1
def z_score(m):
    scaler = StandardScaler()  # Inicializa el escalador estándar
    m_scaled = scaler.fit_transform(m)  # Ajusta y transforma los datos
    return pd.DataFrame(m_scaled, columns=m.columns)  # Devuelve un DataFrame con las mismas columnas

# Escalado robusto: utiliza la mediana y el rango intercuartílico, menos sensible a valores atípicos
def robust_scaler(m):
    scaler = RobustScaler()  # Inicializa el escalador robusto
    m_scaled = scaler.fit_transform(m)  # Ajusta y transforma los datos
    return pd.DataFrame(m_scaled, columns=m.columns)  # Devuelve un DataFrame con las mismas columnas

# Escalado MaxAbs: escala los datos dividiendo entre el valor absoluto máximo de cada característica
def max_abs(m):
    scaler = MaxAbsScaler()  # Inicializa el escalador MaxAbs
    m_scaled = scaler.fit_transform(m)  # Ajusta y transforma los datos
    return pd.DataFrame(m_scaled, columns=m.columns)  # Devuelve un DataFrame con las mismas columnas

# Transformación de potencia (PowerTransformer): aplica la transformación Yeo-Johnson para normalizar la distribución
def power_transformer(m):
    scaler = PowerTransformer(method="yeo-johnson", standardize=True)  # Inicializa el transformador de potencia
    m_scaled = scaler.fit_transform(m)  # Ajusta y transforma los datos
    return pd.DataFrame(m_scaled, columns=m.columns)  # Devuelve un DataFrame con las mismas columnas
