import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Función para balancear con ADASYN (Adaptive Synthetic Sampling)
# Genera nuevas muestras sintéticas de la clase minoritaria adaptándose a la distribución de los datos.
def balanceo_adasyn(m, y):
    adasyn = ADASYN(random_state=42)  # Inicializa el método ADASYN con semilla fija para reproducibilidad
    X_res, y_res = adasyn.fit_resample(m, y)  # Ajusta y genera nuevas muestras balanceadas

    # Muestra la distribución original y la nueva distribución después del balanceo
    print("Distribución original:\n", y.value_counts())
    print("\nDistribución balanceada con ADASYN:\n", y_res.value_counts())
    return X_res, y_res  # Devuelve las nuevas características y etiquetas balanceadas

# Función para balancear con Random UnderSampler (RUS)
# Reduce aleatoriamente la clase mayoritaria para igualar la cantidad de muestras con la minoritaria.
def balanceo_rus(m, y):
    rus = RandomUnderSampler(random_state=42)  # Inicializa el método RUS con semilla fija
    X_res, y_res = rus.fit_resample(m, y)  # Ajusta y elimina muestras de la clase mayoritaria

    # Muestra la distribución original y la nueva distribución después del balanceo
    print("Distribución original:\n", y.value_counts())
    print("\nDistribución balanceada con RUS:\n", y_res.value_counts())
    return X_res, y_res  # Devuelve las nuevas características y etiquetas balanceadas

# Función para balancear con SMOTE (Synthetic Minority Over-sampling Technique)
# Genera nuevas muestras sintéticas de la clase minoritaria interpolando entre vecinos cercanos.
def balanceo_smote(m, y):
    smote = SMOTE(random_state=42)  # Inicializa el método SMOTE con semilla fija
    X_res, y_res = smote.fit_resample(m, y)  # Ajusta y genera nuevas muestras sintéticas

    # Muestra la distribución original y la nueva distribución después del balanceo
    print("Distribución original:\n", y.value_counts())
    print("\nDistribución balanceada con SMOTE:\n", y_res.value_counts())
    return X_res, y_res  # Devuelve las nuevas características y etiquetas balanceadas
