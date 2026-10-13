# recommendation_system.py
# -----------------------------------------------------------------------------
# Sistema de recomendación de acciones de mantenimiento para transformadores
# de potencia en aceite, basado en K-Vecinos más Cercanos (KNN).
#
# Asignatura : Inteligencia Artificial Aplicada al Mantenimiento (IEA503) - INACAP
# Actividad  : Actividad formativa GitHub Copilot (U3 - S14)
# Autor      : Diego Zamora
#
# Variables de entrada (pruebas de diagnóstico habituales en terreno):
#   - Gases disueltos en aceite (DGA, ppm): H2, CH4, C2H2, C2H4, C2H6, CO
#   - Índice de polarización (PI) del aislamiento - IEEE Std 43
#   - Rigidez dieléctrica del aceite (kV) - IEC 60156 / IEC 60422
#   - Humedad en aceite (ppm) - IEC 60814 / IEC 60422
#   - Temperatura de aceite superior (°C), carga (% nominal) y edad (años)
#
# Salida: acción de mantenimiento recomendada + casos históricos similares.
#
# NOTA DE INGENIERÍA: el modelo apoya la decisión, no la reemplaza. Si una
# medición supera un límite crítico normativo, la recomendación se fuerza a
# "intervención correctiva" sin importar lo que prediga el modelo.
# -----------------------------------------------------------------------------

# Importar las librerías necesarias
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

RNG_SEED = 42
DATA_PATH = Path(__file__).parent / "data" / "transformadores.csv"
DOCS_PATH = Path(__file__).parent / "docs"

GASES = ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO"]
OTRAS = ["PI", "rigidez_kV", "humedad_ppm", "temp_aceite_C", "carga_pct", "edad_anios"]
DERIVADAS = ["TDCG", "indice_DGA"]
FEATURES = GASES + OTRAS + DERIVADAS

ACCIONES = {
    0: "Operación normal - mantenimiento de rutina anual",
    1: "Preventivo reforzado - tratamiento de aceite y nueva medición en 6 meses",
    2: "Predictivo - DGA mensual y pruebas eléctricas completas (FRA, tan δ)",
    3: "Correctiva inmediata - evaluar retiro de servicio e inspección interna",
}

# Límites por gas según IEEE C57.104-2008 (Condición 1, 2 y 3; sobre C3 = C4)
LIMITES_DGA = {
    "H2":   (100, 700, 1800),
    "CH4":  (120, 400, 1000),
    "C2H2": (1, 9, 35),
    "C2H4": (50, 100, 200),
    "C2H6": (65, 100, 150),
    "CO":   (350, 570, 1400),
}


# -----------------------------------------------------------------------------
# 1. Criterio experto (etiquetado) - define la "verdad" del dataset sintético
# -----------------------------------------------------------------------------
def condicion_gas(gas, ppm):
    """Retorna condición IEEE C57.104 (1 a 4) para un gas."""
    c1, c2, c3 = LIMITES_DGA[gas]
    if ppm <= c1:
        return 1
    if ppm <= c2:
        return 2
    if ppm <= c3:
        return 3
    return 4


def agregar_derivadas(df):
    """Variables de ingeniería que resumen el DGA para el modelo.

    TDCG       : gases combustibles totales disueltos (IEEE C57.104).
    indice_DGA : peor gas relativo a su límite de Condición 3 (>1 = Condición 4).
    """
    df = df.copy()
    df["TDCG"] = df[GASES].sum(axis=1)
    df["indice_DGA"] = np.max(
        [df[g] / LIMITES_DGA[g][2] for g in GASES], axis=0).round(3)
    return df


def severidad_indicadores(row):
    """Severidad (1 = bueno ... 4 = crítico) de cada indicador de una unidad."""
    sev = [condicion_gas(g, row[g]) for g in GASES]

    # Índice de polarización - IEEE 43: <1 peligroso, 1-2 dudoso, 2-4 bueno
    pi = row["PI"]
    sev.append(4 if pi < 1.0 else 3 if pi < 1.5 else 2 if pi < 2.0 else 1)

    # Rigidez dieléctrica (equipos <= 72,5 kV, criterio basado en IEC 60422)
    kv = row["rigidez_kV"]
    sev.append(4 if kv < 30 else 3 if kv < 40 else 2 if kv < 50 else 1)

    # Humedad en aceite
    h = row["humedad_ppm"]
    sev.append(4 if h > 40 else 3 if h > 30 else 2 if h > 20 else 1)

    # Temperatura de aceite superior (alarma típica 90-95 °C)
    t = row["temp_aceite_C"]
    sev.append(4 if t > 105 else 3 if t > 95 else 2 if t > 85 else 1)
    return sev


def accion_experta(row):
    sev = severidad_indicadores(row)
    if max(sev) == 4:
        return 3
    if sev.count(3) >= 1 or sev.count(2) >= 3:
        return 2
    if sev.count(2) >= 1 or row["edad_anios"] > 30 or row["carga_pct"] > 100:
        return 1
    return 0


def limite_critico_superado(row):
    """Guardia de seguridad: indicadores que por sí solos exigen intervención."""
    alarmas = []
    for g in GASES:
        if row[g] > LIMITES_DGA[g][2]:
            alarmas.append(f"{g} = {row[g]:.0f} ppm (> {LIMITES_DGA[g][2]} ppm, Condición 4 IEEE C57.104)")
    if row["PI"] < 1.0:
        alarmas.append(f"PI = {row['PI']:.2f} (< 1,0: aislamiento comprometido, IEEE 43)")
    if row["rigidez_kV"] < 30:
        alarmas.append(f"Rigidez dieléctrica = {row['rigidez_kV']:.0f} kV (< 30 kV)")
    if row["humedad_ppm"] > 40:
        alarmas.append(f"Humedad = {row['humedad_ppm']:.0f} ppm (> 40 ppm)")
    if row["temp_aceite_C"] > 105:
        alarmas.append(f"Temperatura de aceite = {row['temp_aceite_C']:.0f} °C (> 105 °C)")
    return alarmas


# -----------------------------------------------------------------------------
# 2. Generación del dataset (simula un historial de flota de transformadores)
# -----------------------------------------------------------------------------
def generar_dataset(n=1500, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    edad = rng.uniform(1, 45, n)
    carga = np.clip(rng.normal(70, 18, n), 20, 130)
    envejecimiento = edad / 45  # 0..1

    # Los gases siguen distribuciones log-normales que crecen con edad y carga
    def gas(base, sigma, factor):
        return rng.lognormal(np.log(base) + factor * envejecimiento + 0.01 * (carga - 70), sigma, n)

    df = pd.DataFrame({
        "H2": gas(40, 0.9, 1.2),
        "CH4": gas(35, 0.9, 1.3),
        "C2H2": np.where(rng.random(n) < 0.75, 0.0, gas(2, 1.4, 1.5)),
        "C2H4": gas(15, 1.0, 1.5),
        "C2H6": gas(20, 0.9, 1.2),
        "CO": gas(180, 0.6, 0.9),
        "PI": np.clip(rng.normal(3.0 - 1.4 * envejecimiento, 0.6, n), 0.6, 5.0),
        "rigidez_kV": np.clip(rng.normal(62 - 18 * envejecimiento, 9, n), 15, 85),
        "humedad_ppm": np.clip(rng.normal(10 + 18 * envejecimiento, 6, n), 2, 60),
        "temp_aceite_C": np.clip(45 + 0.45 * carga + rng.normal(0, 6, n), 30, 120),
        "carga_pct": carga,
        "edad_anios": edad,
    }).round(2)

    df["accion"] = df.apply(accion_experta, axis=1)

    # 3 % de ruido de etiqueta: dos especialistas no siempre coinciden
    ruido = rng.random(n) < 0.03
    df.loc[ruido, "accion"] = np.clip(
        df.loc[ruido, "accion"] + rng.choice([-1, 1], ruido.sum()), 0, 3)

    df.insert(0, "id_equipo", [f"TR-{i:04d}" for i in range(1, n + 1)])
    return df


# -----------------------------------------------------------------------------
# 3. Cargar los datos
# -----------------------------------------------------------------------------
def cargar_datos():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generar_dataset().to_csv(DATA_PATH, index=False)
        print(f"Dataset generado en {DATA_PATH}")
    return agregar_derivadas(pd.read_csv(DATA_PATH))


# -----------------------------------------------------------------------------
# 4. Modelo: preprocesamiento + KNN
# -----------------------------------------------------------------------------
def construir_modelo():
    # Los gases son muy asimétricos (ppm de 0 a miles): log1p los normaliza.
    # Luego se escalan TODAS las variables: KNN usa distancias, y sin escalar
    # el CO (cientos de ppm) dominaría sobre el PI (valores de 1 a 5).
    preproc = ColumnTransformer([
        ("gases", Pipeline([("log", FunctionTransformer(np.log1p)),
                            ("esc", StandardScaler())]), GASES),
        ("otras", StandardScaler(), OTRAS),
        ("derivadas", Pipeline([("log", FunctionTransformer(np.log1p)),
                                ("esc", StandardScaler())]), DERIVADAS),
    ])
    return Pipeline([("prep", preproc), ("knn", KNeighborsClassifier())])


def entrenar(df):
    # Preprocesamiento de datos: características y etiqueta
    X = df[FEATURES]
    y = df["accion"]

    # División estratificada: mantiene la proporción de casos críticos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RNG_SEED, stratify=y)

    # Búsqueda del mejor k y ponderación con validación cruzada (5 pliegues)
    busqueda = GridSearchCV(
        construir_modelo(),
        {"knn__n_neighbors": [3, 5, 7, 9, 11, 15], "knn__weights": ["uniform", "distance"]},
        cv=StratifiedKFold(5, shuffle=True, random_state=RNG_SEED),
        scoring="f1_macro",
    )
    busqueda.fit(X_train, y_train)
    modelo = busqueda.best_estimator_
    print(f"Mejores hiperparámetros: {busqueda.best_params_}")

    # Realizar predicciones y evaluar el modelo
    y_pred = modelo.predict(X_test)
    nombres = ["0-Normal", "1-Preventivo", "2-Predictivo", "3-Correctiva"]
    print("\n--- Modelo KNN por sí solo ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f} %")
    print(f"F1 macro : {f1_score(y_test, y_pred, average='macro'):.3f}")

    # Sistema completo = modelo + guardia de límites críticos normativos
    critico = X_test.apply(lambda r: bool(limite_critico_superado(r)), axis=1).values
    y_sis = np.where(critico, 3, y_pred)
    print("\n--- Sistema completo (KNN + guardia normativa) ---")
    print(f"Accuracy : {accuracy_score(y_test, y_sis) * 100:.2f} %")
    print(f"F1 macro : {f1_score(y_test, y_sis, average='macro'):.3f}")
    print("\nReporte por clase (sistema completo):")
    print(classification_report(y_test, y_sis, target_names=nombres, digits=3))

    cm = confusion_matrix(y_test, y_sis)
    print("Matriz de confusión (filas = real, columnas = predicho):")
    print(pd.DataFrame(cm, index=[f"real {i}" for i in range(4)],
                       columns=[f"pred {i}" for i in range(4)]))
    guardar_matriz(cm)
    return modelo, X_train, y_train


def guardar_matriz(cm):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    DOCS_PATH.mkdir(exist_ok=True)
    etiquetas = ["Normal", "Preventivo", "Predictivo", "Correctiva"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), etiquetas, rotation=30)
    ax.set_yticks(range(4), etiquetas)
    ax.set_xlabel("Acción predicha")
    ax.set_ylabel("Acción real (criterio experto)")
    ax.set_title("Matriz de confusión - sistema KNN + guardia")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(DOCS_PATH / "matriz_confusion.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 5. Función de recomendación
# -----------------------------------------------------------------------------
def recommend(modelo, X_train, y_train, mediciones, n_similares=3):
    """Recomienda una acción de mantenimiento para un transformador.

    mediciones: dict con las 12 mediciones (GASES + OTRAS).
    Retorna un dict con la acción, la confianza, alarmas y casos similares.
    """
    x = agregar_derivadas(pd.DataFrame([mediciones]))[FEATURES]
    pred = int(modelo.predict(x)[0])
    prob = float(modelo.predict_proba(x)[0].max())

    # Guardia de seguridad: los límites normativos prevalecen sobre el modelo
    alarmas = limite_critico_superado(x.iloc[0])
    if alarmas and pred < 3:
        pred, prob = 3, 1.0

    # Casos históricos más parecidos (explicabilidad para el planificador)
    x_t = modelo.named_steps["prep"].transform(x)
    dist, idx = modelo.named_steps["knn"].kneighbors(x_t, n_neighbors=n_similares)
    similares = X_train.iloc[idx[0]].assign(
        accion=y_train.iloc[idx[0]].values, distancia=dist[0].round(3))

    return {"accion": ACCIONES[pred], "codigo": pred, "confianza": prob,
            "alarmas": alarmas, "similares": similares}


def imprimir_recomendacion(nombre, r):
    print(f"\n=== {nombre} ===")
    print(f"Acción recomendada : [{r['codigo']}] {r['accion']}")
    print(f"Confianza          : {r['confianza'] * 100:.0f} %")
    for a in r["alarmas"]:
        print(f"  ALARMA: {a}")
    print("Casos históricos más similares:")
    print(r["similares"][["H2", "C2H2", "C2H4", "PI", "rigidez_kV", "edad_anios",
                          "accion", "distancia"]].to_string())


# -----------------------------------------------------------------------------
# 6. Ejemplo de uso
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    datos = cargar_datos()
    print(f"Registros: {len(datos)}")
    print("Distribución de acciones:\n", datos["accion"].value_counts().sort_index(), "\n")

    modelo, X_train, y_train = entrenar(datos)

    casos = {
        "TR-A: 10 MVA, 8 años, pruebas en buen estado": dict(
            H2=35, CH4=30, C2H2=0, C2H4=12, C2H6=18, CO=160, PI=3.2,
            rigidez_kV=64, humedad_ppm=9, temp_aceite_C=68, carga_pct=60, edad_anios=8),
        "TR-B: 25 años, gases de sobrecalentamiento en aumento": dict(
            H2=180, CH4=260, C2H2=0.5, C2H4=85, C2H6=90, CO=420, PI=1.8,
            rigidez_kV=44, humedad_ppm=24, temp_aceite_C=88, carga_pct=92, edad_anios=25),
        "TR-C: acetileno alto (posible arco interno)": dict(
            H2=650, CH4=150, C2H2=48, C2H4=95, C2H6=40, CO=300, PI=2.4,
            rigidez_kV=52, humedad_ppm=14, temp_aceite_C=72, carga_pct=70, edad_anios=14),
    }
    for nombre, med in casos.items():
        imprimir_recomendacion(nombre, recommend(modelo, X_train, y_train, med))
