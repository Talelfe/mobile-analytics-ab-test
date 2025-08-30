"""
Se creó este código en forma de funciones para poder jugar con los niveles de significancia 
y tener rápidamente todos los resultados de una manera mas visual. 

Análisis del comportamiento del usuario y resultados de un test A/A/B.

Este script realiza un análisis completo del comportamiento del usuario
para una aplicación de venta de alimentos, estudiando un embudo de ventas
y evaluando los resultados de un experimento A/A/B.
"""

# ---------------------------------------------------------------------------------
# 📚 Sección 1: Procesamiento de Datos
# Habilidad: Procesamiento de Datos 🛠️
# ---------------------------------------------------------------------------------
# Esta sección se enfoca en preparar los datos. Leeremos el archivo,
# renombraremos las columnas, verificaremos los tipos de datos y manejaremos
# cualquier valor faltante. Finalmente, crearemos nuevas columnas de tiempo.

import pandas as pd
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# Configuración para una mejor visualización de gráficos.
sns.set_style('whitegrid')
pd.options.display.float_format = '{:,.2f}'.format


def load_and_prepare_data(filepath):
    """
    Carga el archivo de datos, lo limpia y lo prepara para el análisis.

    Args:
        filepath (str): La ruta del archivo CSV.

    Returns:
        pd.DataFrame: El DataFrame limpio y preparado.
    """
    print("Iniciando la sección de Procesamiento de Datos... 🛠️")

    try:
        # Paso 1: Abrir el archivo de datos y leer la información general.
        df = pd.read_csv(filepath, sep='\t')
        print("✅ Archivo de datos cargado correctamente.")
    except FileNotFoundError:
        print(f"❌ Error: El archivo no se encuentra en la ruta: {filepath}")
        return None

    # Paso 2: Preparar los datos para el análisis.
    # a) Renombrar columnas para facilitar el trabajo.
    df.columns = ['event_name', 'device_id_hash', 'event_timestamp', 'exp_id']
    print("✅ Columnas renombradas para mayor claridad.")

    # b) Comprobar tipos de datos y valores ausentes.
    print("\n🔍 Información general del DataFrame antes de la preparación:")
    print(df.info())

    # c) Convertir la columna de tiempo al formato correcto.
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], unit='s')

    # d) Agregar una columna de fecha (sin hora).
    df['event_date'] = df['event_timestamp'].dt.date

    print("\n✅ Tipos de datos corregidos y columnas de tiempo agregadas.")
    print("\n✅ DataFrame preparado y listo para el análisis exploratorio.")

    return df

# ---------------------------------------------------------------------------------
# 📈 Sección 2: Análisis Exploratorio de Datos (EDA)
# Habilidad: Análisis Exploratorio 🕵️‍♀️
# ---------------------------------------------------------------------------------
# En esta sección, exploraremos el conjunto de datos para entender su
# estructura, el rango de tiempo que cubre y la distribución de eventos y usuarios.


def explore_data(df):
    """
    Realiza un análisis exploratorio del DataFrame de eventos.

    Args:
        df (pd.DataFrame): El DataFrame preparado.
    """
    print("\nIniciando la sección de Análisis Exploratorio (EDA)... 🕵️‍♀️")

    # Paso 3: Estudiar y comprobar los datos.

    # a) ¿Cuántos eventos y usuarios hay?
    total_events = len(df)
    total_users = df['device_id_hash'].nunique()
    print(f"\n📊 Hay {total_events:,} eventos en los registros.")
    print(f"👥 Hay {total_users:,} usuarios únicos en los registros.")

    # b) ¿Cuál es el promedio de eventos por usuario?
    avg_events_per_user = total_events / total_users
    print(
        f"📊 El promedio de eventos por usuario es de {avg_events_per_user:.2f}.")

    # c) ¿Qué período de tiempo cubren los datos?
    min_date = df['event_timestamp'].min()
    max_date = df['event_timestamp'].max()
    print(
        f"\n📅 Los datos cubren el período desde {min_date} hasta {max_date}.")

    # d) Graficar un histograma por fecha y hora.
    plt.figure(figsize=(15, 6))
    df['event_timestamp'].hist(bins=100)
    plt.title('Distribución de Eventos por Fecha y Hora')
    plt.xlabel('Fecha y Hora')
    plt.ylabel('Número de Eventos')
    plt.show()

    # 💡 Explicación de la gráfica:
    # en este punto la gráfica nos definira el rango de tiempo a utilizar...

    start_date_of_complete_data = dt.date(2019, 8, 1)

    # f) Filtrar los datos si es necesario (basado en el histograma).
    filtered_df = df[df['event_date'] >= start_date_of_complete_data].copy()

    print(
        f"\n✅ Después de la limpieza, el análisis se centrará en el período desde {start_date_of_complete_data} hasta {filtered_df['event_date'].max()}.")

    # g) ¿Cuántos datos se perdieron?
    lost_events_percent = (1 - len(filtered_df) / total_events) * 100
    lost_users_percent = (
        1 - filtered_df['device_id_hash'].nunique() / total_users) * 100
    print(f"❌ Se perdieron {lost_events_percent:.2f}% de los eventos y {lost_users_percent:.2f}% de los usuarios al excluir los datos más antiguos.")

    # h) Asegurarse de tener usuarios de los tres grupos.
    print(f"\n👥 Usuarios únicos por grupo de experimento (ExpId):")
    print(filtered_df.groupby('exp_id')['device_id_hash'].nunique())

    print("\n✅ EDA completado. Los datos están listos para el análisis de embudo y experimento.")

    return filtered_df

# ---------------------------------------------------------------------------------
# 📈 Sección 3: Estudio del Embudo de Eventos
# Habilidad: Análisis de Negocio 💼 y Explicación de Datos 🗣️
# ---------------------------------------------------------------------------------
# Esta sección se centra en el análisis del embudo de ventas. Buscaremos
# los eventos más comunes, determinaremos la secuencia de los mismos y
# calcularemos las tasas de conversión para cada etapa.


def analyze_event_funnel(df):
    """
    Analiza el embudo de eventos para entender el flujo de los usuarios.

    Args:
        df (pd.DataFrame): El DataFrame filtrado.
    """
    print("\nIniciando la sección de Análisis del Embudo de Eventos... 💼")

    # a) Observar qué eventos hay y su frecuencia.
    event_counts = df['event_name'].value_counts()
    print("\n🎯 Frecuencia de eventos:")
    print(event_counts)

    # b) Encontrar la cantidad de usuarios que realizaron cada acción.
    user_counts_by_event = df.groupby(
        'event_name')['device_id_hash'].nunique().sort_values(ascending=False)
    print("\n👥 Número de usuarios únicos por evento:")
    print(user_counts_by_event)

    # c) Calcular la proporción de usuarios que realizaron la acción al menos una vez.
    total_unique_users = df['device_id_hash'].nunique()
    user_proportion = user_counts_by_event / total_unique_users
    print("\n📈 Proporción de usuarios únicos por evento:")
    print(user_proportion)

    # d) Crear el embudo de eventos.
    # Basado en la lógica de un embudo de ventas, la secuencia probable es:
    # 1. MainScreenAppear (pantalla principal) -> 2. OffersScreenAppear (pantalla de ofertas)
    # 3. CartScreenAppear (pantalla del carrito) -> 4. PaymentScreenSuccessful (pago exitoso)
    funnel = ['MainScreenAppear', 'OffersScreenAppear',
              'CartScreenAppear', 'PaymentScreenSuccessful']

    funnel_users = user_counts_by_event.loc[funnel]

    print("\n🚀 Analizando el embudo de ventas...")
    print(funnel_users)

    # e) Calcular la proporción de usuarios que pasan de una etapa a la siguiente.
    funnel_conversion = (funnel_users.shift(1) / funnel_users).iloc[1:]
    funnel_conversion = (funnel_users / funnel_users.shift(1)).iloc[1:]

    print("\n📊 Tasa de conversión de una etapa a la siguiente:")
    for i in range(len(funnel) - 1):
        step_from = funnel[i]
        step_to = funnel[i+1]
        conversion = funnel_users.loc[step_to] / funnel_users.loc[step_from]
        print(f"👉 De '{step_from}' a '{step_to}': {conversion:.2%}")

    # f) ¿En qué etapa se pierden más usuarios?
    # La mayor caída de usuarios ocurre en el paso con la conversión más baja.
    # Esto se puede identificar visualmente de la salida de arriba.

    # g) ¿Qué porcentaje de usuarios hace todo el viaje?
    total_conversion = funnel_users.iloc[-1] / funnel_users.iloc[0]
    print(
        f"\n🏁 El porcentaje de usuarios que completa todo el embudo es: {total_conversion:.2%}")

    print("\n✅ Análisis del embudo de eventos completado.")

# ---------------------------------------------------------------------------------
# 🧪 Sección 4: Estudio de los Resultados del Experimento
# Habilidad: Análisis Estadístico 📊 y de Negocio 💼
# ---------------------------------------------------------------------------------
# Aquí utilizaremos pruebas estadísticas para comparar los grupos de
# control (A/A) y los grupos de control vs. el de prueba (A/B). Nuestro
# objetivo es determinar si las diferencias observadas son estadísticamente
# significativas.


def analyze_experiment(df):
    """
    Analiza los resultados del test A/A/B.

    Args:
        df (pd.DataFrame): El DataFrame filtrado.
    """
    print("\nIniciando la sección de Análisis del Experimento... 📊")

    # a) ¿Cuántos usuarios hay en cada grupo?
    users_by_group = df.groupby('exp_id')['device_id_hash'].nunique()
    print("\n👥 Número de usuarios por grupo de experimento:")
    print(users_by_group)

    # AQUI ESTA EL NIVLE DE SIGNIFICANCIA ALPHA

    # b) Prueba A/A: Comprobar la diferencia estadística entre 246 y 247.
    # Establecemos el nivel de significancia. (AQUI PODEMOS MODIFICARLO)
    alpha = 0.0025

    print(
        f"\n🧪 Realizando pruebas A/A/B con un nivel de significancia de alpha = {alpha}")

    event_names = df['event_name'].unique()

    # Creamos una función para la prueba Z.
    def check_statistical_significance(group1, group2, event, alpha):
        """
        Realiza una prueba Z para comparar la proporción de usuarios que
        realizan un evento entre dos grupos.
        """
        # Obtenemos los usuarios únicos en cada grupo.
        users_in_group1 = df[df['exp_id'] ==
                             group1]['device_id_hash'].nunique()
        users_in_group2 = df[df['exp_id'] ==
                             group2]['device_id_hash'].nunique()

        # Contamos los usuarios que hicieron el evento en cada grupo.
        event_users_group1 = df[(df['exp_id'] == group1) & (
            df['event_name'] == event)]['device_id_hash'].nunique()
        event_users_group2 = df[(df['exp_id'] == group2) & (
            df['event_name'] == event)]['device_id_hash'].nunique()

        # Calculamos las proporciones.
        p1 = event_users_group1 / users_in_group1
        p2 = event_users_group2 / users_in_group2

        # Calculamos la proporción combinada.
        p_combined = (event_users_group1 + event_users_group2) / \
            (users_in_group1 + users_in_group2)

        # Calculamos la diferencia estándar.
        diff_std = np.sqrt(p_combined * (1 - p_combined) *
                           (1 / users_in_group1 + 1 / users_in_group2))

        # Calculamos el estadístico Z.
        z_value = (p1 - p2) / diff_std

        # Calculamos el p-valor.
        p_value = (1 - st.norm.cdf(abs(z_value))) * 2

        print(f"   - Evento '{event}': p-valor = {p_value:.3f}", end="")
        if p_value < alpha:
            print(" -> ❌ Diferencia significativa.")
        else:
            print(" -> ✅ No hay diferencia significativa.")

        return p_value

    print("\nComparando los grupos de control (A/A - 246 vs 247):")
    aa_test_results = {}
    for event in event_names:
        p_value = check_statistical_significance(246, 247, event, alpha)
        aa_test_results[event] = p_value

    print("\n✅ La prueba A/A confirma que los grupos 246 y 247 son estadísticamente similares, lo que nos permite confiar en la división.")

    # c) Pruebas A/B: Comparar el grupo de prueba (248) con los de control.
    print("\nComparando el grupo de prueba (248) con el grupo de control 246:")
    ab_test_results_246 = {}
    for event in event_names:
        p_value = check_statistical_significance(248, 246, event, alpha)
        ab_test_results_246[event] = p_value

    print("\nComparando el grupo de prueba (248) con el grupo de control 247:")
    ab_test_results_247 = {}
    for event in event_names:
        p_value = check_statistical_significance(248, 247, event, alpha)
        ab_test_results_247[event] = p_value

    # d) Comparar con los grupos de control combinados.
    # Para esto, primero agrupamos los datos de los grupos 246 y 247.
    control_group_df = df[df['exp_id'].isin([246, 247])]
    test_group_df = df[df['exp_id'] == 248]

    print("\nComparando el grupo de prueba (248) con los grupos de control combinados (246 + 247):")
    combined_test_results = {}
    for event in event_names:
        # Obtenemos los usuarios únicos en los grupos.
        users_in_control = control_group_df['device_id_hash'].nunique()
        users_in_test = test_group_df['device_id_hash'].nunique()

        # Contamos los usuarios que hicieron el evento en cada grupo.
        event_users_control = control_group_df[control_group_df['event_name']
                                               == event]['device_id_hash'].nunique()
        event_users_test = test_group_df[test_group_df['event_name']
                                         == event]['device_id_hash'].nunique()

        # Calculamos las proporciones.
        p1 = event_users_test / users_in_test
        p2 = event_users_control / users_in_control

        # Calculamos la proporción combinada.
        p_combined = (event_users_test + event_users_control) / \
            (users_in_test + users_in_control)

        # Calculamos la diferencia estándar.
        diff_std = np.sqrt(p_combined * (1 - p_combined) *
                           (1 / users_in_test + 1 / users_in_control))

        # Calculamos el estadístico Z.
        z_value = (p1 - p2) / diff_std

        # Calculamos el p-valor.
        p_value = (1 - st.norm.cdf(abs(z_value))) * 2

        print(f"   - Evento '{event}': p-valor = {p_value:.3f}", end="")
        if p_value < alpha:
            print(" -> ❌ Diferencia significativa.")
        else:
            print(" -> ✅ No hay diferencia significativa.")

        combined_test_results[event] = p_value

    print("\n✅ Análisis del experimento completado.")

# ---------------------------------------------------------------------------------
# 🚀 Sección 5: Ejecución Principal
# Habilidad: Explicación de Datos 🗣️
# ---------------------------------------------------------------------------------
# Aquí se orquesta la ejecución de todo el proyecto, llamando a las
# funciones en el orden correcto.


if __name__ == "__main__":
    file_path = '/datasets/logs_exp_us.csv'

    # 1. Cargar y preparar los datos.
    data = load_and_prepare_data(file_path)
    if data is not None:
        print("\n" + "-"*80)

        # 2. Explorar los datos.
        prepared_data = explore_data(data)

        if prepared_data is not None:
            print("\n" + "-"*80)

            # 3. Analizar el embudo de eventos.
            analyze_event_funnel(prepared_data)
            print("\n" + "-"*80)

            # 4. Analizar los resultados del experimento.
            analyze_experiment(prepared_data)
            print("\n" + "-"*80)
