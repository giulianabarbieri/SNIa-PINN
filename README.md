# SNIa-PINN: Reconstrucción Física de Curvas de Luz para Supernovas Tipo Ia

Este repositorio contiene el desarrollo del proyecto final para el curso de Ecuaciones Diferenciales y Machine Learning (2026), dictado por Facundo Sapienza. 

El objetivo principal es implementar **Redes Neuronales Informadas por la Física (PINNs)** y **Ecuaciones Diferenciales Universales (UDEs)** para abordar el problema de los *gaps* observacionales y la baja cadencia en las series temporales de transientes astronómicos, utilizando datos del broker **ALeRCE** en el marco del ecosistema de **DeepRubin-Explorer**.

## 🌌 El Problema Científico
Las curvas de luz de las Supernovas Tipo Ia (SNIa) suelen presentar muestreos irregulares debido a limitaciones observacionales (condiciones climáticas, rotación de telescopios, etc.). Mientras que los métodos tradicionales como los Procesos Gaussianos (GP) realizan una interpolación puramente estadística, este enfoque propone utilizar la **estructura dinámica de las explosiones estelares** como un ancla para guiar el aprendizaje del modelo en los intervalos sin datos.

## 🛠️ Enfoque Metodológico
* **Estructura Física:** Se utiliza una ecuación diferencial ordinaria (EDO) basada en el balance energético y el decaimiento radiactivo del Níquel-56 ($^{56}Ni$) para modelar la tasa de cambio de la luminosidad.
* **Aproximador Universal:** Una red neuronal se integra dentro del sistema dinámico para aprender componentes no lineales complejos y difíciles de parametrizar analíticamente (como la evolución de la opacidad del eyecta).
* **Función de Pérdida Híbrida:** Combina la fidelidad de los datos observados por el broker ALeRCE con el residuo de la EDO, penalizando aquellas soluciones que violen las leyes de conservación de la energía.

## 🚀 Cómo Empezar

### 1. Requisitos e Instalación
Para instalar las dependencias necesarias, ejecuta el siguiente comando en tu entorno virtual:
```bash
pip install -r requirements.txt
```

### 2. Descarga de Datos (SNIa)
Para iniciar la ingesta y descarga de curvas de luz de Supernovas Tipo Ia desde el broker ALeRCE, ejecuta:
```bash
python3 dataset/loader.py
```
Los datos descargados en formato CSV se guardarán automáticamente en la carpeta `dataset/`.