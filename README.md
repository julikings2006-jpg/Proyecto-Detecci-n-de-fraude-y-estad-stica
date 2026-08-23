<img align="right" src="https://count.getloli.com/get/@:FraudDetectionML?theme=moebooru">

## 🛡️ Financial Fraud Detection System — Probabilistic Machine Learning <img src="https://github.com/Aurorp1g/Aurorp1g/raw/main/cartoon.gif" alt="Hi" width="70" />

<p align="center">
  <a href="https://readme-typing-svg.herokuapp.com">
    <img src="https://readme-typing-svg.herokuapp.com?font=Futura&color=00F5FF&size=24&center=true&vCenter=true&width=1100&height=40&lines=Credit+Card+Fraud+Detection+Analysis+%F0%9F%92%B3;Probabilistic+Foundations+of+Machine+Learning+%F0%9F%A7%A0;Unraveling+Extreme+Class+Imbalance+(0.173%25)+%F0%9F%94%8D;Bayesian+Inference+%7C+KL+Divergence+%7C+Cross+Entropy+%F0%9F%93%88">
  </a>
</p>

### 📌 Resumen del Proyecto <picture style="margin-right: 10px;"><img src="https://github.com/Aurorp1g/Aurorp1g/raw/main/about_me.gif" width="40" alt="About Project"></picture>

<div><img align="right" alt="GIF" src="https://github.com/Aurorp1g/Aurorp1g/raw/main/cartoon.webp" width="280" height="auto" /></div>

- 💳 **Escenario Profesional:** Consultoría estadística para una red de tarjetas de crédito.
- 🎯 **Objetivo:** Auditar y demostrar por qué las métricas de "precisión" tradicionales son engañosas bajo desbalance extremo.
- 📉 **Dataset:** Transacciones de tarjetahabientes europeos (284,807 operaciones, 492 fraudes = $0.173\%$).
- 🧠 **Enfoque:** Evaluación rigurosa mediante 11 conceptos probabilísticos e inferencia bayesiana.
- 🚀 **Resultado:** Identificación de componentes latentes altamente informativas ($V_{17}, V_{12}, V_{16}$) y optimización del Valor Predictivo Positivo (Posterior).
- **Data obtenida de:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

### 🛠️ Tecnologías y Librerías <picture style="margin-right: 10px;"><img src="https://github.com/Aurorp1g/Aurorp1g/raw/main/Software_Tools.gif" width="60" alt="Tools"></picture>

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/-Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/-SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c?style=flat&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/-Seaborn-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

### 📊 Hallazgos Clave & Mapeo Probabilístico 📊 <picture style="margin-right: 10px;"><img src="https://github.com/Aurorp1g/Aurorp1g/raw/main/Statistics.gif" width="30" alt="Stats"></picture>

| Concepto Probabilístico | Aplicación Cuantitativa en el Proyecto |
| :--- | :--- |
| **Probabilidad Condicional** | $P(\text{Fraude} \mid \text{Amount} > X)$ cae en tramos altos ($0.132\%$ en $>\$10$ a $0\%$ en $>\$10,000$). |
| **Teorema de Bayes** | Una regla sobre $V_{14} < -6$ eleva la probabilidad posterior a $70.30\%$ (multiplica la sospecha $407\times$). |
| **Verosimilitud (MLE)** | Ajuste Gaussiano en $V_{14}$ muestra desplazamiento significativo ($\mu_{\text{legítima}}=0.012$ vs $\mu_{\text{fraude}}=-6.972$). |
| **Distribuciones Paramétricas** | Se rechaza normalidad en $V_{17}, V_{12}, V_{16}$ ($p \approx 0$); evidencia de colas pesadas en Q-Q Plots. |
| **Esperanza y Varianza** | Mediana en fraude ($\$9.25$) es menor que en legítimas ($\$22.00$), pero el promedio es mayor ($\$122.21$) por asimetría. |
| **Independencia y Correlación**| Ortogonalidad perfecta en componentes PCA ($r=0.0000$); `Amount` ($r=0.0056$) no predice fraude. |
| **Prior y Posterior** | Un clasificador con $99\%$ sensibilidad/especificidad produce solo $14.6\%$ posterior bajo el prior real de $0.173\%$. |
| **Entropía** | Entropía de clase $H(\text{Class}) = 0.018$ bits; la baja entropía global oculta la extrema dificultad del problema. |
| **Entropía Cruzada** | Modelo balanceado logra Log Loss de $0.093$ (train) y $0.096$ (test), demostrando alta capacidad de generalización. |
| **Divergencia KL** | Jerarquización de variables más informativas: $V_{17}$ ($KL=6.91$), $V_{12}$ ($6.76$), $V_{16}$ ($6.27$), $V_{14}$ ($6.02$). |

---

### 🚨 La Paradoja de la Precisión y Falsos Positivos

<div><img align="right" alt="GIF" src="https://github.com/Aurorp1g/Aurorp1g/raw/main/Right_Side.gif" width="280" height="auto" /></div>

Un clasificador trivial que clasifique **TODAS** las transacciones como legítimas obtendría una **precisión/exactitud del 99.827%**, pero sería completamente inútil para el negocio al no detectar ningún fraude.

Al calcular el **Teorema de Bayes**:
$$\text{Posterior} = \frac{P(\text{Test}^+ \mid \text{Fraude}) \cdot P(\text{Fraude})}{P(\text{Test}^+)}$$

Incluso con pruebas diagnósticas altamente específicas ($99\%$), el **prior extremo del 0.173%** genera aproximadamente 6 falsas alarmas por cada fraude real detectado ($14.6\%$ de Valor Predictivo Positivo).

---

### 📂 Estructura del Repositorio

```bash
├── notebooks/
│   └── ML_Unsupervised_Fraud.ipynb # Notebook con desarrollo de los 11 conceptos
├── README.md                   # Documentación del proyecto
