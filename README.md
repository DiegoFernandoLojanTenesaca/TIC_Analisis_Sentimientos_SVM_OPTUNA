# Bayesian Hyperparameter Optimization of SVM with TPE for Stress Detection in Spanish Social Media

[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Workflow](https://img.shields.io/badge/Workflow-CRISP--DM-blueviolet)](#)
[![Conference](https://img.shields.io/badge/CIT-2026-orange)](https://www.espe.edu.ec/cit-2026/)
[![Publisher](https://img.shields.io/badge/Springer-LNNS-red)](#)

> Code, data and experiments accompanying the paper:
>
> **Bayesian Hyperparameter Optimization of Support Vector Machines Using TPE for Stress Detection in Spanish Social Media During the 2024 Ecuadorian Energy Crisis**
> *Diego F. Lojan T. and Luis Chamba-Eras*
> Universidad Nacional de Loja
> Accepted at **CIT 2026** — *Emerging Research in Intelligent Systems. Proceedings of the CIT 2026, Vols 1–3.* Springer LNNS.
> DOI: *to appear*

---

## Tabla de contenidos

- [Resumen](#resumen)
- [Cómo citar](#cómo-citar)
- [Características principales](#características-principales)
- [Pipeline](#pipeline)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Resultados destacados](#resultados-destacados)
- [Reproducibilidad](#reproducibilidad)
- [Disponibilidad de datos](#disponibilidad-de-datos)
- [Limitaciones](#limitaciones)
- [Trabajo futuro](#trabajo-futuro)
- [Autores](#autores)
- [Agradecimientos](#agradecimientos)
- [Licencia](#licencia)
- [Referencias técnicas](#referencias-técnicas)

---

## Resumen

Este repositorio contiene la implementación completa del estudio sobre detección de estrés psicológico en redes sociales en español durante la crisis energética del Ecuador (abril–diciembre 2024). El trabajo aplica **optimización bayesiana de hiperparámetros (Tree-structured Parzen Estimator, TPE) sobre Máquinas de Soporte Vectorial (SVM)** para clasificación binaria de estrés, validado en dos corpus: 26,600 tweets recolectados de forma ética durante la crisis y 18,515 mensajes del corpus MentalRiskES.

El pipeline integra **CRISP-DM**, embeddings **FastText Skip-gram**, balanceo **SMOTE** y optimización con **Optuna-TPE**, demostrando que modelos clásicos correctamente ajustados pueden competir en eficiencia y precisión con arquitecturas modernas en NLP en español.

---

## Cómo citar

> **Nota:** La cita bibliográfica oficial estará disponible una vez publicado el volumen de Springer LNNS. Mientras tanto se puede citar como:

```bibtex
@inproceedings{lojan2026bayesian,
  author    = {Lojan T., Diego F. and Chamba-Eras, Luis},
  title     = {Bayesian Hyperparameter Optimization of Support Vector Machines Using TPE for Stress Detection in Spanish Social Media During the 2024 Ecuadorian Energy Crisis},
  booktitle = {Emerging Research in Intelligent Systems. Proceedings of the CIT 2026},
  series    = {Lecture Notes in Networks and Systems},
  publisher = {Springer},
  year      = {2026},
  note      = {In press}
}
```

Si usas este código en investigación académica, también puedes utilizar el botón **"Cite this repository"** que aparece en la barra lateral derecha del repositorio en GitHub (basado en el archivo `CITATION.cff`).

---

## Características principales

- **Web scraping ético** con Selenium y Playwright, sujeto a límites estrictos (500 tweets/día, *delays* aleatorios 60–1,200 s, anonimización SHA-256 de identificadores).
- **Pipeline NLP en español**: limpieza, tokenización con NLTK, lematización con Stanza, preservación de marcadores de negación.
- **Embeddings FastText Skip-gram** (128 dimensiones) sobre lemas, con promediado ponderado por IDF y escalado estándar.
- **Balanceo SMOTE** validado por similitud coseno (0.65–0.70) contra muestras reales.
- **Optimización bayesiana** con Optuna-TPE en modo multivariado.
- **Evaluación exhaustiva**: matrices de confusión, curvas ROC y PR, validación cruzada estratificada de 5 folds, análisis fANOVA de importancia de hiperparámetros.

---

## Pipeline

```
[Recolección]        [Limpieza]           [Vectorización]      [Modelado]              [Evaluación]
Selenium/Playwright  Regex + langdetect   FastText 128d        SVM Base                Matrices confusión
Twitter/X            NLTK tokenizer       IDF weighted         SVM + Optuna-TPE        ROC / PR / fANOVA
MentalRiskES (NDA)   Stanza lemmatizer    StandardScaler       5-fold StratifiedCV     Soporte vectorial
                                          SMOTE k=5
```

Sigue la metodología **CRISP-DM** adaptada a 5 fases (excluyendo Deployment): Business Understanding, Data Understanding, Data Preparation, Modeling y Evaluation.

---

## Estructura del proyecto

```
TIC_Analisis_Sentimientos_SVM_OPTUNA/
├── TAREA-1-EXTRACCION-DATOS/
│   ├── Playwright_Scraper/       # Extracción rápida (~100 tweets / 15 s)
│   ├── Selenium_Scraper/         # Alternativa estable (~1.2 tweets / min)
│   └── Twitter_API_Scraper/      # Vía API oficial (opcional, requiere clave)
├── TAREA-2-ANALISIS-LIMPIEZA/
│   └── limpieza_final.ipynb      # Limpieza, normalización, langdetect, SHA-256
├── TAREA-3-EXTRACCION-CARACTERISTICAS/
│   └── procesamiento_textual.ipynb  # Tokenización, lematización, FastText, SMOTE
├── TAREA-4-ENTRENAMIENTO-Y-AJUSTE-HIPERPARAMETROS/
│   └── entrenamiento_modelos.ipynb  # SVM base + Optuna-TPE
├── TAREA-5-METRICAS/
│   └── evaluacion_metricas.ipynb    # Métricas, matrices, curvas ROC/PR
├── CITATION.cff
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Requisitos

| Componente | Especificación recomendada |
|---|---|
| Python | 3.9 o 3.10 |
| RAM | 16 GB (mínimo para embeddings) |
| Almacenamiento | SSD con al menos 5 GB libres |
| CPU | 6+ núcleos (sin GPU requerida) |
| Sistema operativo | Linux, macOS o Windows 10/11 |

Hardware utilizado en el estudio: Intel Core i7-9750H (6 núcleos / 12 hilos a 2.6 GHz), 16 GB DDR4, SSD 446 GB, Windows 11.

---

## Instalación

```bash
git clone https://github.com/DiegoFernandoLojanTenesaca/TIC_Analisis_Sentimientos_SVM_OPTUNA.git
cd TIC_Analisis_Sentimientos_SVM_OPTUNA

python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
.\venv\Scripts\activate

pip install -r requirements.txt
```

Cada carpeta `TAREA-*` puede tener un `requirements.txt` adicional con dependencias específicas (Playwright, Selenium, etc.). Instalarlas al entrar a la carpeta correspondiente.

---

## Uso

### 1. Extracción de datos

```bash
cd TAREA-1-EXTRACCION-DATOS/Playwright_Scraper
pip install -r requirements.txt
python main.py --keywords "palabras_clave" --max_tweets 500
```

El diccionario de 65 palabras clave validado por psicólogo clínico está documentado en el paper (Tabla 1) y en el código fuente.

### 2. Limpieza y preprocesamiento

```bash
cd TAREA-2-ANALISIS-LIMPIEZA
jupyter notebook limpieza_final.ipynb
```

Procesos incluidos: hash SHA-256 de metadatos sensibles, filtrado por idioma con `langdetect`, normalización de texto (preservando tildes y ñ), exportación a CSV anonimizado.

### 3. Extracción de características

```bash
cd TAREA-3-EXTRACCION-CARACTERISTICAS
jupyter notebook procesamiento_textual.ipynb
```

Flujo: tokenización NLTK con stopwords personalizadas, lematización Stanza, embeddings FastText Skip-gram (128d, window=8, min_count=3, epochs=30), promediado IDF-weighted, escalado estándar, balanceo SMOTE (k=5).

> **Nota sobre archivos grandes.** El archivo `fasttext_estres.model.wv.vectors_ngrams.npy` (~1 GB) **no está incluido en este repositorio** por su tamaño. Para regenerarlo basta con re-ejecutar el notebook `procesamiento_textual.ipynb` sobre los datos limpios; alternativamente está disponible en este enlace de Drive: [fasttext_estres model](https://drive.google.com/drive/folders/1FMycNIdyez94pudQLozHRWrVbuPHtAv2?usp=sharing).

### 4. Entrenamiento y optimización

```bash
cd TAREA-4-ENTRENAMIENTO-Y-AJUSTE-HIPERPARAMETROS
jupyter notebook entrenamiento_modelos.ipynb
```

Configuraciones:
- **SVM base**: scikit-learn defaults (RBF, C=1.0, gamma='scale').
- **SVM + Optuna-TPE**: 100 trials (Twitter/X), 50 trials (MentalRiskES), `multivariate=True`, 20 startup trials, seed=42.
- **Validación cruzada**: 5-fold StratifiedKFold.
- **Métrica objetivo**: F1-score promedio.

### 5. Evaluación

```bash
cd TAREA-5-METRICAS
jupyter notebook evaluacion_metricas.ipynb
```

Genera: métricas weighted (accuracy, precision, recall, F1), matrices de confusión absolutas y normalizadas, curvas ROC y Precision-Recall, conteo de support vectors, tiempos de optimización y análisis fANOVA de importancia de hiperparámetros.

---

## Resultados destacados

Comparativa SVM base vs. SVM optimizado con Optuna-TPE (test set, 5-fold cross-validation, desviación estándar < 0.5 % en F1).

| Dataset | Modelo | Accuracy | Precision | Recall | F1-score | Support Vectors |
|---|---|---|---|---|---|---|
| Twitter/X | SVM Base | 91.16 % | 91.14 % | 91.14 % | 91.14 % | 50.9 % |
| Twitter/X | **SVM + Optuna-TPE** | **92.63 %** | **92.80 %** | **92.41 %** | **92.60 %** (+1.46 pp) | **45.2 %** (-11.3 %) |
| MentalRiskES | SVM Base | 95.86 % | 94.13 % | 97.81 % | 95.93 % | 27.0 % |
| MentalRiskES | **SVM + Optuna-TPE** | **97.18 %** | **95.36 %** | **99.17 %** | **97.23 %** (+1.30 pp) | **20.0 %** (-26.0 %) |

Hallazgos principales:

- **Reducción del 62 % en falsos negativos** en MentalRiskES (de 29 a 11 sobre 1,327), lo más relevante en un contexto de *screening* de salud mental.
- **Kernel RBF dominante**: ~90 % de importancia relativa según análisis fANOVA en ambos datasets.
- **Eficiencia mejorada**: reducción de support vectors entre 11.3 % y 26 % sin pérdida de rendimiento predictivo.
- **Optimización en hardware estándar**: ~10 min para Twitter/X (100 trials), ~3.2 h para MentalRiskES (50 trials), sin GPU.

---

## Reproducibilidad

| Aspecto | Valor |
|---|---|
| Semilla aleatoria | `42` (consistente en `train_test_split`, SMOTE, TPE, StratifiedKFold) |
| Particiones | 80 % train / 10 % validación / 10 % test, estratificadas |
| FastText | `vector_size=128`, `window=8`, `min_count=3`, `epochs=30`, `sg=1` |
| SMOTE | `k_neighbors=5`, `sampling_strategy='auto'` |
| Optuna TPE | `multivariate=True`, `n_startup_trials=20`, `consider_prior=True` |
| Hardware referencia | Intel i7-9750H, 16 GB RAM, sin GPU |

Tiempo esperado de ejecución (extremo a extremo): ~6 h para Twitter/X y ~4 h para MentalRiskES.

---

## Disponibilidad de datos

- **Twitter/X (corpus crisis energética 2024):** los textos fueron recolectados mediante *web scraping* ético, anonimizados por hash SHA-256 y generalización geográfica. **No se redistribuye el corpus crudo** por respeto a los Términos de Servicio de X (Twitter) y para preservar la privacidad de los usuarios. Los datos procesados están disponibles bajo solicitud académica formal (`diego.lojan@unl.edu.ec`).
- **MentalRiskES:** corpus propiedad del grupo SINAI-UJA. El acceso requiere acuerdo de confidencialidad directo con los autores: [github.com/sinai-uja/corpusMentalRiskES](https://github.com/sinai-uja/corpusMentalRiskES).

---

## Limitaciones

Documentadas explícitamente en el paper (Sección 5.5):

- Etiquetado por presencia de keywords puede omitir expresiones implícitas de estrés.
- Ausencia de tests estadísticos formales (McNemar) para validar la significancia de las mejoras.
- Corpus Twitter/X limitado a Ecuador en un evento específico; generalización a otras variantes del español requiere adaptación de dominio.
- SMOTE aplicado en espacio de embeddings de 128d (similitud coseno 0.65–0.70 con muestras reales como evidencia mitigatoria).
- No incluye comparación con baselines transformer (BETO, mBERT) — *out of scope* del estudio, reservado para trabajo futuro.

---

## Trabajo futuro

1. Comparación con embeddings basados en transformers (BETO, mBERT, RoBERTa-es) como extractor de features.
2. Capas de explicabilidad (SHAP, LIME) para interpretación clínica de predicciones.
3. Extensión multinacional y multi-crisis (otros eventos en países hispanohablantes).
4. Clasificación *multi-label* para co-ocurrencia de estrés, ansiedad y depresión.
5. Tests estadísticos formales (McNemar, bootstrap con intervalos de confianza).

---

## Autores

**Diego Fernando Lojan Tenesaca** — *Autor principal*
Universidad Nacional de Loja · Carrera de Computación
ORCID: [0009-0003-3882-3889](https://orcid.org/0009-0003-3882-3889)
Email: diego.lojan@unl.edu.ec

**Luis Antonio Chamba-Eras, PhD** — *Director y coautor*
Universidad Nacional de Loja · Director de GITIC
ORCID: [0000-0003-3069-9628](https://orcid.org/0000-0003-3069-9628)
Email: lachamba@unl.edu.ec

---

## Agradecimientos

Este trabajo se deriva del Trabajo de Integración Curricular *"Ajuste de hiperparámetros en SVM con Optuna (TPE) para detectar estrés durante la crisis energética en Ecuador, 2024"* (Universidad Nacional de Loja, 2026).

Desarrollado en el marco del proyecto **32-DI-FEIRNR-2025**: *"Desarrollo de un marco de trabajo de Inteligencia Artificial para integrar y fortalecer las funciones sustantivas en la Universidad Nacional de Loja y su escalabilidad en las Universidades Públicas de Ecuador"*, financiado por la Universidad Nacional de Loja.

Los autores agradecen al psicólogo clínico que contribuyó a la validación experta del diccionario de términos, y al grupo SINAI de la Universidad de Jaén (UJA) por proporcionar acceso al corpus **MentalRiskES** bajo acuerdo formal de confidencialidad.

Cita del corpus:

```bibtex
@inproceedings{marmol-romero-etal-2024-mentalriskes,
  title     = {{MentalRiskES}: A New Corpus for Early Detection of Mental Disorders in {S}panish},
  author    = {M{\'a}rmol-Romero, Ana Mar{\'i}a and Moreno-Mu{\~n}oz, Antonio and Plaza-del-Arco, Flor Miriam and Molina-Gonz{\'a}lez, M. Dolores and Mart{\'i}n-Valdivia, M. Teresa and Ure{\~n}a-L{\'o}pez, L. Alfonso and Montejo-R{\'a}ez, Arturo},
  booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  year      = {2024},
  pages     = {11204--11214},
  publisher = {ELRA and ICCL},
  url       = {https://aclanthology.org/2024.lrec-main.978}
}
```

---

## Licencia

Distribuido bajo licencia MIT. Ver [LICENSE](LICENSE) para detalles.

---

## Referencias técnicas

- [Optuna Documentation](https://optuna.org/)
- [Scikit-learn SVM Guide](https://scikit-learn.org/stable/modules/svm.html)
- [FastText for Text Classification](https://fasttext.cc/)
- [Stanza for Spanish NLP](https://stanfordnlp.github.io/stanza/)
- [SMOTE — imbalanced-learn](https://imbalanced-learn.org/)
- [CRISP-DM Reference](https://www.datascience-pm.com/crisp-dm-2/)
