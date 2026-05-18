# Bayesian Hyperparameter Optimization of SVM with TPE for Stress Detection in Spanish Social Media

[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Workflow](https://img.shields.io/badge/Workflow-CRISP--DM-blueviolet)](#)
[![Conference](https://img.shields.io/badge/CIT-2026-orange)](https://www.espe.edu.ec/cit-2026/)
[![Publisher](https://img.shields.io/badge/Springer-LNNS-red)](#)
[![Status](https://img.shields.io/badge/Status-In%20Press-yellow)](#)

> [!IMPORTANT]
> ### Paper aceptado · CIT 2026 · Springer LNNS
>
> **Bayesian Hyperparameter Optimization of Support Vector Machines Using TPE for Stress Detection in Spanish Social Media During the 2024 Ecuadorian Energy Crisis**
>
> **Diego F. Lojan T.** y **Luis Chamba-Eras**
> Universidad Nacional de Loja
>
> Aceptado en **CIT 2026** — *Emerging Research in Intelligent Systems. Proceedings of the CIT 2026, Vols 1–3*. Springer Lecture Notes in Networks and Systems.
>
> **Estado actual:** En producción editorial. **DOI:** pendiente de asignación oficial.

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

Este repositorio contiene la **implementación completa** del estudio sobre detección de estrés psicológico en redes sociales en español durante la **crisis energética del Ecuador (abril–diciembre 2024)**. El trabajo aplica **optimización bayesiana de hiperparámetros (Tree-structured Parzen Estimator, TPE)** sobre **Máquinas de Soporte Vectorial (SVM)** para clasificación binaria de estrés, validado en dos corpus:

- **26,600 tweets** recolectados de forma ética durante la crisis.
- **18,515 mensajes** del corpus MentalRiskES.

El pipeline integra **CRISP-DM**, embeddings **FastText Skip-gram**, balanceo **SMOTE** y optimización con **Optuna-TPE**, demostrando que modelos clásicos correctamente ajustados pueden competir en eficiencia y precisión con arquitecturas modernas en NLP en español.

---

## Cómo citar

> [!WARNING]
> ### Cita oficial **no disponible todavía**
>
> El paper se encuentra **en producción editorial** en Springer. La cita bibliográfica formal (con páginas, volumen y DOI definitivos) **estará disponible una vez publicado** el volumen de Springer LNNS, previsto para 2026.
>
> Esta sección se actualizará automáticamente con la cita oficial y el DOI cuando el volumen aparezca indexado en SpringerLink.

**Cita provisional** (sujeta a cambios — usar solo si es estrictamente necesario citarlo antes de la publicación oficial):

```bibtex
@inproceedings{lojan2026bayesian,
  author    = {Lojan T., Diego F. and Chamba-Eras, Luis},
  title     = {Bayesian Hyperparameter Optimization of Support Vector Machines
               Using {TPE} for Stress Detection in Spanish Social Media
               During the 2024 Ecuadorian Energy Crisis},
  booktitle = {Emerging Research in Intelligent Systems.
               Proceedings of the CIT 2026},
  series    = {Lecture Notes in Networks and Systems},
  publisher = {Springer},
  year      = {2026},
  note      = {In press}
}
```

> [!TIP]
> Cuando el repositorio incluya el archivo `CITATION.cff`, GitHub mostrará automáticamente un botón **"Cite this repository"** en la barra lateral derecha del repo, con la cita lista para copiar en BibTeX o APA.

---

## Características principales

| Componente | Descripción |
|---|---|
| **Web scraping ético** | Selenium y Playwright con límites estrictos: 500 tweets/día, *delays* aleatorios 60–1,200 s, anonimización **SHA-256** de identificadores. |
| **Pipeline NLP en español** | Limpieza, tokenización con **NLTK**, lematización con **Stanza**, preservación de marcadores de negación. |
| **Embeddings** | **FastText Skip-gram** 128d sobre lemas, promediado ponderado por **IDF** y escalado estándar. |
| **Balanceo de clases** | **SMOTE** validado por similitud coseno (0.65–0.70) contra muestras reales. |
| **Optimización bayesiana** | **Optuna-TPE** en modo multivariado, 100/50 trials. |
| **Evaluación** | Matrices de confusión, curvas ROC y PR, **5-fold StratifiedKFold**, análisis **fANOVA** de importancia de hiperparámetros. |

---

## Pipeline

```text
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   RECOLECCIÓN    │ →  │     LIMPIEZA     │ →  │  VECTORIZACIÓN   │ →  │     MODELADO     │ →  │    EVALUACIÓN    │
├──────────────────┤    ├──────────────────┤    ├──────────────────┤    ├──────────────────┤    ├──────────────────┤
│ Selenium         │    │ Regex            │    │ FastText 128d    │    │ SVM Base         │    │ Confusión        │
│ Playwright       │    │ langdetect       │    │ IDF weighted     │    │ SVM + Optuna-TPE │    │ ROC / PR         │
│ Twitter/X        │    │ NLTK tokenizer   │    │ StandardScaler   │    │ 5-fold CV        │    │ fANOVA           │
│ MentalRiskES NDA │    │ Stanza lemmatizer│    │ SMOTE k=5        │    │ seed=42          │    │ Support vectors  │
└──────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘
```

Sigue la metodología **CRISP-DM** adaptada a **5 fases** (excluyendo Deployment): Business Understanding, Data Understanding, Data Preparation, Modeling y Evaluation.

---

## Estructura del proyecto

```text
TIC_Analisis_Sentimientos_SVM_OPTUNA/
├── TAREA-1-EXTRACCION-DATOS/
│   ├── Playwright_Scraper/       Extracción rápida (~100 tweets / 15 s)
│   ├── Selenium_Scraper/         Alternativa estable (~1.2 tweets / min)
│   └── Twitter_API_Scraper/      Vía API oficial (opcional, requiere clave)
├── TAREA-2-ANALISIS-LIMPIEZA/
│   └── limpieza_final.ipynb      Limpieza, normalización, langdetect, SHA-256
├── TAREA-3-EXTRACCION-CARACTERISTICAS/
│   └── procesamiento_textual.ipynb   Tokenización, lematización, FastText, SMOTE
├── TAREA-4-ENTRENAMIENTO-Y-AJUSTE-HIPERPARAMETROS/
│   └── entrenamiento_modelos.ipynb   SVM base + Optuna-TPE
├── TAREA-5-METRICAS/
│   └── evaluacion_metricas.ipynb     Métricas, matrices, curvas ROC/PR
├── CITATION.cff
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Requisitos

| Componente | Especificación recomendada |
|---|---|
| **Python** | 3.9 o 3.10 |
| **RAM** | 16 GB (mínimo para embeddings) |
| **Almacenamiento** | SSD con al menos 5 GB libres |
| **CPU** | 6+ núcleos (sin GPU requerida) |
| **Sistema operativo** | Linux, macOS o Windows 10/11 |

**Hardware utilizado en el estudio:** Intel Core i7-9750H (6 núcleos / 12 hilos a 2.6 GHz), 16 GB DDR4, SSD 446 GB, Windows 11.

---

## Instalación

```bash
git clone https://github.com/DiegoFernandoLojanTenesaca/TIC_Analisis_Sentimientos_SVM_OPTUNA.git
cd TIC_Analisis_Sentimientos_SVM_OPTUNA

python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
.\venv\Scripts\activate

pip install -r requirements.txt
```

> [!NOTE]
> Cada carpeta `TAREA-*` puede tener su propio `requirements.txt` con dependencias específicas (Playwright, Selenium, etc.). Conviene instalarlas al entrar a la carpeta correspondiente.

---

## Uso

### 1. Extracción de datos

```bash
cd TAREA-1-EXTRACCION-DATOS/Playwright_Scraper
pip install -r requirements.txt
python main.py --keywords "palabras_clave" --max_tweets 500
```

El **diccionario de 65 palabras clave** validado por psicólogo clínico está documentado en el paper (Tabla 1) y en el código fuente.

### 2. Limpieza y preprocesamiento

```bash
cd TAREA-2-ANALISIS-LIMPIEZA
jupyter notebook limpieza_final.ipynb
```

**Procesos incluidos:**
- Hash **SHA-256** de metadatos sensibles.
- Filtrado por idioma con `langdetect`.
- Normalización de texto preservando **tildes y ñ**.
- Exportación a CSV anonimizado.

### 3. Extracción de características

```bash
cd TAREA-3-EXTRACCION-CARACTERISTICAS
jupyter notebook procesamiento_textual.ipynb
```

**Flujo:** Tokenización NLTK con stopwords personalizadas → lematización Stanza → embeddings FastText Skip-gram (128d, window=8, min_count=3, epochs=30) → promediado IDF-weighted → escalado estándar → balanceo SMOTE (k=5).

> [!CAUTION]
> **Archivo grande no incluido en el repositorio.**
>
> El archivo `fasttext_estres.model.wv.vectors_ngrams.npy` (~1 GB) **no se versiona en GitHub** por su tamaño.
>
> **Dos formas de obtenerlo:**
> 1. **Regenerarlo** re-ejecutando el notebook `procesamiento_textual.ipynb` sobre los datos limpios.
> 2. **Descargarlo** desde Drive: [`fasttext_estres model`](https://drive.google.com/drive/folders/1FMycNIdyez94pudQLozHRWrVbuPHtAv2?usp=sharing).

### 4. Entrenamiento y optimización

```bash
cd TAREA-4-ENTRENAMIENTO-Y-AJUSTE-HIPERPARAMETROS
jupyter notebook entrenamiento_modelos.ipynb
```

| Configuración | Valor |
|---|---|
| **SVM base** | scikit-learn defaults (`RBF`, `C=1.0`, `gamma='scale'`) |
| **SVM + Optuna-TPE** | 100 trials (Twitter/X), 50 trials (MentalRiskES) |
| **Sampler TPE** | `multivariate=True`, 20 startup trials, `seed=42` |
| **Validación cruzada** | 5-fold `StratifiedKFold` |
| **Métrica objetivo** | F1-score promedio |

### 5. Evaluación

```bash
cd TAREA-5-METRICAS
jupyter notebook evaluacion_metricas.ipynb
```

**Outputs generados:**
- Métricas weighted: **accuracy, precision, recall, F1**.
- Matrices de confusión absolutas y normalizadas.
- Curvas **ROC** y **Precision-Recall**.
- Conteo de **support vectors**.
- Tiempos de optimización.
- Análisis **fANOVA** de importancia de hiperparámetros.

---

## Resultados destacados

Comparativa **SVM base** vs. **SVM optimizado con Optuna-TPE** sobre el conjunto de test, validación cruzada de 5 folds, **desviación estándar < 0.5 %** en F1.

| Dataset | Modelo | Accuracy | Precision | Recall | F1-score | Support Vectors |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Twitter/X | SVM Base | 91.16 % | 91.14 % | 91.14 % | 91.14 % | 50.9 % |
| Twitter/X | **SVM + Optuna-TPE** | **92.63 %** | **92.80 %** | **92.41 %** | **92.60 %** `+1.46 pp` | **45.2 %** `−11.3 %` |
| MentalRiskES | SVM Base | 95.86 % | 94.13 % | 97.81 % | 95.93 % | 27.0 % |
| MentalRiskES | **SVM + Optuna-TPE** | **97.18 %** | **95.36 %** | **99.17 %** | **97.23 %** `+1.30 pp` | **20.0 %** `−26.0 %` |

### Hallazgos principales

> [!TIP]
> **Reducción del 62 % en falsos negativos** en MentalRiskES (de 29 a 11 sobre 1,327 casos).
> Es el resultado **más relevante en un contexto de *screening* de salud mental**, donde cada caso no detectado tiene implicaciones reales.

- **Kernel RBF dominante:** ~90 % de importancia relativa según análisis **fANOVA** en ambos datasets.
- **Eficiencia mejorada:** reducción de support vectors entre **11.3 %** y **26 %** sin pérdida de rendimiento predictivo.
- **Optimización en hardware estándar:** ~10 min (Twitter/X, 100 trials) y ~3.2 h (MentalRiskES, 50 trials), **sin GPU**.

---

## Reproducibilidad

| Aspecto | Valor |
|---|---|
| **Semilla aleatoria** | `42` (consistente en `train_test_split`, SMOTE, TPE, StratifiedKFold) |
| **Particiones** | **80 %** train · **10 %** validación · **10 %** test, estratificadas |
| **FastText** | `vector_size=128`, `window=8`, `min_count=3`, `epochs=30`, `sg=1` |
| **SMOTE** | `k_neighbors=5`, `sampling_strategy='auto'` |
| **Optuna TPE** | `multivariate=True`, `n_startup_trials=20`, `consider_prior=True` |
| **Hardware referencia** | Intel i7-9750H · 16 GB RAM · sin GPU |

**Tiempo esperado de ejecución de extremo a extremo:** ~6 h para Twitter/X y ~4 h para MentalRiskES.

---

## Disponibilidad de datos

> [!IMPORTANT]
> Los datos crudos **no se redistribuyen públicamente** por motivos éticos y de licencia.

### Twitter/X — corpus crisis energética 2024

Los textos fueron recolectados mediante **web scraping ético**, anonimizados con **hash SHA-256** y generalización geográfica. **No se redistribuye el corpus crudo** por respeto a los Términos de Servicio de X (Twitter) y para preservar la privacidad de los usuarios.

**Acceso a datos procesados:** disponible bajo **solicitud académica formal** a `diego.lojan@unl.edu.ec`.

### MentalRiskES

Corpus propiedad del grupo **SINAI-UJA** (Universidad de Jaén). El acceso requiere **acuerdo formal de confidencialidad** directo con los autores:

[github.com/sinai-uja/corpusMentalRiskES](https://github.com/sinai-uja/corpusMentalRiskES)

---

## Limitaciones

Documentadas explícitamente en la **Sección 5.5** del paper:

- **Etiquetado por keywords** puede omitir expresiones implícitas de estrés.
- **Ausencia de tests estadísticos formales** (McNemar) para validar la significancia de las mejoras.
- **Corpus Twitter/X limitado a Ecuador** en un evento específico; generalización a otras variantes del español requiere adaptación de dominio.
- **SMOTE en espacio de embeddings 128d** (mitigado por similitud coseno 0.65–0.70 con muestras reales).
- **No incluye baselines transformer** (BETO, mBERT) — *out of scope* del estudio, reservado para trabajo futuro.

---

## Trabajo futuro

1. **Comparación con embeddings transformer** (BETO, mBERT, RoBERTa-es) como extractor de features.
2. **Capas de explicabilidad** (SHAP, LIME) para interpretación clínica de predicciones.
3. **Extensión multinacional y multi-crisis** (otros eventos en países hispanohablantes).
4. **Clasificación multi-label** para co-ocurrencia de estrés, ansiedad y depresión.
5. **Tests estadísticos formales** (McNemar, bootstrap con intervalos de confianza).

---

## Autores

### Diego Fernando Lojan Tenesaca — *Autor principal*
Universidad Nacional de Loja · Carrera de Computación
**ORCID:** [`0009-0003-3882-3889`](https://orcid.org/0009-0003-3882-3889)
**Email:** `diego.lojan@unl.edu.ec`

### Luis Antonio Chamba-Eras, PhD — *Director y coautor*
Universidad Nacional de Loja · Director de GITIC
**ORCID:** [`0000-0003-3069-9628`](https://orcid.org/0000-0003-3069-9628)
**Email:** `lachamba@unl.edu.ec`

---

## Agradecimientos

Este trabajo se deriva del **Trabajo de Integración Curricular**:

> *"Ajuste de hiperparámetros en SVM con Optuna (TPE) para detectar estrés durante la crisis energética en Ecuador, 2024"*
> Universidad Nacional de Loja, 2026.

Desarrollado en el marco del proyecto **32-DI-FEIRNR-2025**:

> *"Desarrollo de un marco de trabajo de Inteligencia Artificial para integrar y fortalecer las funciones sustantivas en la Universidad Nacional de Loja y su escalabilidad en las Universidades Públicas de Ecuador"*
> Financiado por la **Universidad Nacional de Loja**.

Los autores agradecen al **psicólogo clínico** que contribuyó a la validación experta del diccionario de términos, y al grupo **SINAI** de la Universidad de Jaén (UJA) por proporcionar acceso al corpus **MentalRiskES** bajo acuerdo formal de confidencialidad.

### Cita del corpus MentalRiskES

```bibtex
@inproceedings{marmol-romero-etal-2024-mentalriskes,
  title     = {{MentalRiskES}: A New Corpus for Early Detection of
               Mental Disorders in {S}panish},
  author    = {M{\'a}rmol-Romero, Ana Mar{\'i}a and Moreno-Mu{\~n}oz, Antonio
               and Plaza-del-Arco, Flor Miriam and Molina-Gonz{\'a}lez, M. Dolores
               and Mart{\'i}n-Valdivia, M. Teresa and Ure{\~n}a-L{\'o}pez, L. Alfonso
               and Montejo-R{\'a}ez, Arturo},
  booktitle = {Proceedings of the 2024 Joint International Conference on
               Computational Linguistics, Language Resources and Evaluation
               (LREC-COLING 2024)},
  year      = {2024},
  pages     = {11204--11214},
  publisher = {ELRA and ICCL},
  url       = {https://aclanthology.org/2024.lrec-main.978}
}
```

---

## Licencia

Distribuido bajo licencia **MIT**. Ver [`LICENSE`](LICENSE) para detalles.

---

## Referencias técnicas

- **[Optuna Documentation](https://optuna.org/)**
- **[Scikit-learn SVM Guide](https://scikit-learn.org/stable/modules/svm.html)**
- **[FastText for Text Classification](https://fasttext.cc/)**
- **[Stanza for Spanish NLP](https://stanfordnlp.github.io/stanza/)**
- **[SMOTE — imbalanced-learn](https://imbalanced-learn.org/)**
- **[CRISP-DM Reference](https://www.datascience-pm.com/crisp-dm-2/)**
