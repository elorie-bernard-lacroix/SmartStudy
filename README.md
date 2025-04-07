# SmartStudy

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Students often feel overwhelmed with work, deadlines, and extracurriculars. The stress of managing these responsibilities can make it difficult for students to establish effective study habits. In this proposal, we discuss the development of a machine learning model designed to recommend personalized lifestyle changes, such as the optimal weekly study time, the need for tutoring, and increased lecture attendance for students. Overall, this tool aims to increase organization and time-management among students and ultimately help them succeed academically.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering) and a
│                         short `-` delimited description, e.g.`1.0-jqp-initial-data-exploration`.
│                           0 - Data exploration 
│                           1 - Data cleaning and feature creation
│                           2 - Visualizations
│                           3 - Modeling (training and evaluating machine learning models)
│                               3.0X – Baseline Model
│                               3.1X – GPA Predictor 
│                               3.2X – Optimizer for study habits
│                               3.3X – KNN sxample students finder
│                               3.4X – Text generation (GPT-4)
│                               3.5X – Final integrated model
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         smartstudy and configuration for tools like black
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── smartstudy   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes smartstudy a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── generate_scaler.py      <- Code to scale data for modeling
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── train_tabpfn.py      <- Code to train tabpfn model into serialized model     
    │   ├── optimizer.py         <- Code to run optimization with tabpfn model  
    │   ├── knn_matching.py      <- Code to run knn matching using user input       
    │   └── gpt_utils.py         <- Code to run gpt-4 model for text generation
    │
    ├── app.py                  <- Code to run the GUI and call relevant scripts
    │
    └── plots.py                <- Code to create visualizations
    
```

--------

