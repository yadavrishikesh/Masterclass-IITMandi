**A modular curriculum that guides learners from descriptive statistics to advanced deep learning and remote sensing, with fully commented Python notebooks and hands-on exercises.**

## Table of Contents

1. [Module 1 · Foundations of Statistics](#module-1--foundations-of-statistics)  
2. [Module 2 · Time Series and Spatial Analytics](#module-2--time-series-and-spatial-analytics)  
3. [Module 3 · Predictive Modeling and Neural Networks from Scratch](#module-3--predictive-modeling-and-neural-networks-from-scratch)  
4. [Module 4 · Deep Learning Architectures with TensorFlow](#module-4--deep-learning-architectures-with-tensorflow)  
5. [Module 5 · Remote Sensing with NASA Resources](#module-5--remote-sensing-with-nasa-resources)  
6. [Getting Started](#getting-started)  
7. [Repository Layout](#repository-layout)  
8. [Contributing](#contributing)  
9. [License](#license)  

---

## Module 1 · Foundations of Statistics

**Goal:** Build an intuitive and formal understanding of statistical thinking and data exploration.

### Topics Covered

- Descriptive statistics: mean, median, variance, skewness, kurtosis
- Probability distributions and basic probability theory
- Confidence intervals and hypothesis testing
- Visualization tools: histograms, scatter plots, boxplots, heatmaps
- Python libraries used: NumPy, Pandas, Matplotlib, Seaborn, Plotly

### Assets

- Notebooks: `notebooks/01_statistics/*.ipynb`
- Exercises: `exercises/01_statistics/*.ipynb`
- Sample data sets for practice

---

## Module 2 · Time Series and Spatial Analytics

**Goal:** Model temporal dynamics and spatial patterns for real-world applications.

### Topics Covered

- Time series decomposition and seasonality
- AR, MA, ARIMA, SARIMA models
- Vector AutoRegressive (VAR) models
- Forecasting techniques and accuracy metrics
- Introduction to geospatial data structures
- Spatial interpolation (kriging), spatial autocorrelation (Moran's I)

### Assets

- Notebooks: `notebooks/02_time_spatial/*.ipynb`
- Exercises: `exercises/02_time_spatial/*.ipynb`
- Synthetic and real spatial/time series datasets

---

## Module 3 · Predictive Modeling and Neural Networks from Scratch

**Goal:** Transition from classical statistical models to building and training neural networks manually and with TensorFlow.

### Part A – Classical Predictive Models

- Regression models: linear, logistic
- Regularization: Lasso, Ridge, Elastic Net
- Tree-based models: Decision Trees, Random Forests, XGBoost
- Evaluation: accuracy, precision, recall, AUC-ROC, cross-validation

### Part B – Neural Networks from Scratch

- Theory of feedforward neural networks
- Implementing forward and backpropagation using NumPy
- Custom training loop with loss functions and optimizers

### Part C – TensorFlow-based Neural Networks

- Rebuilding the same network with TensorFlow/Keras
- Model training and evaluation
- Saving/loading models and making predictions

---

## Module 4 · Deep Learning Architectures with TensorFlow

**Goal:** Explore advanced deep learning models for image and sequence data.

### Topics Covered

- Convolutional Neural Networks (CNNs): architecture, padding, pooling
- Recurrent Neural Networks (RNNs): sequence modeling basics
- Long Short-Term Memory (LSTM) networks: handling long dependencies
- Practical tools: dropout, early stopping, data augmentation
- Transfer learning and pre-trained models

### Assets

- Notebooks: `notebooks/04_deep_learning/*.ipynb`
- Example datasets: CIFAR-10, IMDB, MNIST

---

## Module 5 · Remote Sensing with NASA Resources

**Goal:** Learn to access, process, and analyze satellite data using open-source NASA tools.

### Topics Covered

- Basics of satellite remote sensing and electromagnetic spectra
- Accessing satellite imagery via NASA **Worldview**
- Extracting and visualizing environmental data from **NASA Giovanni**
- Case studies: wildfire monitoring, ocean chlorophyll levels
- Preprocessing steps for spatial and temporal alignment
- Integration of remote sensing with machine learning workflows

### Assets

- Notebooks: `notebooks/05_remote_sensing/*.ipynb`
- Practice exercises inspired by NASA ARSET training

---

## Getting Started

### Prerequisites

- Python 3.10 or later
- pip or conda installed
- Jupyter Notebook or JupyterLab
- (Optional) GPU for deep learning modules

### Installation

```bash
git clone https://github.com/<your_username>/<your_repo>.git
cd <your_repo>
pip install -r requirements.txt
````

### Running Notebooks

```bash
jupyter lab
# or
jupyter notebook
```

Open the desired module and follow the instructions inside.

---

## Repository Layout

```
├── data/                   # Raw and processed data sets
├── notebooks/
│   ├── 01_statistics/
│   ├── 02_time_spatial/
│   ├── 03_predictive_nn/
│   ├── 04_deep_learning/
│   └── 05_remote_sensing/
├── exercises/              # Practice notebooks
├── models/                 # Saved model weights
├── reports/                # Generated visualizations and figures
├── requirements.txt
└── LICENSE
```

---

## Contributing

Pull requests are welcome! For major changes, please open an issue to discuss improvements. Follow consistent code formatting and include helpful comments where applicable.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgements

* NASA ARSET Program
* TensorFlow and Keras Open Source Community
* Developers of Matplotlib, Seaborn, Plotly, scikit-learn

