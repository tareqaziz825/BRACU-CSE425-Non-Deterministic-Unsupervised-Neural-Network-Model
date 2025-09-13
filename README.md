# BRACU-CSE425-Non-Deterministic-Unsupervised-Neural-Network-Model
Implementation of a non-deterministic unsupervised neural network model for clustering.


Wine Dataset Clustering with PyTorch and Scikit-Learn
--------

Overview
--------
This project explores clustering techniques on the UCI Wine dataset using
PyTorch and Scikit-Learn. It demonstrates how to load and preprocess data,
apply unsupervised clustering algorithms such as K-Means and Gaussian Mixture
Models, and evaluate clustering performance using metrics like Silhouette Score,
Adjusted Rand Index (ARI), and Normalized Mutual Information (NMI).

The project is intended as a hands-on learning resource for unsupervised learning,
feature scaling, and evaluation of clustering algorithms.

Project Structure
-----------------
- notebook/       -> Contains the main Jupyter Notebook with code and experiments
- requirements.txt -> Lists exact dependencies to recreate the environment
- README.txt       -> Project documentation (this file)

Key Features
------------
- Data loading and preprocessing with pandas and scikit-learn
- Standardization of features using StandardScaler
- Implementation of clustering with:
  - KMeans
  - GaussianMixture
- Evaluation using:
  - Silhouette Score
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
- Visualization of clustering results using Matplotlib

Setup Instructions
------------------
1. Clone this repository:
   git clone https://github.com/your-username/your-repo-name.git

2. Navigate into the project directory:
   cd your-repo-name

3. Create a virtual environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate   (Linux/Mac)
   venv\Scripts\activate      (Windows)

4. Install dependencies:
   pip install -r requirements.txt

5. Launch Jupyter Notebook to explore the code:
   jupyter notebook

Dependencies
------------
All required packages are listed in requirements.txt:
- torch==2.3.0
- pandas==2.2.2
- numpy==1.26.4
- scikit-learn==1.5.0
- matplotlib==3.9.0

Usage
-----
Open the notebook and run all cells to:
1. Load and preprocess the Wine dataset
2. Train clustering models (KMeans, GaussianMixture)
3. Evaluate clustering results with multiple metrics
4. Visualize performance and compare methods

Contributing
------------
Contributions, issues, and feature requests are welcome.
Feel free to fork this repo and submit pull requests.
