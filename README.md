# Boston-Price-Predictor
🏡 Boston House Price Predictor This is a Machine Learning web application that predicts the median value of homes in Boston based on various neighborhood features. By using Polynomial Regression, the model achieves a high accuracy score, capturing complex relationships that a simple linear model misses.
Features
High Accuracy: Uses a Degree-2 Polynomial Regression model with an R² score of 0.86.

Interactive UI: Built with Streamlit, allowing users to adjust 13 different features (like crime rate, room count, and tax rates) via a sidebar.

Data Driven: Trained on the classic Boston Housing Dataset.
Technologies Used
Python (Core Logic)

Scikit-Learn (Machine Learning & Preprocessing)

Streamlit (Web Interface)

Pandas & NumPy (Data Manipulation)
How to Run Locally
Clone this repository or download the files.

Install the requirements:

Bash

pip install streamlit scikit-learn pandas
Run the application:

Bash

streamlit run app.py
📝 Dataset Columns
RM: Average number of rooms

LSTAT: Percentage of lower status population

CRIM: Per capita crime rate

NOX: Nitric oxides concentration

(And 9 other socio-economic factors)
