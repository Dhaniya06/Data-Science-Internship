# House Price Prediction using Regression

## Objective
Train a simple regression model to estimate house prices from area, bedrooms, age, and location.

## Tools
Python, Pandas, scikit-learn, Matplotlib.

## Run
`pip install -r requirements.txt`  
`python house_price_prediction.py`

## Dataset
`data/house_prices.csv` is synthetic practice data. Prices are illustrative and are not market estimates.

## Workflow
- Load and inspect data
- Split into training and test sets
- Scale numeric columns and one-hot encode location
- Train Linear Regression
- Evaluate with MAE, RMSE, and R-squared
- Plot actual versus predicted values

## Limitations
Real property prices depend on many variables absent from this small synthetic dataset. Do not use the model for real property valuation.
