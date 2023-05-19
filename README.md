# [MumbaiHousingPrices](https://abdealib520-mumbaihousingprices-app-w4wk4l.streamlit.app/)
This website will tell you what would be the price of your dream home in Mumbai.

# Dataset
The dataset that was used was from [Kaggle](https://www.kaggle.com/datasets/sameep98/housing-prices-in-mumbai)

# Data Cleaning and Feature Engineering
1. The boolean columns like New/Resale,Gymnasium,etc. were removed because the correlation between them and the target variable was too low.
2. I tried to use PCA on the boolean columns but even the resulting 3 factors did not show more than 0.1 correlation with the target feature
3. Created a feature called Price_per_sqft to remove outliers.
4. The outliers were removed by location and by using the 3 sigma method.
5. Applied a log transform on the price feature to make it more normalized.

# Model Training
1. Tested many different models like Linear, Lasso, Decision Tree, Gradient Booster, Ada booster on the data.
2. Used Grid Search for Hyperparameter tuning. Also used multithreading to speed up the hyperparameter tuning process.
3. The best model ended up being Gradient Booster with a score of 91%.
