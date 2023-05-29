# [MumbaiHousingPrices](https://abdealib520-mumbaihousingprices-app-w4wk4l.streamlit.app/)
This website will tell you what would be the price of your dream home in Mumbai.

# Dataset
The data was scraped entirely from [MagicBricks](https://www.magicbricks.com)

# Shortcomings
Generally speaking, houses and flats built more recently are much more likely to be sold on an online platform like MagicBricks. Home owmners are much less likely to sell their homes online, instead relying on local brokers. So MagicBricks mostly only has construction companies looking to sell their flats. Which is why the data taken from MagicBricks is likely to be biased towards more expensive and recently constructed houses and flats.

# Data Cleaning and Feature Engineering
1. Converted features like Price, Area and Bedrooms from strings like '1024sqft' to numeric values.
2. Replaced missing values with the median.
3. Used One Hot Encoding on the Locations feature.
4. Features used for model training were Price as target feature and Area, Location and Bedrooms as input features.
5. Applied a log transform on the Price feature to make it more normalized.

# Model Training
1. Tested many different models like Linear, Lasso, Decision Tree, Gradient Booster, Ada booster on the data.
2. Used Grid Search for Hyperparameter tuning. Also used multithreading to speed up the hyperparameter tuning process.
3. The best model ended up being Gradient Booster with a score of 82%.
