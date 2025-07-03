# AI-ML-Practical-Application-Project2
Using ML CRISP-DM framework - Predict Used Car Sales Price for Car Dealerships Business Value.

# Overview:
The original dataset contained information on 3 million used cars. The practice dataset contains information on 426K cars to ensure speed of processing. The objective of this project is to determine what factors make a car more or less expensive. After detailed analysis, a clear recommendation is provided to the client -- a used car dealership -- as to what consumers value in a used car.

# Business Understanding:
Car dealerships operate on a simple principle: buy low, sell high. This makes accurately predicting a car's final sale price crucial for their profitability. Beyond just profit, however, understanding car demand is equally vital for ensuring fast inventory turnover. While many factors influence demand, the demographics and preferences of potential drivers are typically the most significant.

# Exploratory Data Analysis and Data Cleaning:
Data Overview and Initial Cleaning Strategy
The dataset, comprising 426,880 records and 18 features, presented significant challenges due to its lack of cleanliness, prevalence of missing values, and the inclusion of irrelevant advertisement entries. Specifically, the "model" column contained numerous "bogus" entries (e.g., "SPECIAL FINANCE PROGRAM 2020," "BUY HERE PAY HERE"), which fortunately correlated with null values in the "manufacturer" column, allowing for their automatic removal during cleaning. Furthermore, many feature columns required data type transformations.

Based on domain knowledge and initial exploration:
* The "VIN" and "id" columns were deemed redundant as unique identifiers, given the existing data indexing, and were subsequently dropped.
* The "region" column was also removed due to the presence of the more specific "state" column.
* As previously noted in the EDA section, the "model" column provided no additional valuable information and was therefore dropped.
* Rows with missing "manufacturer" and "year" data simultaneously, or those with a "price" of zero, were critical information for our model and thus removed.
* A summary of initial missing values revealed substantial gaps in features such as 'size' (71.77%), 'cylinders' (41.62%), 'condition' (40.79%), 'VIN' (37.73%), 'drive' (30.59%), 'paint_color' (30.50%), and 'type' (21.75%), among others.

Following the initial data type transformations and the imputation of missing values with appropriate strings, the missing data percentages were significantly reduced, with "manufacturer" at 3.8% and "odometer" at 0.56%. The remaining missing values in these columns were then handled using a .dropna() approach. No duplicate records were found, and the data cleaning process ultimately resulted in a refined dataset of 375,619 rows with 14 features.

Distribution Analysis and Outlier Handling
Post-cleaning, histogram plots revealed significant skewness and outliers within key numerical features. The 'price' distribution was highly right-skewed with extreme outliers, while 'year' was left-skewed, and 'odometer' was right-skewed.

To address these, specific outlier removal strategies were employed:
For 'price', the calculated mean of $83,781 and a remarkably high standard deviation of $12,983,817 underscored the extreme right skew. Consequently, prices exceeding $150,000 were removed.
For the 'year' column, records falling outside of [mean - 2 * standard deviation] and [mean + 2 * standard deviation] were excluded.
For the 'odometer' column, records exceeding [mean + 2 * standard deviation] were removed.

The numerical columns have high correlations, namely, price and year at 0.57, and price and odometer at -0.53. However, we can also see that odometer and year are highly correlated at -0.67

# Building ML Models:
Prior to model construction, it is imperative that the data undergoes appropriate transformation, normalization, and scaling. To streamline this preparatory phase and the subsequent modeling process, a robust pipeline has been implemented.

This pipeline initiates with the OneHotEncoder for converting categorical features. Following this, the data is normalized using a QuantileTransformer, after which a StandardScaler is applied for feature scaling. These prepared features then serve as input for subsequent models, ultimately feeding into the final regression model.

# Model Performance Evaluation
Our initial evaluation established a baseline linear model with Root Mean Squared Error (RMSE) scores of 11,590 on the training data and 11,574 on the test data. Subsequent modeling with regularized linear regressions demonstrated significant improvements.

Specifically:
Lasso regression (alpha=0.5) yielded RMSE scores of 8,676 (training) and 8,637 (test).
Ridge regression (alpha=0.5) achieved RMSE scores of 8,685 (training) and 8,647 (test).
Standard Linear Regression produced RMSE scores of 8,679 (training) and 8,641 (test).
As illustrated in the accompanying charts—which exhibit near-perfect alignment even under magnification—the performance metrics for Lasso, Ridge, and standard Linear Regression are remarkably similar and represent a notable improvement over the baseline model.

A pipeline with Lasso as automatic feature selection with RMSE of 9340 and 9302 for Train and Test data revealed 10 features as the most important ones: ferrari, fuel_diesel, title_status_clean, drive_fwd, drive_unknown_drive, type_pickup, cylinders_4 cylinders, cylinders_8 cylinders, year, and odometer.

# Grid Search CV:
Initial modeling, employing a GridSearchCV pipeline with Lasso regression to automatically select between 4 and 10 features, consistently indicated that a newer model year, lower odometer reading, and a clean title are the most influential positive predictors of vehicle value across the general market.

However, the analysis of regression coefficients also revealed an unexpected penalty for 4-cylinder engines and front-wheel drive (FWD) within these models. This counterintuitive finding suggests a misalignment with the presumed operational realities and specific needs of the car dealership.

Adhering to the CRISP-DM Process Model, this necessitates a crucial revisit to the "Business Understanding" phase. A review of the type feature clearly shows that SUVs, Sedans, and Pickups constitute the predominant vehicle categories in the dataset. To enhance the predictive accuracy and business relevance of our machine learning models, the strategy has been revised: the data will now be segmented into these three primary categories, and dedicated modeling pipelines will be developed and executed for each. Furthermore, during the subsequent data cleaning, it was determined to retain only recent model years by truncating older entries.

# Deployment: 
A vehicle's general price is strongly influenced by its newer model year, lower odometer reading, and a clean title. However, predicting exact pricing becomes more intricate when accounting for diverse vehicle types and specific features. Given the scope of this report and the constraints of the provided dataset, vehicles have been further segmented into three primary categories: SUVs, Sedans, and Pickups.

Within the SUV market, in addition to the aforementioned general features, our analysis suggests prioritizing models from Toyota, Land Rover, or Tesla, while generally avoiding Mitsubishi and 4-cylinder SUVs. It's crucial to acknowledge that this guidance is derived solely from the provided dataset and should be supplemented with consideration for external factors like government incentives for electric vehicles and broader market shifts.

For the Sedan category, a dual strategy appears viable: either target well-conditioned, more affordable cars often equipped with higher-cylinder engines, or focus on the luxury segment, including brands like BMW, Audi, Mercedes, Porsche, and Lexus.

Finally, for Pickup vehicles, the data indicates a preference for 4WD and more powerful diesel engines, while Nissan models and 4-cylinder engines appear less favorable. Geographically, California, Arizona, and Oregon demonstrate stronger sales performance for pickups.
