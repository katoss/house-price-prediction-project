# Predicting Real Estate Prices in Bangalore (Machine Learning Project)
In this project I built a linear regression model to predict house prices in Bangalore, and created a web application from it. 
[__Click here__](http://ec2-54-209-136-199.compute-1.amazonaws.com/) to see the result!

This work is based on an awesome tutorial by [codebasics on YouTube](https://www.youtube.com/playlist?list=PLeo1K3hjS3uu7clOTtwsp94PcHbzqpAdg). Definitely check out their channel!

## Technologies used
* Jupyter Notebook
* Python (pandas, matplotlib, scikit-learn)
* HTML / CSS / JavaScript
* Flask
* AWS EC2

![Schematic representation of project: UI, flask server, and linear regression model](img/schema.png)

_Schematic representation of the project (Image is a screenshot from [@codebasic tutorial](https://www.youtube.com/watch?v=rdfbcdP75KI&list=PLeo1K3hjS3uu7clOTtwsp94PcHbzqpAdg&index=1))_


## Process Overview
### Model
The data used is the [Bengaluru House price dataset from kaggle](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data). 
Columns: area_type, availability, location, size (number of bedrooms), society, total_sqft, bath (count), balcony (count), price.

#### Data Cleaning
* Dropping columns not interesting for model (area_type, society, balcony, availability)
* Handling NA values (dropping rows with NA, since they are not many)
* Making values in size-column uniform (only keep number from string, convert to int), rename to bhk
* Check for outliers (check unique values in bhk, remove houses with more than 20 bedrooms)
* Clean total_sqft column (sometimes there are ranges of values, calculate mean for these)

#### Feature Engineering
* New feature 'price_per_sqft': price/total_sqft
* Problem with 'location': categorical, too many (>1000) options. Usually for text data, creation of dummy columns with one-hot encoding. In this case, this would mean >1000 new columns (too many).
* Solution: Remove locations that appear <=10 times, summarize them under "other" (leaves us with <300 new columns)
* Dimensionality reduction

#### Outlier/Error Removal
* Remove rows with less than threshold value (300sqft) for sqft per bedroom
* Remove rows where price_per_squarefoot is too low or high (anything above or below one standard deviation)
* Remove rows where 2-bedroom apartments are more expensive than 3-bedroom apartment for same size and area (Note: I am wondering if it really makes sense to remove these as outliers. Big rooms can be preferable and more expensive, e.g. for people without children)
* Remove rows where number of bathrooms > number of rooms
* drop price_per_sqft column used for outlier detection, and size (because it is represented in bhk), to only keep columns used in model

#### Model Building
* Continue feature-engineering by using one-hot encoding for location-column (pd.get_dummies for location). 
* To avoid dummy-variable trap, we should have one column less than values. One column can be dropped (if all other columns are 0, it means the dropped column is the respective value). Drop 'other'.
* export model to pickle file
### Web App
#### Backend
Python Flask Server
#### Frontend 
UI/Website
#### Deployment


one hot encoding
gridsearchcv