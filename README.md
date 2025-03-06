# Retail-Sales-Analysis-and-Forecast
# Retail Sales Analysis and Forecast

## Overview
This project leverages predictive analytics to drive strategic business decisions by forecasting future sales trends and identifying high-value customers. Two models were implemented: a **Logistic Regression model** to predict high-spending customers and an **ARIMA time series model** to forecast sales for the next 12 months.

## Dataset Description
### Columns:
- **InvoiceNo** - Unique invoice identifier.
- **StockCode** - Product code.
- **Description** - Product name.
- **Quantity** - Number of products purchased.
- **InvoiceDate** - Date and time of transaction.
- **UnitPrice** - Price per unit.
- **CustomerID** - Unique customer identifier.
- **Country** - Customer's country.

### Identified Issues:
- 1,454 missing values in `Description`.
- 135,080 missing values in `CustomerID`.
- 10,624 negative values in `Quantity`.
- 2 negative values in `UnitPrice`.
- 2,515 zero values in `UnitPrice`.
- Special characters and blanks in `Description`.
- 5,268 duplicate rows.
- Some rows in `Country` labeled as 'Unspecified'.

## Data Cleaning Process
- Filled missing values in `Description` as "Unknown".
- Assigned unique negative `CustomerID` values for missing entries.
- Removed rows with negative `Quantity` and zero or negative `UnitPrice`.
- Dropped 5,268 duplicate rows.
- Standardized `Country` by removing 'Unspecified' entries.

## Exploratory Data Analysis (EDA)

### Key Insights
#### Best-Selling Products
- **Top product:** *Paper Craft Little Birdie* (80,995 units sold).
- **Second:** *Medium Ceramic Top Storage Jar* (78,033 units).
- **Third:** *World War 2 Gliders Asstd Designs* (54,855 units).

#### Top 10 Revenue-Generating Products
- **Top product:** *Dotcom Postage* ($206,248.77 revenue).
- **Second:** *Regency Cakestand 3 Tier* ($174,131.04 revenue).
- **Third:** *Paper Craft Little Birdie* ($168,469.60 revenue).

#### Total Revenue Per Country
- **United Kingdom:** $9,001,855.17
- **Netherlands:** $285,446.34
- **EIRE:** $283,170.52

#### Number of Transactions Per Country
- **United Kingdom:** 18,019 transactions.
- **Germany:** 457 transactions.
- **France:** 392 transactions.

## Visualizations
![Sales trend](https://github.com/joeldonaldworld/Retail-Sales-Analysis-and-Forecast/blob/joeldonaldworld-patch-1/Sales%20trend.png)


![Best selling products](https://github.com/joeldonaldworld/Retail-Sales-Analysis-and-Forecast/blob/joeldonaldworld-patch-1/Top%20selling%20products.png)


![Top revenue generating products](https://github.com/joeldonaldworld/Retail-Sales-Analysis-and-Forecast/blob/joeldonaldworld-patch-1/Top%20revenue%20generating%20products.png)


![Top customers](https://github.com/joeldonaldworld/Retail-Sales-Analysis-and-Forecast/blob/joeldonaldworld-patch-1/Top%20customers.png)
  

![Top countries by sales](https://github.com/joeldonaldworld/Retail-Sales-Analysis-and-Forecast/blob/joeldonaldworld-patch-1/Top%20countries.png)


![Top countries by transaction](https://github.com/joeldonaldworld/Retail-Sales-Analysis-and-Forecast/blob/joeldonaldworld-patch-1/top%20countries%20by%20transactions.png)


## Feature Engineering
### Key Features Created:
1. **Transaction-Based Features**
   - `TotalSales = Quantity * UnitPrice`
   - `AvgSpendPerPurchase = df.groupby("CustomerID")["TotalSales"].transform("mean")`
   - `HighSpender = (AvgSpendPerPurchase > Median(AvgSpendPerPurchase)).astype(int)`

2. **Customer Behavior Features**
   - `TotalTransactions = df.groupby("CustomerID")["InvoiceNo"].transform("nunique")`
   - `AvgItemsPerPurchase = df.groupby("CustomerID")["Quantity"].transform("mean")`
   - `FrequentBuyer = (TotalTransactions > Median(TotalTransactions)).astype(int)`

3. **Time-Based Features**
   - Extracted **Year, Month, Day, Hour, Weekday** from `InvoiceDate`.
   - Calculated `Recency` as the number of days since last purchase.
   - Created a `Seasonality` flag for peak sales months (Nov-Dec).

4. **Scaling Numerical Features**
   - Used `MinMaxScaler` to normalize features like `Quantity`, `UnitPrice`, `TotalSales`, `Recency`.

## Data Preprocessing
- Dropped irrelevant columns (`InvoiceNo`, `StockCode`, `Description`, `InvoiceDate`).
- Encoded categorical features (`Country`, `ProductCategory`).
- Applied **SMOTE** to balance class distribution.

## Machine Learning Model: Predicting High-Spending Customers
- **Model Used:** Logistic Regression
- **Performance Metrics:**
  - **Accuracy:** 89.82%
  - **Precision:** 91.54%
  - **Recall:** 89.82%
  - **F1 Score:** 89.71%
- **Overfitting Check:**
  - **Training Accuracy:** 96.15%
  - **Test Accuracy:** 89.82%
  - **Conclusion:** Possible overfitting detected.
- **Hyperparameter Tuning:**
  - **Best Regularization Strength (`C`)**: 100
  - **Tuned Test Accuracy:** 99.16%
  - **Conclusion:** Overfitting significantly reduced.
 
  ## Feature Importance
  ![]()

## Sales Forecasting using ARIMA
- **12-month sales forecast generated using ARIMA.**
![Sales forecast](https://github.com/joeldonaldworld/Retail-Sales-Analysis-and-Forecast/blob/joeldonaldworld-patch-1/Sales%20forecast.png)

- **Key Observations:**
  - Steady growth, fluctuating around 1,850 - 1,865 units.
  - February sees the highest sales (1,865 units).
  - No drastic spikes or drops.
    
![12 month forecast table](https://github.com/joeldonaldworld/Retail-Sales-Analysis-and-Forecast/blob/joeldonaldworld-patch-1/Forecast%20for%20the%20next%2012%20months.png)

## Business Recommendations
- **Inventory Optimization:** Given the steady growth in sales, businesses should ensure sufficient stock levels to meet demand, particularly in February, which sees a slight peak.
- **Pricing Strategy:** `UnitPrice` has the highest impact on high spenders. Businesses should experiment with pricing strategies to maximize revenue without discouraging purchases.
- **Geographical Expansion:** The UK dominates in transactions and revenue, but potential growth opportunities exist in Germany and France.
- **Targeted Promotions:** Customers with high `AvgSpendPerPurchase` and frequent transactions should be targeted for loyalty programs and exclusive offers.
- **Product Strategy:** High revenue-generating products should be prioritized in marketing and inventory planning.
- **Seasonal Adjustments:** Peak sales months (Nov-Dec) should be leveraged with promotional campaigns to maximize revenue.

## Conclusion
This project provides actionable insights into customer behavior, sales forecasting, and revenue generation. The predictive models help businesses make data-driven decisions, optimize inventory, and enhance customer engagement.

---

