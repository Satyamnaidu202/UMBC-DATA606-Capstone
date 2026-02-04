# Hotel Booking Cancellation Prediction

Prepared for the UMBC Data Science Master Degree Capstone  
by Dr. Chaojie (Jay) Wang

**Author:** KALLA SHANKAR RAM SATYAM NAIDU  

**GitHub Repository:** https://github.com/Satyamnaidu202/UMBC-DATA606-Capstone  
**LinkedIn Profile:** https://www.linkedin.com/public-profile/settings?trk=d_flagship3_profile_self_view_public_profile  

---

## 1. Background

### 1.1 Problem Overview
The hospitality industry experiences significant financial losses due to hotel booking cancellations.
Cancellations lead to inefficient room utilization, revenue loss, and challenges in demand forecasting.
Understanding the factors that influence booking cancellations can help hotels improve revenue management
strategies and operational planning.

This project focuses on predicting whether a hotel booking will be canceled using historical booking data.
By applying data analysis and machine learning techniques, the project aims to identify patterns and
key predictors of cancellation behavior.

### 1.2 Importance of the Study
Accurate cancellation prediction enables hotels to:
- Optimize pricing strategies
- Improve overbooking decisions
- Reduce revenue loss
- Enhance customer relationship management

From a data science perspective, this problem represents a real-world binary classification task
with structured data, making it suitable for predictive analytics and model interpretability.

### 1.3 Research Questions
The primary research questions addressed in this project are:
1. What booking characteristics are most strongly associated with hotel booking cancellations?
2. Can machine learning models accurately predict whether a booking will be canceled?
3. Which features contribute most to the prediction of cancellations?
4. How can predictive insights support better decision-making in hotel management?

---

## 2. Data

### 2.1 Data Source
The dataset used in this project is the **Hotel Booking Demand Dataset**, obtained from Kaggle.
The data was originally collected from real hotel booking systems and has been widely used
for academic research.

### 2.2 Dataset Description
- **File Name:** hotel_bookings.csv  
- **File Size:** Approximately 5 MB  
- **Number of Rows:** 119,390  
- **Number of Columns:** 32  

### 2.3 Time Period
The dataset contains booking records from **July 2015 to August 2017**.

### 2.4 Unit of Observation
Each row in the dataset represents **one individual hotel booking**.

### 2.5 Data Dictionary (Selected Key Variables)

| Column Name | Data Type | Description | Possible Values |
|------------|----------|-------------|----------------|
| hotel | Categorical | Type of hotel | City Hotel, Resort Hotel |
| is_canceled | Integer | Whether the booking was canceled | 0 = Not Canceled, 1 = Canceled |
| lead_time | Integer | Days between booking and arrival | Non-negative integers |
| arrival_date_year | Integer | Arrival year | 2015–2017 |
| arrival_date_month | Categorical | Arrival month | Jan–Dec |
| stays_in_weekend_nights | Integer | Weekend nights booked | 0 or more |
| stays_in_week_nights | Integer | Weekday nights booked | 0 or more |
| adults | Integer | Number of adults | 0 or more |
| children | Integer | Number of children | 0 or more |
| meal | Categorical | Meal plan | BB, HB, FB, SC |
| market_segment | Categorical | Booking channel | Online TA, Offline TA, Corporate |
| deposit_type | Categorical | Deposit type | No Deposit, Refundable, Non Refund |
| customer_type | Categorical | Type of customer | Transient, Contract, Group |
| adr | Float | Average Daily Rate | Continuous |

### 2.6 Target Variable
- **Target / Label:** `is_canceled`
  - 1 = Booking canceled
  - 0 = Booking not canceled

### 2.7 Feature Variables
Potential predictors include:
- Lead time
- Hotel type
- Market segment
- Deposit type
- Customer type
- Number of guests
- Length of stay
- Average daily rate (ADR)
- Arrival month and year

These features will be further evaluated and refined during exploratory data analysis
and feature selection stages.

---

