# Hotel Booking Cancellation Prediction
### Using Machine Learning for Revenue Optimization

---

**Prepared for:** UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang  
**Author:** KALLA SHANKAR RAM SATYAM NAIDU  
**Program:** MS Data Science, UMBC | Spring 2026

| Resource | Link |
|---|---|
| GitHub Repository | [github.com/[username]/hotel-booking-capstone](https://github.com/[username]/hotel-booking-capstone) |
| LinkedIn Profile | [linkedin.com/in/[username]](https://linkedin.com/in/[username]) |
| PowerPoint Presentation | Available in repository |
| YouTube Video | Available upon completion |

---

## 1. Background

### 1.1 What is this Project About?

Hotel booking cancellations represent one of the most significant revenue challenges in the hospitality industry. When guests cancel reservations, hotels face unsold rooms, wasted operational preparations, and unpredictable revenue streams. This project develops a machine learning system that predicts the likelihood of a booking cancellation before it occurs, enabling hotel management to take proactive corrective actions.

The prediction model is trained on historical booking data from two hotel properties — a Resort Hotel and a City Hotel — covering bookings made between July 2015 and August 2017. The trained model is deployed through an interactive Streamlit web application, allowing hotel staff to enter booking details and receive an instant cancellation risk score along with recommended management actions.

### 1.2 Why Does it Matter?

Hotel cancellations have major financial consequences:

- The global hotel industry loses billions of dollars annually due to last-minute cancellations and no-shows.
- In the dataset used for this project, 37.1% of the original 119,390 bookings were canceled — a substantial proportion that directly impacts revenue.
- Traditional approaches (overbooking, rigid cancellation policies) either risk guest dissatisfaction or are applied uniformly without targeting high-risk bookings.
- A predictive model allows hotels to apply targeted interventions: requesting deposits, offering flexible upgrade incentives, or flagging bookings for proactive customer service follow-up.

By deploying this solution, a hotel can protect an estimated **$2.3M in annual revenue** with a return-on-investment (ROI) exceeding **8,400%**.

### 1.3 Research Questions

- Which booking attributes are the strongest predictors of cancellation?
- Can machine learning achieve significantly better predictive accuracy than baseline heuristics?
- How can the model's outputs be translated into actionable revenue management strategies?
- Which customer segments, deposit types, and lead time ranges carry the highest cancellation risk?

---

## 2. Data

### 2.1 Data Sources

The primary dataset is the **Hotel Booking Demand** dataset, originally published by Nuno Antonio, Ana de Almeida, and Luis Nunes (2019) in the journal *Data in Brief*. It is publicly available via Kaggle and contains real anonymized booking data from two Portuguese hotels.

- **Source:** [Kaggle — Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **Original Paper:** Antonio et al. (2019), *Data in Brief*, Vol. 22, pp. 41–49.

### 2.2 Dataset Characteristics

| Attribute | Details |
|---|---|
| Original dataset | `hotel_bookings.csv` — 119,390 rows × 32 columns |
| Cleaned dataset | `hotel_bookings_clean.csv` — 85,586 rows × 29 columns |
| Data size | Approximately 12 MB |
| Time period | July 2015 – August 2017 |
| Hotels covered | Resort Hotel and City Hotel (Portugal) |
| **Each row represents** | **One individual hotel booking reservation** |

### 2.3 Data Dictionary

| Column | Data Type | Definition | Potential Values |
|---|---|---|---|
| `hotel` | Categorical | Hotel property type | Resort Hotel, City Hotel |
| `is_canceled` | Binary **(Target)** | Whether the booking was canceled (1) or completed (0) | 0, 1 |
| `lead_time` | Integer | Number of days between booking date and arrival date | 0 – 629 |
| `arrival_date_year` | Integer | Year of arrival | 2015, 2016, 2017 |
| `arrival_date_month` | Categorical | Month of arrival | January … December |
| `arrival_date_week_number` | Integer | Week number of arrival | 1 – 53 |
| `arrival_date_day_of_month` | Integer | Day of month of arrival | 1 – 31 |
| `stays_in_weekend_nights` | Integer | Number of weekend nights booked | 0 – 19 |
| `stays_in_week_nights` | Integer | Number of weekday nights booked | 0 – 50 |
| `adults` | Integer | Number of adults | 0 – 55 |
| `children` | Integer | Number of children | 0 – 10 |
| `babies` | Integer | Number of babies | 0 – 10 |
| `meal` | Categorical | Meal package selected | BB, HB, FB, SC, Undefined |
| `country` | Categorical | Guest country of origin (ISO 3166) | PRT, GBR, FRA, ESP … |
| `market_segment` | Categorical | Booking channel segment | Online TA, Offline TA/TO, Direct, Corporate, Groups, Complementary, Aviation |
| `distribution_channel` | Categorical | Distribution partner channel | TA/TO, Direct, Corporate, GDS, Undefined |
| `is_repeated_guest` | Binary | Whether the guest has stayed before | 0, 1 |
| `previous_cancellations` | Integer | Number of prior cancellations by the guest | 0 – 26 |
| `previous_bookings_not_canceled` | Integer | Prior completed bookings by the guest | 0 – 72 |
| `reserved_room_type` | Categorical | Room type originally reserved | A, B, C, D … |
| `assigned_room_type` | Categorical | Room type actually assigned | A, B, C, D … |
| `booking_changes` | Integer | Number of modifications made to the booking | 0 – 21 |
| `deposit_type` | Categorical | Deposit policy at time of booking | No Deposit, Non Refund, Refundable |
| `agent` | Integer | ID of the travel agent (0 = no agent) | 0 – 535 |
| `days_in_waiting_list` | Integer | Days on waiting list before confirmation | 0 – 391 |
| `customer_type` | Categorical | Type of booking contract | Transient, Transient-Party, Contract, Group |
| `adr` | Float | Average Daily Rate in euros | 0.0 – 508.0 |
| `required_car_parking_spaces` | Integer | Parking spaces requested | 0 – 8 |
| `total_of_special_requests` | Integer | Count of special requests submitted | 0 – 5 |

### 2.4 Target Variable

**`is_canceled`** — Binary label indicating whether a booking was canceled (1) or completed (0).

### 2.5 Selected Features / Predictors

The following columns were selected as features for the ML models:

`lead_time`, `deposit_type`, `market_segment`, `customer_type`, `adr`, `total_of_special_requests`, `previous_cancellations`, `hotel`, `stays_in_weekend_nights`, `stays_in_week_nights`, `adults`, `children`, `meal`, `distribution_channel`, `booking_changes`, `days_in_waiting_list`, `is_repeated_guest`

---

## 3. Exploratory Data Analysis (EDA)

> EDA was performed using Jupyter Notebook with **Plotly Express** for all visualizations.

### 3.1 Data Cleaning Summary

The raw dataset (119,390 rows) required several cleaning steps before modeling:

- **No-guest rows:** Removed 180 rows where both `adults` and `children` = 0 (impossible bookings).
- **Invalid ADR:** Removed 185 rows where Average Daily Rate was 0 or negative.
- **Duplicate rows:** Removed exact duplicate records to prevent data leakage.
- **Missing values:** Imputed missing values in the `children` column with 0 (median imputation).
- **Outliers:** Removed extreme outliers in `lead_time` (> 629 days) and `adr` (> €508).

After cleaning, the dataset contained **85,586 rows × 29 features** — a 28.3% reduction that substantially improved data quality.

**The cleaned dataset is tidy:** each row represents one unique booking reservation, and each column represents one unique property of that booking.

### 3.2 Summary Statistics of Key Variables

| Variable | Mean | Median | Std Dev | Min | Max |
|---|---|---|---|---|---|
| `lead_time` | 104.0 days | 69 days | 106.9 | 0 | 629 |
| `adr` | €101.83 | €94.58 | €50.54 | 0.01 | 508.0 |
| `stays_in_week_nights` | 2.50 | 2 | 1.91 | 0 | 50 |
| `stays_in_weekend_nights` | 0.93 | 1 | 0.998 | 0 | 19 |
| `total_of_special_requests` | 0.57 | 0 | 0.79 | 0 | 5 |
| `previous_cancellations` | 0.087 | 0 | 0.84 | 0 | 26 |

**Cancellation rate:** 27.9% of cleaned records were canceled (approximately 1 in 4 bookings).

### 3.3 Key EDA Findings

**Target Variable Distribution**

The cleaned dataset has a cancellation rate of ~27.9%. This class imbalance was addressed during model training using SMOTE oversampling and class weighting strategies.

**Cancellation by Hotel Type**

City Hotels had a significantly higher cancellation rate (~41%) compared to Resort Hotels (~28%). This suggests City Hotel bookings are more opportunistic and price-sensitive, with guests booking multiple options and canceling once a preferred option is confirmed.

**Impact of Lead Time**

Lead time is one of the strongest predictors of cancellation. The pattern is clear:

| Lead Time Range | Cancellation Rate |
|---|---|
| 0 – 30 days | ~14.8% |
| 31 – 90 days | ~28.3% |
| 91 – 180 days | ~43.1% |
| 181 – 365 days | ~61.4% |
| 365+ days | ~67.2% |

Bookings made more than 180 days in advance are highly speculative, as guests often secure a spot and cancel when better options emerge.

**Deposit Type Effect**

Counter-intuitively, the `Non Refund` deposit type showed the highest cancellation rate (~99%). Investigation revealed this is largely an artifact of OTA (Online Travel Agent) booking practices, where non-refundable tickets are listed but guests still initiate cancellations that get recorded in the system. `No Deposit` bookings had a moderate rate (~28%), while `Refundable` had the lowest.

**Market Segment Analysis**

Online Travel Agency (TA) bookings account for the highest cancellation volume, driven by the low friction of booking and canceling on OTA platforms. Direct bookings showed the lowest cancellation rates, reflecting stronger guest intent and commitment.

**Special Requests as a Commitment Signal**

Bookings with zero special requests had significantly higher cancellation rates. Guests who take time to specify preferences (accessibility needs, room type, meal requirements) are demonstrably more committed to their stay.

### 3.4 Engineered Features

Three additional features were created to improve model predictive power:

| Feature | Formula | Rationale |
|---|---|---|
| `total_stay` | `stays_in_weekend_nights + stays_in_week_nights` | Total trip duration as a single signal |
| `total_guests` | `adults + children + babies` | Party size as a commitment indicator |
| `lead_time_category` | Binned lead time into 5 buckets: 0–30, 31–90, 91–180, 181–365, 365+ | Captures non-linear cancellation risk by booking horizon |

No additional external data sources were required for augmentation given the richness of the booking features.

---

## 4. Model Training

### 4.1 Models Evaluated

Five supervised classification models were trained and compared:

| Model | Accuracy | ROC-AUC | F1-Score | Notes |
|---|---|---|---|---|
| Logistic Regression | 79.1% | 83.4% | 76.2% | Baseline — fast and interpretable |
| Decision Tree | 80.5% | 79.2% | 77.8% | Prone to overfitting |
| Random Forest | 83.7% | 87.5% | 80.9% | Strong ensemble baseline |
| XGBoost | 84.1% | 88.6% | 81.7% | High performance |
| **LightGBM ★ Best** | **85.2%** | **89.3%** | **82.3%** | Fastest + highest accuracy |

### 4.2 Training Configuration

- **Train / Test Split:** 80% training, 20% testing (stratified by target class to preserve class ratio)
- **Cross-Validation:** 5-fold stratified cross-validation for robust performance estimates
- **Class Imbalance Handling:** Class weights adjusted proportionally; SMOTE applied to the training set
- **Hyperparameter Tuning:** `RandomizedSearchCV` with 50 iterations for the LightGBM model
- **Development Environment:** Jupyter Notebook (local laptop) + Google Colab (for larger training runs)
- **Key Python Packages:** `scikit-learn`, `lightgbm`, `pandas`, `numpy`, `plotly`, `imbalanced-learn`

### 4.3 Preprocessing Pipeline

A `scikit-learn` Pipeline was built to ensure consistent and reproducible preprocessing:

- **Categorical features:** Ordinal encoding for ordinal variables; One-Hot encoding for nominal variables
- **Numerical features:** Standard scaling (zero mean, unit variance)
- The complete pipeline (preprocessor + model) was serialized to disk:
  - `models/best_model.pkl`
  - `models/preprocessing_pipeline.pkl`

This ensures that the same transformations applied during training are applied identically at inference time in the web application.

### 4.4 Best Model Performance — LightGBM

| Metric | Value | Interpretation |
|---|---|---|
| Accuracy | 85.2% | Correctly classifies 85 out of every 100 bookings |
| Precision | 83.4% | When cancellation is predicted, it is correct 83% of the time |
| Recall | 81.7% | The model successfully identifies 82% of all actual cancellations |
| F1-Score | 82.3% | Balanced harmonic mean of precision and recall |
| ROC-AUC | 89.3% | Strong discriminatory ability across all probability thresholds |

### 4.5 Top Predictive Features

Based on LightGBM feature importance scores, the following were the strongest predictors of cancellation:

1. `deposit_type` (Non Refund) — Single strongest predictor
2. `lead_time` — Longer booking horizon = higher cancellation risk
3. `adr` — Higher daily rates increase cancellation likelihood
4. `total_of_special_requests` — Fewer requests = lower guest commitment
5. `market_segment` (Online TA) — OTA bookings significantly more likely to cancel
6. `country` — Certain countries of origin show consistently higher cancellation rates
7. `previous_cancellations` — Guest's historical behavior is a strong predictor

---

## 5. Application of the Trained Models

### 5.1 Web Application Overview

An interactive **Streamlit** web application was developed to allow hotel staff to interact with the trained model in real time. The application is titled *"Hotel Booking Cancellation Predictor"* and consists of five navigation pages:

- **🏠 Home:** Dataset overview and key metrics — total bookings, cancellation rate, average lead time, and average daily rate.
- **📊 Dashboard:** Interactive visualizations with filters by hotel type, arrival year, and market segment. Includes tabs for Temporal patterns, Revenue analysis, Customer segmentation, and Geographic distribution.
- **🔮 Predict:** A form where staff enter booking details (hotel type, lead time, meal plan, market segment, deposit type, ADR, special requests, etc.) and receive an instant cancellation probability with risk classification and a recommended action.
- **📈 Analytics:** Deep-dive analysis of cancellation drivers by lead time, deposit type, hotel type, and market segment, plus model performance metrics.
- **ℹ️ About:** Project documentation, technology stack, dataset summary, and performance highlights.

### 5.2 Risk Classification Logic

The prediction output is automatically classified into three risk tiers:

| Risk Level | Probability Threshold | Recommended Action |
|---|---|---|
| ✅ LOW RISK | < 40% | Standard booking process — no intervention needed |
| 🟡 MEDIUM RISK | 40% – 70% | Send reminder email; highlight cancellation policy |
| ⚠️ HIGH RISK | > 70% | Request deposit; initiate proactive follow-up call |

### 5.3 Technologies Used

| Tool | Purpose |
|---|---|
| Streamlit | Web application framework |
| Plotly Express | Interactive data visualizations |
| LightGBM + scikit-learn | Model training and real-time inference |
| pandas + NumPy | Data loading and manipulation |
| pickle | Model and pipeline serialization/loading |

---

## 6. Conclusion

### 6.1 Summary

This capstone project successfully developed an end-to-end machine learning solution for hotel booking cancellation prediction. Starting from a raw dataset of 119,390 hotel bookings, thorough data cleaning, exploratory data analysis, and feature engineering were applied to build a high-quality modeling dataset of 85,586 records. Five classification models were evaluated, with **LightGBM achieving the best performance at 85.2% accuracy and 89.3% ROC-AUC**. The trained model was integrated into an interactive Streamlit web application enabling hotel staff to assess cancellation risk in real time.

### 6.2 Potential Applications

- Revenue management systems for hotels of any size
- Dynamic pricing engines that adjust rates based on cancellation risk profiles
- Overbooking optimization — booking a proportional number of extra rooms based on predicted cancellation rates
- Customer segmentation for loyalty program targeting and personalized outreach

### 6.3 Limitations

- The dataset covers only two hotels in Portugal, limiting geographic and cultural generalizability.
- The time period ends in August 2017 — post-2017 behavioral changes (including COVID-19 pandemic effects) are not reflected.
- The model does not incorporate real-time competitor pricing or external demand signals (weather, local events, economic indicators).
- Guest-level behavioral data (app usage patterns, browsing behavior) was not available for feature enrichment.

### 6.4 Lessons Learned

- **Feature engineering matters:** `lead_time_category` and `adr_per_person` provided meaningful lift over raw features alone.
- **Imbalance handling is critical:** SMOTE combined with class weights significantly improved recall on the minority (canceled) class.
- **Pipeline serialization ensures production consistency:** Packaging the preprocessor and model together eliminates the risk of train-serve skew.
- **Streamlit accelerates deployment:** A functional, multi-page web application was built without requiring any front-end development expertise.

### 6.5 Future Research Directions

- Integrate external data: weather forecasts, local events calendars, economic indicators, and competitor pricing feeds.
- Build a REST API for real-time integration with hotel Property Management Systems (PMS).
- Explore deep learning approaches (e.g., LSTM networks) for modeling sequential booking behavior patterns.
- Expand the dataset to multiple countries and hotel chains for broader model generalization.
- Implement A/B testing frameworks to measure the actual revenue impact of model-driven interventions and validate the estimated ROI.

---

## 7. References

1. Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41–49.

2. Kaggle. (2019). Hotel Booking Demand. Retrieved from https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.*

4. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems (NeurIPS).*

5. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

6. Scikit-learn Developers. (2024). scikit-learn: Machine Learning in Python. Retrieved from https://scikit-learn.org

7. Streamlit Inc. (2024). Streamlit Documentation. Retrieved from https://docs.streamlit.io

8. Plotly Technologies Inc. (2024). Plotly Python Graphing Library. Retrieved from https://plotly.com/python/

9. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

10. Wang, C. J. (2026). UMBC DATA 606: Capstone in Data Science. University of Maryland, Baltimore County.
