# EV Vehicle Demand Prediction using Deep Learning
# Project Overview
The rapid adoption of electric vehicles (EVs) has created a strong need for accurate forecasting of future EV demand to support efficient charging infrastructure and energy planning. This project presents a deep learning–based EV demand forecasting system that predicts long-term EV adoption trends using historical data and an LSTM time-series model.
The system focuses on cumulative EV adoption and county-level forecasting, providing realistic, data-driven insights through an interactive web application.

# Objectives


Forecast future EV demand using deep learning techniques


Capture long-term EV adoption patterns from historical data


Provide county-level EV demand predictions


Support infrastructure planning and sustainable mobility


Deploy predictions through an interactive web application



# Key Features


LSTM-based long-term time-series forecasting


Cumulative EV adoption modeling (real charging demand)


County-wise EV demand prediction


Comparative multi-county growth analysis


Interactive Streamlit dashboard


End-to-end deployable system



# Why LSTM?
Electric vehicle adoption data is sequential and trend-based.
LSTM (Long Short-Term Memory) networks are well-suited for this task because they:


Learn long-term temporal dependencies


Handle time-series data effectively


Provide stable and accurate long-term forecasts



# Dataset


Source: Historical Electric Vehicle Registration Data


Region: Washington State (County-level)


Vehicle Type: Passenger Electric Vehicles


Key Columns: Date, State, County, EV Count


The dataset is cleaned, aggregated, and transformed into cumulative EV adoption to reflect real-world charging demand.

# Methodology


Data collection and preprocessing


Cumulative EV adoption calculation


Data normalization using Min-Max scaling


Time-series sequence creation (sliding window)


LSTM model training and validation


Model evaluation using RMSE and R² score


Deployment using Streamlit web application



#  Model Performance


Evaluation Metrics:


Root Mean Square Error (RMSE)


R-squared (R²) score




Achieved R² Score: ~ 0.96
This indicates strong learning of EV adoption trends and reliable forecasting performance.


# Web Application
The Streamlit-based dashboard allows users to:


Select a county


Choose forecast duration (up to 36 months)


Visualize historical and predicted EV demand


Compare EV growth across multiple counties



# End Users


Government transport departments


EV charging infrastructure planners


Power distribution companies


Smart city and urban planners


EV manufacturers and investors



# Technologies Used


Python


TensorFlow / Keras


Streamlit


Pandas & NumPy


Matplotlib


Scikit-learn



# How to Run the Project
1 Install Dependencies
pip install -r requirements.txt

2 Run the Application
streamlit run app.py


# Real-World Applications


EV charging station planning


Power grid load forecasting


Sustainable transportation planning


Smart city infrastructure development


Policy-making and investment analysis



# Future Enhancements


Include population growth and policy incentives


Extend forecasting to state and national levels


Integrate real-time data and cloud deployment


Add advanced visualization and analytics



# Conclusion
This project demonstrates the effective use of deep learning to forecast future electric vehicle demand. By combining an LSTM-based time-series model with an interactive web application, the system delivers a practical, scalable, and data-driven solution for future-ready EV infrastructure planning.
