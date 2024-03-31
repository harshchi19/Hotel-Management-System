# Hotel-Management-System
This is a Hotel Management System that provides various analyses and forecasts based on the hotel's data. The system includes functionalities such as data visualization, ARIMA forecasting, analysis of food bills, and training a RandomForestClassifier for predicting bill types.
Setup

To run this system, ensure you have the following Python libraries installed:

    pandas
    streamlit
    seaborn
    matplotlib
    scikit-learn
    statsmodels

You can install these dependencies using pip:

pip install pandas streamlit seaborn matplotlib scikit-learn statsmodels

Usage

To use the Hotel Management System, follow these steps:

    Clone the repository or download the code files.
    Make sure you have the required dataset (data.csv) in the same directory as the code files.
    Open a terminal or command prompt and navigate to the directory containing the code files.
    Run the following command:

    streamlit run hotel_management_system.py
    Once the Streamlit server starts, you can interact with the system through your web browser.

Functionality
1. Visualize Data

    Visualizes data using various plots such as violin plots, bar plots, time series plots, and 2D histograms.

2. ARIMA Forecasting

    Forecasts the total amount of transactions using ARIMA (AutoRegressive Integrated Moving Average) model and displays the forecast for the next 2 months.

3. Analyze Food Bills

    Analyzes food bills data to identify the top 20 most in-demand food items.

4. Train RandomForestClassifier

    Trains a RandomForestClassifier to predict bill types based on features such as total amount, delivery type, and item.

Contributing

Contributions to this project are welcome. You can contribute by:

    Adding new features
    Improving existing code
    Fixing bugs
    Enhancing documentation

