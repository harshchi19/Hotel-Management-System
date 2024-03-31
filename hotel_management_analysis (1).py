import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import LabelEncoder

# Load data
@st.cache_data
def load_data():
    dtypes = {'Column6': str, 'Column7': str}  # Replace 'Column6' and 'Column7' with actual column names
    df = pd.read_csv('D:\\bekar\\KHUSHI PROJECT\\data (2).csv', dtype={'Column6': str, 'Column7': str})
    return df

# Preprocess data
def preprocess_data(df):
    # Drop unnecessary columns
    df.drop(columns=['Quantity', 'PriceperQuantity', 'Kitchen_place', 'Sales_Type'], inplace=True)
    # Convert 'Date' column to datetime with the appropriate format
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    return df

# Visualize data
def visualize_data(df):
    st.subheader('Visualize Data')
    # Violin plot
    plt.figure(figsize=(12, 1.2 * len(df['Delivery_Type'].unique())))
    sns.violinplot(x='Total_Amount', y='Delivery_Type', data=df, inner='box', palette='Dark2', hue='Delivery_Type')  # Set hue='Delivery_Type'
    st.pyplot()
    # Bar plot
    st.subheader('Bar Plot')
    df.groupby('Bill_Type').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
    plt.gca().spines[['top', 'right']].set_visible(False)
    st.pyplot()
    # Time series plot
    st.subheader('Time Series Plot')
    fig, ax = plt.subplots(figsize=(15, 8))
    df_sorted = df.sort_values('Date', ascending=True)
    for series_name, series in df_sorted.groupby('Bill_Type'):
        ax.plot(series['Date'], series['Total_Amount'], label=series_name)
    ax.legend(title='Bill_Type')
    st.pyplot()
    # 2D histogram
    st.subheader('2D Histogram')
    df_2dhist = pd.DataFrame({
        x_label: grp['Delivery_Type'].value_counts()
        for x_label, grp in df.groupby('Bill_Type')
    })
    sns.heatmap(df_2dhist, cmap='viridis')
    st.pyplot()

# ARIMA Forecasting
def arima_forecast(df):
    st.subheader('ARIMA Forecasting')
    # Convert 'Date' column to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    # Aggregate total amount by month
    monthly_total = df['Total_Amount'].resample('ME').sum()  # Change 'M' to 'ME'
    # Split data into train and test sets
    train = monthly_total.iloc[:-2]
    test = monthly_total.iloc[-2:]
    # Fit ARIMA model
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    # Forecast next 2 months
    forecast = model_fit.forecast(steps=2)
    st.write("Forecast for next 2 months:")
    st.write(forecast)
    # Plot actual vs. predicted values
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.plot(test.index, forecast, label='Forecast')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Total Amount')
    plt.title('ARIMA Forecast')
    st.pyplot()

# Analyze Food Bills
def analyze_food_bills(df):
    st.subheader('Analyze Food Bills')
    food_df = df[df['Bill_Type'] == 'Food Bills']
    demand_2014_to_2023 = food_df.groupby('Item').size().sort_values(ascending=False)
    st.write("Top 20 Most In-Demand Food Items (2014-2023)")
    st.bar_chart(demand_2014_to_2023.head(20))

# Train a RandomForestClassifier
def train_random_forest(df):
    st.subheader('Train RandomForestClassifier')
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    # Encode categorical variables
    df['Delivery_Type'] = label_encoder.fit_transform(df['Delivery_Type'])
    df['Item'] = label_encoder.fit_transform(df['Item'])
    # Split data into features (X) and target variable (y)
    X = df[['Total_Amount', 'Delivery_Type', 'Item']]
    y = df['Bill_Type']
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize and train a Random Forest classifier
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    # Predict the target variable for the test set
    y_pred = classifier.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy:", accuracy)
    st.write(classification_report(y_test, y_pred))
    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classifier.classes_,
                yticklabels=classifier.classes_)
    st.pyplot()


# Main function
def main():
    st.title('Hotel Management Analysis')
    df = load_data()
    df = preprocess_data(df)
    visualize_data(df)
    arima_forecast(df)
    analyze_food_bills(df)
    train_random_forest(df)


if __name__ == '__main__':
    main()
