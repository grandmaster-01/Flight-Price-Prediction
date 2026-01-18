# Flight Price Prediction ✈️

A machine learning project that predicts flight ticket prices using exploratory data analysis (EDA) and feature engineering techniques. This project analyzes various factors that influence airline ticket prices and builds a predictive model to forecast flight costs.

## 📋 Project Overview

This project demonstrates a complete machine learning workflow for predicting flight prices in the Indian aviation market. It includes comprehensive data preprocessing, feature engineering, exploratory data analysis, and model training using Random Forest Regressor.

The goal is to help users understand price patterns and predict flight costs based on various parameters such as airline, journey date, source, destination, and other relevant features.

## ✨ Key Features

- **Data Preprocessing**: Handling missing values, outlier detection and treatment
- **Feature Engineering**: 
  - Temporal feature extraction (journey dates, months, day of week)
  - Airline and route encoding
  - Journey duration calculations
  - Arrival/departure time features
- **Exploratory Data Analysis**: 
  - Distribution analysis of flight prices
  - Correlation analysis between features
  - Interactive visualizations
- **Machine Learning Model**: Random Forest Regressor for price prediction
- **Feature Importance Analysis**: Identification of key factors influencing flight prices

## 🛠️ Technologies Used

- **Python 3.9.7**
- **Data Manipulation**: pandas, NumPy
- **Machine Learning**: scikit-learn (Random Forest)
- **Visualization**: Matplotlib, Seaborn, Altair
- **Development Environment**: Jupyter Notebook
- **Platform**: Google Colab compatible

## 📊 Dataset

The project uses flight price data from the Indian aviation market stored in `Flight_Price_Prediction.xlsx`.

### Features:
- **Airline**: Name of the airline carrier
- **Date_of_Journey**: Date of travel
- **Source**: Departure city (Delhi, Kolkata, Chennai, etc.)
- **Destination**: Arrival city (Cochin, Bangalore, etc.)
- **Route**: Flight route information
- **Dep_Time**: Departure time
- **Arrival_Time**: Arrival time
- **Duration**: Total journey duration
- **Total_Stops**: Number of stops
- **Additional_Info**: Additional flight information

### Target Variable:
- **Price**: Flight ticket price (in Indian Rupees)

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn altair
```

### Running the Project

1. Clone the repository:
```bash
git clone https://github.com/grandmaster-01/Flight-Price-Prediction.git
cd Flight-Price-Prediction
```

2. Open the Jupyter Notebook:
```bash
jupyter notebook Flight_Price_Prediction.ipynb
```

3. Run all cells sequentially to:
   - Load and explore the dataset
   - Perform data preprocessing
   - Engineer features
   - Train the Random Forest model
   - Visualize results and feature importance

## 📈 Model Performance

The project uses **Random Forest Regressor** as the primary prediction model, which:
- Captures complex non-linear relationships between features
- Provides feature importance rankings
- Handles both numerical and categorical variables effectively

Feature importance analysis reveals which factors most significantly influence flight pricing, helping to understand the dynamics of airline ticket costs.

## 📁 Project Structure

```
Flight-Price-Prediction/
│
├── Flight_Price_Prediction.ipynb   # Main Jupyter notebook with code
├── Flight_Price_Prediction.xlsx    # Dataset file
└── README.md                        # Project documentation
```

## 🔍 Key Insights

The project reveals important patterns in flight pricing through:
- Feature importance visualization using Mean Decrease in Impurity (MDI)
- Correlation analysis between different flight attributes
- Distribution patterns of prices across various airlines and routes
- Impact of temporal factors (date, time) on pricing

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📝 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Dataset source: Indian aviation market data
- Built using scikit-learn and other open-source libraries

---

**Note**: This project is for educational and research purposes. For production use, consider additional model validation, cross-validation, and hyperparameter tuning.
