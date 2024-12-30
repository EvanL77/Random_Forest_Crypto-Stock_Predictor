import sys
import pandas as pd
from pandas.tseries.offsets import BDay
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QLineEdit, QLabel, QHBoxLayout, QComboBox, QListWidget
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas
)
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf

class BacktestApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Backtesting App")
        self.setGeometry(100, 100, 800, 600)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)
        
        # Inputs
        input_layout = QHBoxLayout()
        layout.addLayout(input_layout)
        
        self.ticker_label = QLabel("Ticker:")
        self.ticker_input = QLineEdit("BTC-USD")
        input_layout.addWidget(self.ticker_label)
        input_layout.addWidget(self.ticker_input)
        
        self.start_label = QLabel("Start Date:")
        self.start_input = QLineEdit("2015-01-01")
        input_layout.addWidget(self.start_label)
        input_layout.addWidget(self.start_input)
        
        self.end_label = QLabel("End Date:")
        self.end_input = QLineEdit("2024-01-01")
        input_layout.addWidget(self.end_label)
        input_layout.addWidget(self.end_input)
        
        self.predictors = []
        self.predictors_label = QLabel("Selected Predictors: None")
        layout.addWidget(self.predictors_label)
        
        # Dropdown Menu for Feature Selection
        self.feature_dropdown = QComboBox()
        self.feature_dropdown.addItems([
            "SMA_50", "SMA_200", "Close_to_SMA_50", 
            "Close_to_SMA_200", "RSI", "Price_Change"
        ])
        layout.addWidget(self.feature_dropdown)
        
        # Add Button
        self.add_button = QPushButton("Add Predictor")
        self.add_button.clicked.connect(self.add_predictor)
        layout.addWidget(self.add_button)
        
        # Remove Button
        self.remove_button = QPushButton("Remove Predictor")
        self.remove_button.clicked.connect(self.remove_predictor)
        layout.addWidget(self.remove_button)
        
        # List to display current predictors
        self.predictor_list_widget = QListWidget()
        layout.addWidget(self.predictor_list_widget)
        
        # Button
        self.run_button = QPushButton("Run Backtest")
        self.run_button.clicked.connect(self.run_backtest)
        layout.addWidget(self.run_button)
        
        # Plot area
        self.canvas = FigureCanvas(plt.figure())
        layout.addWidget(self.canvas)
    
    def add_predictor(self):
        selected_feature = self.feature_dropdown.currentText()
        if selected_feature not in self.predictors:  # Prevent duplicates
            self.predictors.append(selected_feature)
        self.update_predictors_view()
    
    def remove_predictor(self):
        """Remove the selected predictor from the list."""
        selected_items = self.predictor_list_widget.selectedItems()
        if selected_items:
            for item in selected_items:
                self.predictors.remove(item.text())
                self.predictor_list_widget.takeItem(self.predictor_list_widget.row(item))
            self.update_predictors_view()
    
    def update_predictors_view(self):
        """Update the predictors label and list widget."""
        self.predictors_label.setText(f"Selected Predictors: {', '.join(self.predictors) if self.predictors else 'None'}")
        self.predictor_list_widget.clear()
        self.predictor_list_widget.addItems(self.predictors)
    
    def fetch_data(self, ticker, start_date, end_date):
        """Fetch historical data for the given ticker."""
        return yf.download(ticker, start=start_date, end=end_date)

    def feature_engineering(self, data):
        """Add features for backtesting."""
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        data['Close_to_SMA_50'] = data['Close'] / data['SMA_50']
        data['Close_to_SMA_200'] = data['Close'] / data['SMA_200']
        data['Price_Change'] = data['Close'].pct_change()
        
        def calculate_rsi(data, column="Close", window=14):
            delta = data[column].diff()  
            gain = delta.where(delta > 0, 0)  
            loss = -delta.where(delta < 0, 0)  
            avg_gain = gain.rolling(window=window, min_periods=1).mean()
            avg_loss = loss.rolling(window=window, min_periods=1).mean()
            rs = avg_gain / avg_loss  # Relative Strength
            rsi = 100 - (100 / (1 + rs))  # RSI formula
            return rsi
        
        data['RSI'] = calculate_rsi(data)
        return data.dropna()

    def train_model(self, train, predictors):
        """Train a Random Forest model."""
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train[predictors], train["Close"])
        return model

    def run_backtest(self):
        """Run the backtesting process."""
        ticker = self.ticker_input.text()
        start_date = self.start_input.text()
        end_date = self.end_input.text()
        
        # Fetch data
        data = self.fetch_data(ticker, start_date, end_date)
        data = self.feature_engineering(data)
        
        # Define predictors
        # predictors = ['SMA_50', 'SMA_200', 'Close_to_SMA_50', 'Close_to_SMA_200', 'Price_Change', 'RSI']
        predictors = self.predictors 
        
        # Split data for backtest
        train = data.iloc[:-100]
        test = data.iloc[-100:]
        
        # Train model
        model = self.train_model(train, predictors)
        test["Predictions"] = model.predict(test[predictors])
        
        # Predict future prices for the next year
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + BDay(1), periods=252, freq=BDay())
    
        future_data = pd.DataFrame(index=future_dates)
        for predictor in predictors:
            # Propagate the last known value of each predictor
            future_data[predictor] = data[predictor].iloc[-1]
    
        future_data["Predictions"] = model.predict(future_data[predictors])
        future_data["Close"] = None
        
        extended_test = pd.concat([test, future_data])
        
        # Plot results
        self.plot_results(extended_test)
    
    def plot_results(self, test):
        """Plot actual vs predicted prices."""
        ax = self.canvas.figure.subplots()
        ax.clear()
        
        actual = test.dropna(subset=["Close"])
        future = test.loc[test["Close"].isna()]  
        
        ax.plot(actual.index, actual["Close"], label="Actual Price", color="blue")
        ax.plot(test.index, test["Predictions"], label="Predicted Price", color="red", linestyle="--")
        ax.plot(future.index, future["Predictions"], label="Future Predictions", color="green", linestyle=":")
    
        ax.set_title("Backtesting Results")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        
        self.canvas.draw()

# Main entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = BacktestApp()
    main_window.show()
    sys.exit(app.exec())

