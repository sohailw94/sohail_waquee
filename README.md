Stock Trading Strategy with Sentiment Analysis and Bayesian Optimization
Project Overview
This project analyzes Reddit stock mentions using sentiment analysis to identify stocks with the most positive, neutral, or negative sentiment. It then performs technical analysis on these stocks using VWAP (Volume Weighted Average Price) and momentum indicators to generate buy/sell signals. To further improve the strategy, Bayesian optimization is implemented to optimize key parameters such as the momentum window and VWAP period.
Key Features
* Reddit Sentiment Analysis: Scrapes Reddit comments from the 'wallstreetbets' subreddit and analyzes stock sentiment based on mentions.
* VWAP and Momentum Indicators: Performs technical analysis using VWAP and momentum for buy/sell signals.
* Bayesian Optimization: Implements Bayesian optimization to automatically fine-tune parameters for better performance.
* Visualization: Provides charts of stock prices, VWAP, and buy/sell signals for selected stocks.
Installation
Prerequisites
Ensure you have the following installed:
1. Python 3.x - This project is written in Python.
2. PIP - The Python package manager to install dependencies.

API Credentials
You need Reddit API credentials to use the Reddit API via praw. Create an app on Reddit  and use your credentials (client_id, client_secret, user_agent) in the code.
Files
Ensure that you have the following files in the project folder:
1. company_tickers.json: Contains company names and ticker symbols for stock mention detection in Reddit comments. Make sure this file is present in your working directory.
Usage
1. Run the Code
After installing dependencies and configuring the Reddit API, you can run the Python script to:
* Scrape Reddit comments from 'wallstreetbets' for stock mentions.
* Perform sentiment analysis (positive, neutral, negative) on each stock.
* Visualize the top 5 mentioned stocks in each sentiment category.
* Download historical stock data for analysis using VWAP and momentum.
* Generate buy/sell signals based on trading indicators.
* Use Bayesian optimization to fine-tune the trading parameters.
2. Modify Optimization Parameters
The following parameters are optimized using Bayesian optimization:
* Momentum Window: Defines how far back momentum is calculated.
* VWAP Period: Defines the period over which VWAP is calculated.
Bayesian optimization will attempt to find the best combination of these parameters for improved buy/sell signals.

Output
* A heatmap showing the top 5 stock mentions by sentiment.
* Stock price charts with VWAP, momentum, and buy/sell signals for the top stocks.
* A summary table showing the cumulative momentum and buy/sell signals for the best-performing stocks.
* The best parameters optimized via Bayesian optimization.
Bayesian Optimization
Bayesian optimization is implemented to fine-tune the momentum and VWAP period for better buy/sell signals. It works by:
* Defining a search space for momentum window size and VWAP period.
* Evaluating performance based on cumulative buy/sell signals and momentum.
* Iterating through possible parameter values to find the optimal configuration.
This method helps improve the model's ability to generate accurate trading signals.
Visualization
The script generates the following visualizations:
* Heatmap: Displays the top 5 stock mentions for each sentiment category (positive, neutral, negative).
* Stock Charts: For each of the selected stocks, a plot shows the VWAP, stock price, and buy/sell signals over time.
License
This project is open-source and free to use under the MIT License.
