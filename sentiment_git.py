#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:21:49 2024

@author: sohailwaquee
"""

import praw
from collections import defaultdict
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import schedule
import time
from datetime import datetime
import yfinance as yf
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

# analysing stock mentions and classifying them as poitive, neutral and negative
#A file containing all tickers and firm title is used for lookup
file_path="/Users/sohailwaquee/Downloads/company_tickers.json"

with open(file_path,'r') as file:
    data=json.load(file)
df=pd.DataFrame(data)
df1=df.T
df1=df1.drop('cik_str',axis=1)
df1['title'] = df1['title'].apply(lambda x: x.split()[0] if isinstance(x, str) else x)
df1['title']=df1['title'].str.lower()

tickr_to_company=dict(zip(df1['ticker'],df1['title']))
company_to_ticker={row['title']:row['ticker'] for _,row in df1.iterrows()}

# Reddit API credentials 
reddit = praw.Reddit(client_id="your_reddit_id",client_secret="your_secret_below the id",user_agent="name of reddit app/0.0.1 by reddit_username")

positive_stock_mentions=defaultdict(int)
neutral_stock_mentions=defaultdict(int)
negative_stock_mentions=defaultdict(int)

subred=reddit.subreddit('wallstreetbets')
query='stocks'
# going through 100 reddit posts and all comments and classifying
for submission in subred.search(query,limit=100):
    submission.comments.replace_more(limit=0)
    
    for comment in submission.comments.list():
        comment_body=comment.body.lower()
        
        analysis=TextBlob(comment_body)
        senti=analysis.sentiment.polarity
        
        
        for title, ticker in company_to_ticker.items():
            if title in comment_body:
                if senti > 0:
                    positive_stock_mentions[ticker] += 1
                elif senti == 0:
                    neutral_stock_mentions[ticker] += 1
                else:
                    negative_stock_mentions[ticker] += 1
        for word in comment_body.split():
            stock_ticker = word.upper()
            if stock_ticker in tickr_to_company:
                if senti > 0:
                    positive_stock_mentions[stock_ticker] += 1
                elif senti == 0:
                    neutral_stock_mentions[stock_ticker] += 1
                else:
                    negative_stock_mentions[stock_ticker] += 1
        
positive_heatmap_data=pd.DataFrame.from_dict(positive_stock_mentions,orient='index',columns=['Positive Mentions'])
neutral_heatmap_data=pd.DataFrame.from_dict(neutral_stock_mentions,orient='index',columns=['Neutral Mentions'])
negative_heatmap_data=pd.DataFrame.from_dict(negative_stock_mentions,orient='index',columns=['Negative Mentions'])

positive_heatmap_data['Positive Mentions'] = pd.to_numeric(positive_heatmap_data['Positive Mentions'], errors='coerce').fillna(0).astype(int)
neutral_heatmap_data['Neutral Mentions'] = pd.to_numeric(neutral_heatmap_data['Neutral Mentions'], errors='coerce').fillna(0).astype(int)
negative_heatmap_data['Negative Mentions'] = pd.to_numeric(negative_heatmap_data['Negative Mentions'], errors='coerce').fillna(0).astype(int)

combined_heatmap_data1=positive_heatmap_data.join(neutral_heatmap_data,how='outer').fillna(0)
combined_heatmap_data=combined_heatmap_data1.join(negative_heatmap_data,how='outer').fillna(0)


# Getting the top 5 most mentioned stocks for each sentiment category
top_5_positive = positive_heatmap_data.nlargest(5, 'Positive Mentions')
top_5_neutral = neutral_heatmap_data.nlargest(5, 'Neutral Mentions')
top_5_negative = negative_heatmap_data.nlargest(5, 'Negative Mentions')

# Combining the top 5 for each sentiment into one DataFrame for plotting
top_5_combined_data = top_5_positive.join(top_5_neutral, how='outer').join(top_5_negative, how='outer').fillna(0)
# Plot the heatmap for the top 5 stocks in each sentiment category
plt.figure(figsize=(13, 9))
sns.heatmap(top_5_combined_data, annot=True, fmt='.0f', cmap="YlGnBu", cbar_kws={'label': 'Mentions'})

plt.title('Top 5 Stock Mentions by Sentiment', fontsize=16)
plt.xlabel('Sentiment Baskets', fontsize=14)
plt.ylabel('Stocks', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()  
plt.show()

# Exporting full data for all stocks to an Excel file
excel_file_path = "/Users/sohailwaquee/Downloads/stock_mentions_by_sentiment.xlsx"
combined_heatmap_data.to_excel(excel_file_path)

positive_heatmap_data = positive_heatmap_data.sort_values(by='Positive Mentions', ascending=False).head(5)
tickers = positive_heatmap_data.index.tolist()

#Calculating VWAP and Momentum and generating signals
def calculate_vwap(df):
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['Typical_Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df

def calc_momentum(df, period=10):
    df['Momentum'] = df['Close'] - df['Close'].shift(period)
    return df

def gen_sig(df):
    buy_sig = (df['Close'] > df['VWAP']) & (df['Momentum'] > 0)
    sell_sig = (df['Close'] < df['VWAP']) & (df['Momentum'] < 0)
    df['Signal'] = np.where(buy_sig, "Buy", np.where(sell_sig, "Sell", "Hold"))
    return df

# Defining the objective function for Bayesian optimization
def objective(params):
    momentum_period = params[0]
    total_return = 0
    
    for ticker in tickers:
        df = yf.download(ticker, period='6mo', interval='1d')
        df = calculate_vwap(df)
        df = calc_momentum(df, period=momentum_period)
        df = gen_sig(df)
        
        buy_signals = df[df['Signal'] == 'Buy']
        sell_signals = df[df['Signal'] == 'Sell']
        
        
        min_signals = min(len(buy_signals), len(sell_signals))
        
        if min_signals > 0:
            
            buy_prices = buy_signals['Close'].values[:min_signals]
            sell_prices = sell_signals['Close'].values[:min_signals]
            
           
            returns = (sell_prices - buy_prices).sum()
            total_return += returns
    
    return -total_return

# Defining search space
search_space = [Integer(5, 20, name='momentum_period')]

# Bayesian Optimization
res = gp_minimize(objective, search_space, n_calls=30, random_state=0)

best_momentum_period = res.x[0]

# Re-run the trading strategy with optimized parameters
stock_data_dict = {}

for ticker in tickers:
    stock_data = yf.download(ticker, period='6mo', interval='1d')
    stock_data = calculate_vwap(stock_data)
    stock_data = calc_momentum(stock_data, period=best_momentum_period)
    stock_data = gen_sig(stock_data)
    stock_data_dict[ticker] = stock_data

# Displaying optimized parameters
print(f"Optimized momentum period: {best_momentum_period}")

# Visualization of best stock signals
for ticker, stock_data in stock_data_dict.items():
    plt.figure(figsize=(12, 8))
    plt.plot(stock_data.index.to_numpy(), stock_data['Close'].to_numpy(), label='Close Price', color="blue")
    plt.plot(stock_data.index.to_numpy(), stock_data['VWAP'].to_numpy(), label="VWAP", color='orange')


    buy_signal = stock_data['Signal'] == 'Buy'
    sell_signal = stock_data['Signal'] == 'Sell'
    
    plt.scatter(stock_data.index[buy_signal], stock_data['Close'][buy_signal], marker='^', color='green', label='buy signal', s=100)
    plt.scatter(stock_data.index[sell_signal], stock_data['Close'][sell_signal], marker='v', color='red', label='sell signal', s=100)
    
    plt.title(f'VWAP, Momentum (Optimized) and Signals for {ticker}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
