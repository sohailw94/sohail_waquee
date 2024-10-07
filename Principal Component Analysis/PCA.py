import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import xlwings as xw

# Downloading historical data
tickers = ["TSLA", "GOOGL", "GS", "XOM", "AMZN", "PG"]
data = yf.download(tickers, start='2019-01-01', end='2024-09-30')
df = data['Adj Close']

# Calculating daily returns and standardize
returns = df.pct_change().dropna()
returns_standard = (returns - returns.mean()) / returns.std()

pca = PCA(n_components=len(tickers))
pca.fit(returns_standard)

# Principal components and explained variance
principle_components = pca.components_
components1=[str(x+1) for x in range(pca.n_components_)]
explained_variance = pca.explained_variance_ratio_
explained_var_pct=explained_variance*100

pc_df = pd.DataFrame(principle_components, columns=[f"PC{i+1}" for i in range(len(tickers))], index=tickers)
plt.figure(figsize=(10,6))
plt.bar(components1,explained_var_pct)
plt.title("Ratio of Explained Variance")
plt.xlabel("Principle Component #")
plt.ylabel("%")
plt.show()

xw.view(pc_df)
print(f"Explained Variance: {explained_variance}")

# Transforming returns using PCA
scores = pca.transform(returns_standard)
scores_df = pd.DataFrame(scores, index=returns_standard.index, columns=[f"PC{i+1}" for i in range(len(tickers))])

# Add the first three principal component to the original DataFrame
df = df.join(scores_df[['PC1', 'PC2','PC3']], how='left')



# Calculating rolling mean and standard deviation for PC1,PC2 and PC3.
window = 30
df['mean_PC1'] = df['PC1'].rolling(window=window).mean()
df['std_PC1'] = df['PC1'].rolling(window=window).std()
df['mean_PC2'] = df['PC2'].rolling(window=window).mean()
df['std_PC2'] = df['PC2'].rolling(window=window).std()
df['mean_PC3'] = df['PC3'].rolling(window=window).mean()
df['std_PC3'] = df['PC3'].rolling(window=window).std()

# Filling missing values in rolling calculations
df['mean_PC1'].fillna(method='bfill', inplace=True)
df['std_PC1'].fillna(method='bfill', inplace=True)
df['mean_PC2'].fillna(method='bfill', inplace=True)
df['std_PC2'].fillna(method='bfill', inplace=True)
df['mean_PC3'].fillna(method='bfill', inplace=True)
df['std_PC3'].fillna(method='bfill', inplace=True)


#z-scores for PC1
df['z_score_PC1'] = (df['PC1'] - df['mean_PC1']) / df['std_PC1']
df['z_score_PC2'] = (df['PC2'] - df['mean_PC2']) / df['std_PC2']
df['z_score_PC3'] = (df['PC3'] - df['mean_PC3']) / df['std_PC3']

# Filling missing values in z_score_PC1
df['z_score_PC1'].fillna(0, inplace=True)
df['z_score_PC2'].fillna(0, inplace=True)
df['z_score_PC3'].fillna(0, inplace=True)

# Trading signals thresholds by trial and error,due to bullish nature of stocks, a higher exit threshold discourages exit faster.
entry_threshold_pc1 = 1
exit_threshold_pc1 = 3
entry_threshold_pc2 = 0.9
exit_threshold_pc2 = 2.95
entry_threshold_pc3 = 0.8
exit_threshold_pc3 = 2.9

df['long'] = np.where(
    (df['z_score_PC1'] < entry_threshold_pc1) | 
    (df['z_score_PC2'] < entry_threshold_pc2) | 
    (df['z_score_PC3'] < entry_threshold_pc3), 
    1, 
    0
)
df['short'] = np.where(
    (df['z_score_PC1'] > exit_threshold_pc1) | 
    (df['z_score_PC2'] > exit_threshold_pc2) | 
    (df['z_score_PC3'] > exit_threshold_pc3), 
    -1, 
    0
)

df['Position'] = df['long'] + df['short']

# Filling missing values in Position due to shift
df['Position'] = df['Position'].shift(1)
df['Position'].fillna(0, inplace=True)

# Calculating strategy returns
df['Strategy_Return'] = df['Position'] * returns.mean(axis=1)
df['Strategy_Return'].fillna(0, inplace=True)

# Calculating cumulative returns
df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod() - 1
df['Cumulative_Market_Return'] = (1 + returns.mean(axis=1)).cumprod() - 1

# Filling missing values in cumulative returns
df['Cumulative_Market_Return'].fillna(method='bfill', inplace=True)

# Printing final performance metrics
print(df[['Cumulative_Strategy_Return', 'Cumulative_Market_Return']].tail)
xw.view(df[['Cumulative_Strategy_Return','Cumulative_Market_Return']])


# Plotting cumulative returns for market and strategy.
plt.figure(figsize=(14, 7))
plt.plot(df.index.values, df['Cumulative_Strategy_Return'].values, label='Strategy Cumulative Return')
plt.plot(df.index.values, df['Cumulative_Market_Return'].values, label='Market Cumulative Return')
plt.title('Strategy vs Market Performance')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()
