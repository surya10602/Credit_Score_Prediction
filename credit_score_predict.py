import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import ast

def calculate_credit_scores(input_file, output_file):
    with open(input_file) as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    if isinstance(df['actionData'].iloc[0], str):
        df['actionData'] = df['actionData'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {})
    
    df['amount'] = df['actionData'].apply(lambda x: float(x.get('amount', 0)))
    df['assetPriceUSD'] = df['actionData'].apply(lambda x: float(x.get('assetPriceUSD', 1)))
    df['tx_value_usd'] = df['amount'] * df['assetPriceUSD']
    df['assetSymbol'] = df['actionData'].apply(lambda x: x.get('assetSymbol', 'unknown'))
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    features = pd.DataFrame(index=df['userWallet'].unique())
    
    user_activity = df.groupby('userWallet').agg({
        'timestamp': ['min', 'max', 'count'],
        'tx_value_usd': 'sum'
    })
    user_activity.columns = ['_'.join(col) for col in user_activity.columns] 
    features['tx_count'] = user_activity['timestamp_count']
    features['total_value'] = user_activity['tx_value_usd_sum']
    features['age_days'] = (user_activity['timestamp_max'] - user_activity['timestamp_min']).dt.days
    features['tx_freq'] = features['tx_count'] / (features['age_days'] + 1)
    
    tx_types = df.groupby(['userWallet', 'action'])['tx_value_usd'].sum().unstack(fill_value=0)
    for col in ['deposit', 'borrow', 'repay', 'redeem', 'liquidation']:
        col_name = f"{col}_ratio"
        if col in tx_types.columns:
            features[col_name] = tx_types[col] / features['total_value']
        else:
            features[col_name] = 0
    
    borrows = df[df['action'] == 'borrow'].copy()
    repays = df[df['action'] == 'repay'].copy()
    
    repayment_times = []
    for _, borrow in borrows.iterrows():
        try:
            asset_symbol = borrow['assetSymbol']
            matching_repays = repays[
                (repays['userWallet'] == borrow['userWallet']) & 
                (repays['assetSymbol'] == asset_symbol)
            ]
            
            if not matching_repays.empty:
                first_repay = matching_repays.iloc[0]
                repayment_time = (first_repay['timestamp'] - borrow['timestamp']).total_seconds() / 3600
                repayment_times.append({
                    'wallet': borrow['userWallet'], 
                    'repayment_time': repayment_time,
                    'borrow_value': borrow['tx_value_usd']
                })
        except Exception as e:
            print(f"Error processing borrow transaction: {e}")
            continue
    
    if repayment_times:
        repayment_df = pd.DataFrame(repayment_times)
        repayment_stats = repayment_df.groupby('wallet').agg({
            'repayment_time': ['mean', 'std'],
            'borrow_value': 'sum'
        })
        repayment_stats.columns = ['mean_repayment_time', 'repayment_time_std', 'total_borrowed']
        features = features.join(repayment_stats, how='left')
    
    time_features = df.groupby('userWallet')['timestamp'].agg(
        hour_of_day=lambda x: x.dt.hour.mean(),
        day_of_week=lambda x: x.dt.dayofweek.mean()
    )
    features = features.join(time_features)
    
    features = features.fillna(0)
    
    features['activity_score'] = np.log1p(features['tx_count'] + features['total_value']/1e6) * 10
    features['longevity_score'] = np.minimum(features['age_days'] * 0.5, 100)
    features['repayment_score'] = np.where(
        features['mean_repayment_time'] > 0,
        100 / (1 + np.log1p(features['mean_repayment_time'])),
        50
    )
    
    features['liquidation_penalty'] = features.get('liquidation_ratio', 0) * -200
    features['risk_penalty'] = (features.get('borrow_ratio', 0) * -50) * np.log1p(features['total_borrowed']/1e4)
    
    anomaly_features = features[['tx_freq', 'hour_of_day', 'day_of_week']].fillna(0)
    clf = IsolationForest(contamination=0.05, random_state=42)
    features['anomaly_score'] = clf.fit_predict(anomaly_features)
    features['anomaly_penalty'] = np.where(features['anomaly_score'] == -1, -100, 0)
    
    base_score = (
        features['activity_score'] + 
        features['longevity_score'] + 
        features['repayment_score'] + 
        features['liquidation_penalty'] + 
        features['risk_penalty'] + 
        features['anomaly_penalty']
    )
    
    scaler = MinMaxScaler(feature_range=(0, 1000))
    scores = scaler.fit_transform(base_score.values.reshape(-1, 1))
    features['credit_score'] = scores.round().astype(int)
    
    result = features[['credit_score']].reset_index().rename(columns={'index': 'wallet'})
    result_dict = result.set_index('wallet')['credit_score'].to_dict()
    
    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    return result_dict


import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_score_distribution(score_file, output_image='score_distribution.png'):
    try:
        with open(score_file) as f:
            scores = json.load(f)
        
        df = pd.DataFrame.from_dict(scores, orient='index', columns=['score'])
        
        plt.style.use('ggplot')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bins = np.arange(0, 1100, 100)
        labels = [f"{i}-{i+99}" for i in range(0, 1000, 100)]
        df['range'] = pd.cut(df['score'], bins=bins, labels=labels, right=False)
        counts = df['range'].value_counts().sort_index()
        
        colors = []
        for label in counts.index:
            score_min = int(label.split('-')[0])
            if score_min < 400:
                colors.append('#ff6b6b')  
            elif score_min >= 800:
                colors.append('#51cf66')  
            else:
                colors.append('#fcc419')  
        
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=0.7)
        
        ax.set_title('Aave V2 Wallet Credit Score Distribution', pad=20, fontsize=14)
        ax.set_xlabel('Score Range', labelpad=10)
        ax.set_ylabel('Number of Wallets', labelpad=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.xticks(rotation=45, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        plt.savefig(output_image, dpi=120, bbox_inches='tight')
        
        plt.show()
        
        plt.close()
        
        print(f"Successfully generated and displayed {output_image}")
        return True
    
    except Exception as e:
        print(f"Error generating graph: {str(e)}")
        return False


if __name__ == "__main__":
    calculate_credit_scores('user-wallet-transactions.json', 'credit_score.json')
    plot_score_distribution('credit_score.json')
