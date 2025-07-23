# Credit_Score_Prediction
A machine learning-based credit scoring system that evaluates wallet addresses interacting with Aave V2, generating credit scores from 0-1000 based on transaction behavior patterns.

## Methodology
### Score Calculation Approach
1. **Feature Engineering**:
   - Transaction frequency and recency
   - Deposit/borrow ratios
   - Repayment timeliness
   - Liquidation history
   - Temporal patterns (time of day/week)

2. **Scoring Model**:
   ```python
   base_score = (
       activity_score + 
       longevity_score + 
       repayment_score + 
       liquidation_penalty + 
       risk_penalty + 
       anomaly_penalty
   )
3. **Scores scaled to 0-1000 range using MinMaxScaler**

4. **Isolation Forest for anomaly detection**

## System Architecture:

<img width="400" height="600" alt="system_architecture" src="https://github.com/user-attachments/assets/44cf2185-7dfe-4948-8571-eae27fa5701f" />

#### Data Pipeline:

- Input: Aave V2 transaction JSON

- Processing: Clean, normalize, and extract features

- Output: Processed DataFrame with engineered features

#### Scoring Engine:

- Calculates base score components

- Applies penalties for risky behavior

- Normalizes final scores

#### Analysis Module:

- Generates distribution visualizations

- Comparative behavior analysis

- Risk category breakdowns
