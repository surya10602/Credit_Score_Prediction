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

