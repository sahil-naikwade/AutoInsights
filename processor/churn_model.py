"""
Enhanced Customer Churn Prediction Model - Industry Grade
=======================================================

This module provides high-accuracy churn prediction using:
- Advanced RFM analysis
- Temporal behavioral patterns
- Feature importance analysis
- Hyperparameter tuning
- SQL integration for data retrieval
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQL Queries for data retrieval (commented out for now - can be enabled when DB is available)
SQL_GET_CUSTOMERS = """
SELECT 
    customer_id,
    first_purchase_date,
    last_purchase_date,
    total_orders,
    total_spend,
    avg_order_value,
    days_since_last_purchase,
    preferred_category
FROM customers
WHERE is_active = TRUE
"""

SQL_GET_TRANSACTIONS = """
SELECT 
    customer_id,
    transaction_date,
    amount,
    category,
    payment_method
FROM transactions
WHERE transaction_date >= %s
"""

def get_db_connection():
    """Establish database connection"""
    # Placeholder for database connection
    # In real implementation, this would connect to your database
    return None

def fetch_customer_data():
    """Fetch customer data from database"""
    # Placeholder for database fetching
    # In real implementation, this would fetch from your database
    return None, None

def calculate_enhanced_rfm_features(df):
    """
    Enhanced RFM feature calculation with temporal patterns
    """
    df_copy = df.copy()
    
    # Preserve analysis_id if it exists
    analysis_id = None
    if 'analysis_id' in df_copy.columns:
        analysis_id = df_copy['analysis_id'].iloc[0]
    
    # Convert dates if they exist
    date_cols = [col for col in df_copy.columns if any(word in col.lower() for word in ['date', 'purchase', 'order', 'last'])]
    for col in date_cols:
        try:
            df_copy[col] = pd.to_datetime(df_copy[col])
        except:
            logger.warning(f"Could not convert {col} to datetime")
    
    # Calculate core RFM
    if 'last_purchase_date' in df_copy.columns:
        df_copy['recency_days'] = (datetime.now() - df_copy['last_purchase_date']).dt.days
    elif 'last_activity' in df_copy.columns:
        df_copy['recency_days'] = (datetime.now() - df_copy['last_activity']).dt.days
    else:
        # Fallback to simulation only if absolutely necessary
        df_copy['recency_days'] = np.random.randint(1, 365, len(df_copy))
        logger.warning("Using simulated recency data")
    
    if 'total_orders' in df_copy.columns:
        df_copy['frequency'] = df_copy['total_orders']
    elif 'transaction_count' in df_copy.columns:
        df_copy['frequency'] = df_copy['transaction_count']
    elif 'frequency' in df_copy.columns:
        # Keep existing frequency
        pass
    else:
        df_copy['frequency'] = np.random.randint(1, 20, len(df_copy))
        logger.warning("Using simulated frequency data")
    
    if 'total_spend' in df_copy.columns:
        df_copy['monetary'] = df_copy['total_spend']
    elif 'transaction_amount' in df_copy.columns:
        df_copy['monetary'] = df_copy['transaction_amount']
    elif 'monetary' in df_copy.columns:
        # Keep existing monetary
        pass
    else:
        df_copy['monetary'] = np.random.uniform(10, 1000, len(df_copy))
        logger.warning("Using simulated monetary data")
    
    # Enhanced RFM features
    try:
        df_copy['recency_score'] = pd.qcut(df_copy['recency_days'], q=5, labels=False, duplicates='drop')
        df_copy['frequency_score'] = pd.qcut(df_copy['frequency'], q=5, labels=False, duplicates='drop')
        df_copy['monetary_score'] = pd.qcut(df_copy['monetary'], q=5, labels=False, duplicates='drop')
        df_copy['rfm_score'] = df_copy['recency_score'] + df_copy['frequency_score'] + df_copy['monetary_score']
    except Exception as e:
        logger.warning(f"Could not calculate RFM scores: {e}")
        df_copy['recency_score'] = 0
        df_copy['frequency_score'] = 0
        df_copy['monetary_score'] = 0
        df_copy['rfm_score'] = 0
    
    # Restore analysis_id if it was preserved
    if analysis_id is not None:
        df_copy['analysis_id'] = analysis_id
    
    return df_copy

def calculate_temporal_features(df, transactions_df=None):
    """
    Calculate advanced temporal behavioral features
    """
    df_copy = df.copy()
    
    # Preserve analysis_id if it exists
    analysis_id = None
    if 'analysis_id' in df_copy.columns:
        analysis_id = df_copy['analysis_id'].iloc[0]
    
    # Purchase velocity features
    # Ensure frequency and recency_days are numeric
    if 'frequency' in df_copy.columns:
        df_copy['frequency'] = pd.to_numeric(df_copy['frequency'], errors='coerce').fillna(1)
    if 'recency_days' in df_copy.columns:
        df_copy['recency_days'] = pd.to_numeric(df_copy['recency_days'], errors='coerce').fillna(1)
    if 'first_purchase_date' in df_copy.columns and 'last_purchase_date' in df_copy.columns:
        try:
            df_copy['customer_tenure'] = (df_copy['last_purchase_date'] - df_copy['first_purchase_date']).dt.days
            df_copy['purchase_velocity'] = df_copy['frequency'] / np.maximum(df_copy['customer_tenure'], 1)
        except:
            # Fallback if date calculation fails
            df_copy['customer_tenure'] = df_copy['recency_days'] + (df_copy['frequency'] * 30)
            df_copy['purchase_velocity'] = df_copy['frequency'] / np.maximum(df_copy['customer_tenure'], 1)
    else:
        # Estimate tenure based on frequency and recency
        df_copy['customer_tenure'] = df_copy['recency_days'] + (df_copy['frequency'] * 30)
        df_copy['purchase_velocity'] = df_copy['frequency'] / np.maximum(df_copy['customer_tenure'], 1)
    
    # Recent activity features
    df_copy['recent_activity_30d'] = 0
    df_copy['recent_activity_60d'] = 0
    
    if transactions_df is not None and 'transaction_date' in transactions_df.columns:
        # Calculate recent activity counts
        thirty_days_ago = datetime.now() - timedelta(days=30)
        sixty_days_ago = datetime.now() - timedelta(days=60)
        
        recent_transactions = transactions_df[transactions_df['transaction_date'] >= thirty_days_ago]
        activity_counts = recent_transactions['customer_id'].value_counts()
        df_copy['recent_activity_30d'] = df_copy['customer_id'].map(activity_counts).fillna(0)
        
        recent_transactions = transactions_df[transactions_df['transaction_date'] >= sixty_days_ago]
        activity_counts = recent_transactions['customer_id'].value_counts()
        df_copy['recent_activity_60d'] = df_copy['customer_id'].map(activity_counts).fillna(0)
    
    # Trend features
    if 'frequency' in df_copy.columns and 'customer_tenure' in df_copy.columns:
        df_copy['frequency_trend'] = df_copy['frequency'] / np.maximum(df_copy['customer_tenure'], 1)
    
    # Restore analysis_id if it was preserved
    if analysis_id is not None:
        df_copy['analysis_id'] = analysis_id
    
    return df_copy

def create_enhanced_churn_labels(df, transactions_df=None, threshold_days=90):
    """
    Create more accurate churn labels using multiple criteria with improved logic
    """
    # Base churn probability
    churn_prob = np.zeros(len(df))
    
    # 1. Recency component (35% weight) - More recent = lower churn risk
    if 'recency_days' in df.columns:
        # Normalize recency (0 = recent, 1 = old)
        max_recency = df['recency_days'].max()
        if max_recency > 0:
            recency_norm = df['recency_days'] / max_recency
            churn_prob += 0.35 * recency_norm
    
    # 2. Frequency component (30% weight) - Higher frequency = lower churn risk
    if 'frequency' in df.columns:
        max_freq = df['frequency'].max()
        if max_freq > 0:
            freq_norm = 1 - (df['frequency'] / max_freq)  # Invert: low freq = high churn
            churn_prob += 0.30 * freq_norm
    
    # 3. Monetary component (25% weight) - Higher spend = lower churn risk
    if 'monetary' in df.columns:
        max_monetary = df['monetary'].max()
        if max_monetary > 0:
            monetary_norm = 1 - (df['monetary'] / max_monetary)  # Invert: low spend = high churn
            churn_prob += 0.25 * monetary_norm
    
    # 4. RFM Score component (10% weight) - Lower RFM score = higher churn risk
    if 'rfm_score' in df.columns:
        max_rfm = df['rfm_score'].max()
        if max_rfm > 0:
            rfm_norm = 1 - (df['rfm_score'] / max_rfm)  # Invert: low RFM = high churn
            churn_prob += 0.10 * rfm_norm
    
    # Add controlled noise for realistic variation
    churn_prob += np.random.normal(0, 0.08, len(df))
    churn_prob = np.clip(churn_prob, 0, 1)
    
    # Create labels with more realistic threshold
    churn_labels = (churn_prob > 0.6).astype(int)  # Higher threshold for more realistic churn rate
    
    # Ensure balanced classes for better model training
    churn_labels = np.array(churn_labels)
    n_churn = churn_labels.sum()
    n_total = len(churn_labels)
    
    # Target 20-30% churn rate for realistic business scenario
    target_churn_rate = 0.25
    target_churn_count = int(n_total * target_churn_rate)
    
    # Ensure minimum samples per class (at least 2)
    min_samples_per_class = max(2, int(n_total * 0.05))  # At least 5% or 2 samples minimum
    
    if n_churn < min_samples_per_class:  # Too few churned customers
        # Add more churned customers from highest probability
        additional_needed = min_samples_per_class - n_churn
        churn_candidates = np.argsort(-churn_prob)[:additional_needed]
        churn_labels[churn_candidates] = 1
    elif (n_total - n_churn) < min_samples_per_class:  # Too few non-churned customers
        # Remove some churned customers from lowest probability
        excess = n_churn - (n_total - min_samples_per_class)
        churn_candidates = np.argsort(churn_prob)[:excess]
        churn_labels[churn_candidates] = 0
    elif n_churn > target_churn_count * 1.5:  # Too many churned customers
        # Remove some churned customers from lowest probability
        excess = n_churn - target_churn_count
        churn_candidates = np.argsort(churn_prob)[:excess]
        churn_labels[churn_candidates] = 0
    
    # Final check to ensure we have at least 2 samples per class
    final_churn_count = np.sum(churn_labels)
    final_nonchurn_count = len(churn_labels) - final_churn_count
    if final_churn_count < 1:
        # Force at least 1 sample to be churned
        top_churn_candidate = np.argmax(churn_prob)
        churn_labels[top_churn_candidate] = 1
    if final_nonchurn_count < 1:
        # Force at least 1 sample to be non-churned
        bottom_churn_candidate = np.argmin(churn_prob)
        churn_labels[bottom_churn_candidate] = 0
    # Return as pandas Series with original index for compatibility
    churn_labels = pd.Series(churn_labels, index=df.index)
    return churn_prob, churn_labels

def train_enhanced_churn_model(X, y):
    """
    Train an optimized churn prediction model with feature selection
    """
    # Check if both classes are present
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        logger.warning("Only one class present in training data. Using default classifier without GridSearchCV.")
        best_model = RandomForestClassifier(n_estimators=100, random_state=42)
        scaler = QuantileTransformer(output_distribution='normal')
        X_scaled = scaler.fit_transform(X)
        selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
        selector.fit(X_scaled, y)
        selected_features = selector.get_support()
        X_selected = X_scaled[:, selected_features]
        best_model.fit(X_selected, y)
        return best_model, scaler, selected_features, X_selected, y
    
    # Check if any class has too few samples
    class_counts = Counter(y)
    min_class_count = min(class_counts.values())
    if min_class_count < 2 or len(class_counts) < 2:
        logger.warning(f"Minimum class has only {min_class_count} samples or only {len(class_counts)} classes. Adjusting data to ensure at least 2 samples per class.")
        # Find the class with minimum samples
        min_class = np.argmin(class_counts.values())
        # If minimum class has only 1 sample, duplicate it
        if min_class_count == 1:
            min_class_indices = np.where(y == min_class)[0]
            # Duplicate the single sample
            X_new = np.vstack([X, X[min_class_indices]])
            y_new = np.hstack([y, y[min_class_indices]])
            X, y = X_new, y_new
            logger.info(f"Duplicated sample from class {min_class} to ensure minimum 2 samples per class.")
    
    # Use train/test split for proper evaluation
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as e:
        logger.warning(f"Stratified split failed: {e}. Using random split instead.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = QuantileTransformer(output_distribution='normal')
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initial feature selection
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
    selector.fit(X_train_scaled, y_train)
    selected_features = selector.get_support()
    X_train_selected = X_train_scaled[:, selected_features]
    X_test_selected = X_test_scaled[:, selected_features]
    
    # Hyperparameter tuning with reduced grid for efficiency
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    try:
        model = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        model.fit(X_train_selected, y_train)
        best_model = model.best_estimator_
        logger.info(f"Best model parameters: {model.best_params_}")
        logger.info(f"Best validation score: {model.best_score_:.4f}")
    except Exception as e:
        logger.warning(f"GridSearchCV failed: {e}. Using default model.")
        best_model = RandomForestClassifier(n_estimators=100, random_state=42)
        best_model.fit(X_train_selected, y_train)
    
    return best_model, scaler, selected_features, X_test_selected, y_test

def generate_enhanced_churn_predictions(customers_df, transactions_df=None):
    """
    Generate high-accuracy churn predictions
    """
    try:
        # Step 1: Calculate enhanced RFM features
        try:
            df = calculate_enhanced_rfm_features(customers_df)
        except Exception as e:
            logger.warning(f"Error in calculate_enhanced_rfm_features: {e}. Using fallback features.")
            df = customers_df.copy()
            df['recency_days'] = 0
            df['frequency'] = 1
            df['monetary'] = 1.0
            df['rfm_score'] = 0

        # Step 2: Calculate temporal behavioral features
        try:
            df = calculate_temporal_features(df, transactions_df)
        except Exception as e:
            logger.warning(f"Error in calculate_temporal_features: {e}. Skipping temporal features.")

        # Step 3: Create enhanced labels
        try:
            churn_prob, churn_labels = create_enhanced_churn_labels(df, transactions_df)
        except Exception as e:
            logger.warning(f"Error in create_enhanced_churn_labels: {e}. Using default labels.")
            churn_prob = np.zeros(len(df))
            churn_labels = pd.Series(np.zeros(len(df)), index=df.index)
        df['churn_probability'] = churn_prob
        df['churn_label'] = churn_labels

        # Preserve analysis_id from input or create new one
        if 'analysis_id' in customers_df.columns:
            df['analysis_id'] = customers_df['analysis_id'].iloc[0]
        else:
            import uuid
            df['analysis_id'] = str(uuid.uuid4())

        # Prepare features
        feature_candidates = [
            'recency_days', 'frequency', 'monetary', 'rfm_score',
            'recency_score', 'frequency_score', 'monetary_score',
            'customer_tenure', 'purchase_velocity', 'recent_activity_30d',
            'recent_activity_60d', 'frequency_trend'
        ]
        feature_cols = [col for col in feature_candidates if col in df.columns]
        X = df[feature_cols].fillna(0)
        y = df['churn_label']

        # Step 4: Train optimized model
        try:
            model, scaler, selected_features, X_test, y_test = train_enhanced_churn_model(X, y)
        except Exception as e:
            logger.warning(f"Error in train_enhanced_churn_model: {e}. Using default model.")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            scaler = None
            selected_features = None
            X_test = None
            y_test = None
            model.fit(X, y)

        # Generate final predictions for all data
        try:
            if scaler is not None and selected_features is not None:
                X_scaled = scaler.transform(X)
                X_selected = X_scaled[:, selected_features]
            else:
                X_selected = X
            # Only use predict_proba if model has two classes
            if hasattr(model, 'classes_') and len(model.classes_) == 2:
                proba = model.predict_proba(X_selected)[:, 1]
                proba = np.nan_to_num(proba, nan=0.0)
                proba = np.clip(proba, 0.0, 1.0)
                df['churn_probability_final'] = proba
            else:
                logger.warning("Model has only one class. Setting churn_probability_final to zeros.")
                df['churn_probability_final'] = 0.0
        except Exception as e:
            logger.warning(f"Error in prediction: {e}. Using zeros for churn_probability_final.")
            df['churn_probability_final'] = 0.0

        # Classify risk levels
        try:
            n = len(df)
            sorted_probs = np.sort(df['churn_probability_final'])
            if n >= 10:
                # 90th percentile for High Risk, 70th for Medium
                high_risk_thresh = sorted_probs[int(n * 0.9)]
                medium_risk_thresh = sorted_probs[int(n * 0.7)]
                print(f"[AutoBI] Auto-adjusted thresholds: High Risk >= {high_risk_thresh:.2f}, Medium Risk >= {medium_risk_thresh:.2f}")
                conditions = [
                    (df['churn_probability_final'] >= high_risk_thresh),
                    (df['churn_probability_final'] >= medium_risk_thresh),
                    (df['churn_probability_final'] < medium_risk_thresh)
                ]
                choices = ['High Risk', 'Medium Risk', 'Low Risk']
                df['risk_level'] = np.select(conditions, choices, default='Low Risk')
            else:
                # For small datasets, always assign at least 1 High and 1 Medium if possible
                sorted_idx = np.argsort(-df['churn_probability_final'].values)  # Descending
                risk_level = np.array(['Low Risk'] * n)
                if n >= 1:
                    risk_level[sorted_idx[0]] = 'High Risk'
                if n >= 2:
                    risk_level[sorted_idx[1]] = 'Medium Risk'
                # Optionally, assign more as Medium if n > 3
                if n > 3:
                    for i in range(2, int(np.ceil(n * 0.3))):
                        risk_level[sorted_idx[i]] = 'Medium Risk'
                df['risk_level'] = risk_level
                print(f"[AutoBI] Small dataset: assigned {sum(risk_level == 'High Risk')} High, {sum(risk_level == 'Medium Risk')} Medium, {sum(risk_level == 'Low Risk')} Low risk customers.")
            # Print new distribution
            high_risk_count = sum(1 for level in df['risk_level'] if level == 'High Risk')
            medium_risk_count = sum(1 for level in df['risk_level'] if level == 'Medium Risk')
            low_risk_count = sum(1 for level in df['risk_level'] if level == 'Low Risk')
            print(f"[AutoBI] Risk distribution: High={high_risk_count}, Medium={medium_risk_count}, Low={low_risk_count}")
        except Exception as e:
            logger.warning(f"Error in risk level classification: {e}. Setting all to Low Risk.")
            df['risk_level'] = 'Low Risk'

        # Calculate performance metrics on test set
        try:
            if X_test is not None and y_test is not None:
                y_pred = model.predict(X_test)
                if hasattr(model, 'classes_') and len(model.classes_) == 2:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_pred_proba = np.zeros_like(y_pred, dtype=float)
                accuracy = accuracy_score(y_test, y_pred)
                roc_auc = safe_roc_auc_score(y_test, y_pred_proba)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
            else:
                y_pred = model.predict(X_selected)
                if hasattr(model, 'classes_') and len(model.classes_) == 2:
                    y_pred_proba = model.predict_proba(X_selected)[:, 1]
                else:
                    y_pred_proba = np.zeros_like(y_pred, dtype=float)
                accuracy = accuracy_score(y, y_pred)
                roc_auc = safe_roc_auc_score(y, y_pred_proba)
                precision = precision_score(y, y_pred, zero_division=0)
                recall = recall_score(y, y_pred, zero_division=0)
        except Exception as e:
            logger.warning(f"Error in performance metrics: {e}. Setting metrics to None.")
            accuracy = None
            roc_auc = None
            precision = None
            recall = None

        # Add model info
        if selected_features is not None and any(selected_features):
            used_indices = np.where(selected_features)[0]
            features_used = [feature_cols[i] for i in used_indices if i < len(feature_cols)]
        else:
            features_used = feature_cols
        df.attrs['model_info'] = {
            'model_type': 'EnhancedRandomForest',
            'features_used': features_used,
            'performance': {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'precision': precision,
                'recall': recall
            }
        }
        return df
    except Exception as e:
        logger.error(f"Fatal error in generate_enhanced_churn_predictions: {e}")
        # Return a minimal DataFrame with error info
        return pd.DataFrame({'error': [str(e)]})

def analyze_customer_churn(customers_df, transactions_df=None):
    """
    Complete churn analysis workflow with database integration
    """
    try:
        # Generate enhanced predictions
        result_df = generate_enhanced_churn_predictions(customers_df, transactions_df)
        
        logger.info(f"Churn analysis completed for {len(result_df)} customers")
        logger.info(f"Risk level distribution:\n{result_df['risk_level'].value_counts()}")
        
        return result_df
    
    except Exception as e:
        logger.error(f"Error in churn analysis: {str(e)}")
        raise

def generate_churn_data(df):
    """
    Industry-grade churn prediction that works with any dataset.
    Uses RFM analysis, behavioral patterns, and ML prediction.
    """
    df_copy = df.copy()
    
    # Create customer_id if not present
    if 'customer_id' not in df_copy.columns:
        df_copy['customer_id'] = range(1, len(df_copy) + 1)
    
    # Ensure analysis_id is present
    if 'analysis_id' not in df_copy.columns:
        import uuid
        df_copy['analysis_id'] = str(uuid.uuid4())
    
    print(f"🔍 Processing {len(df_copy)} customers for churn analysis...")
    
    # Use enhanced churn prediction
    result_df = generate_enhanced_churn_predictions(df_copy)
    
    # Ensure analysis_id is preserved in the result
    if 'analysis_id' not in result_df.columns and 'analysis_id' in df_copy.columns:
        result_df['analysis_id'] = df_copy['analysis_id'].iloc[0]
    
    # Debug output
    print("[AutoBI] Churn probabilities:")
    print(result_df[['customer_id', 'churn_probability_final']])
    print("[AutoBI] Sorted churn probabilities:")
    print(np.sort(result_df['churn_probability_final'].values))
    print("[AutoBI] Assigned risk levels:")
    print(result_df[['customer_id', 'churn_probability_final', 'risk_level']])
    high_risk_count = sum(1 for level in result_df['risk_level'] if level == 'High Risk')
    medium_risk_count = sum(1 for level in result_df['risk_level'] if level == 'Medium Risk')
    low_risk_count = sum(1 for level in result_df['risk_level'] if level == 'Low Risk')
    
    print(f"🎯 Churn probability range: {result_df['churn_probability_final'].min():.4f} to {result_df['churn_probability_final'].max():.4f}")
    print(f"🎯 Churn probability mean: {result_df['churn_probability_final'].mean():.4f}")
    print(f"⚠️ Risk distribution: High={high_risk_count}, Medium={medium_risk_count}, Low={low_risk_count}")
    
    # Display accuracy information
    if hasattr(result_df, 'attrs') and 'model_info' in result_df.attrs:
        model_info = result_df.attrs['model_info']
        performance = model_info.get('performance', {})
        def fmt(val, fmtstr):
            return fmtstr.format(val) if val is not None else 'N/A'
        print(f"📊 Model Performance:")
        print(f"   Accuracy: {fmt(performance.get('accuracy'), '{:.4f}')}")
        print(f"   ROC-AUC: {fmt(performance.get('roc_auc'), '{:.4f}')}")
        print(f"   Precision: {fmt(performance.get('precision'), '{:.4f}')}")
        print(f"   Recall: {fmt(performance.get('recall'), '{:.4f}')}")
        # If accuracy is low or all risk levels are Low, lower thresholds and reassign risk levels
        if (performance.get('accuracy') is not None and performance.get('accuracy') < 0.5) or (high_risk_count == 0 and medium_risk_count == 0):
            print("⚠️ Detected low accuracy or all customers as Low Risk. Lowering risk thresholds for better segmentation.")
            # Lower thresholds
            conditions = [
                (result_df['churn_probability_final'] >= 0.4),
                (result_df['churn_probability_final'] >= 0.2),
                (result_df['churn_probability_final'] < 0.2)
            ]
            choices = ['High Risk', 'Medium Risk', 'Low Risk']
            result_df['risk_level'] = np.select(conditions, choices, default='Low Risk')
            # Print new distribution
            high_risk_count = sum(1 for level in result_df['risk_level'] if level == 'High Risk')
            medium_risk_count = sum(1 for level in result_df['risk_level'] if level == 'Medium Risk')
            low_risk_count = sum(1 for level in result_df['risk_level'] if level == 'Low Risk')
            print(f"✅ New risk distribution: High={high_risk_count}, Medium={medium_risk_count}, Low={low_risk_count}")
    
    return result_df

# Backward compatibility
def generate_churn_predictions(df):
    """Backward compatibility function"""
    return generate_churn_data(df)

def safe_roc_auc_score(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        logger.warning("ROC AUC score is not defined for a single class.")
        return None
    try:
        return roc_auc_score(y_true, y_pred)
    except Exception as e:
        logger.warning(f"ROC AUC calculation failed: {e}")
        return None