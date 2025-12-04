"""
Feature Engineering for Stablecoin Risk Detection

Computes 15+ features from transaction and market data:
- Transaction-level features (value ratio, whale activity)
- Supply metrics (mint/burn ratio, supply change rate)
- Concentration metrics (Gini coefficient, holder growth)
- Volatility metrics (realized volatility, volume z-score)
- Cross-asset features (correlation, exchange flow)
- Temporal features (hour, day of week)
- Macro features (treasury yields, spread)

Author: Aditya Sakhale
Institution: NYU School of Professional Studies
Date: November 2025
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler
import redis
import os


class FeatureEngineer:
    """
    Feature engineering pipeline for stablecoin transactions.
    Uses Redis for caching computed features and historical data.
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize feature engineer.
        
        Args:
            use_cache: Whether to use Redis caching
        """
        self.use_cache = use_cache
        self.scaler = StandardScaler()
        
        # Connect to Redis if caching enabled
        if use_cache:
            try:
                self.redis_client = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", 6379)),
                    db=int(os.getenv("REDIS_DB", 0)),
                    decode_responses=True
                )
                self.redis_client.ping()
                print("Redis connected for feature caching")
            except redis.ConnectionError:
                print("Redis not available, caching disabled")
                self.use_cache = False
        
        # Feature definitions with calculation methods
        self.feature_definitions = {
            'mint_burn_ratio': 'Ratio of minting to burning activity',
            'concentration_index': 'Gini coefficient of holder distribution',
            'realized_volatility': '30-day rolling standard deviation',
            'net_exchange_flow': 'Net flow to/from exchanges',
            'tx_value_ratio': 'Transaction value / average transaction value',
            'cross_asset_corr': 'Correlation with other stablecoins',
            'whale_activity': 'Binary indicator for large transactions',
            'volume_zscore': 'Z-score of 24h volume',
            'supply_change_rate': 'Daily supply change rate',
            'holder_growth_rate': 'Daily holder growth rate',
            'hour_sin': 'Sine of hour (cyclical encoding)',
            'hour_cos': 'Cosine of hour (cyclical encoding)',
            'day_of_week': 'Day of week (0-6)',
            'treasury_yield_2y': '2-year treasury yield',
            'treasury_spread': '10Y-2Y treasury spread'
        }
    
    def compute_features(
        self,
        stablecoin: str,
        sender: str,
        receiver: str,
        value: float,
        timestamp: datetime,
        historical_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Compute all features for a transaction.
        
        Args:
            stablecoin: Token symbol (USDT, USDC, DAI, BUSD)
            sender: Sender address
            receiver: Receiver address
            value: Transaction value
            timestamp: Transaction timestamp
            historical_data: Optional historical data for context
            
        Returns:
            Dictionary of feature name -> value
        """
        features = {}
        
        # Check cache first
        if self.use_cache:
            cache_key = f"features:{stablecoin}:{timestamp.date()}"
            cached = self._get_cached_features(cache_key)
            if cached:
                # Update with transaction-specific features
                cached.update(self._compute_transaction_features(value, timestamp))
                return cached
        
        # Compute supply metrics
        features.update(self._compute_supply_metrics(stablecoin, historical_data))
        
        # Compute concentration metrics
        features.update(self._compute_concentration_metrics(stablecoin, historical_data))
        
        # Compute volatility metrics
        features.update(self._compute_volatility_metrics(stablecoin, historical_data))
        
        # Compute cross-asset metrics
        features.update(self._compute_cross_asset_metrics(stablecoin, historical_data))
        
        # Compute transaction-specific features
        features.update(self._compute_transaction_features(value, timestamp))
        
        # Compute macro features
        features.update(self._compute_macro_features(historical_data))
        
        # Cache features
        if self.use_cache:
            self._cache_features(cache_key, features)
        
        return features
    
    def _compute_supply_metrics(
        self, 
        stablecoin: str, 
        historical_data: Optional[Dict]
    ) -> Dict[str, float]:
        """Compute supply-related features"""
        if historical_data and 'supply' in historical_data:
            supply = historical_data['supply']
            mints = historical_data.get('mints', 0)
            burns = historical_data.get('burns', 0)
            
            mint_burn_ratio = mints / max(burns, 1)
            supply_change = historical_data.get('supply_change', 0)
            supply_change_rate = supply_change / max(supply, 1)
        else:
            # Default values
            mint_burn_ratio = 1.0
            supply_change_rate = 0.0
        
        return {
            'mint_burn_ratio': mint_burn_ratio,
            'supply_change_rate': supply_change_rate
        }
    
    def _compute_concentration_metrics(
        self,
        stablecoin: str,
        historical_data: Optional[Dict]
    ) -> Dict[str, float]:
        """Compute holder concentration features"""
        if historical_data and 'holder_distribution' in historical_data:
            distribution = np.array(historical_data['holder_distribution'])
            concentration_index = self._gini_coefficient(distribution)
            
            holders_today = historical_data.get('holders', 1000)
            holders_yesterday = historical_data.get('holders_yesterday', 1000)
            holder_growth_rate = (holders_today - holders_yesterday) / max(holders_yesterday, 1)
        else:
            concentration_index = 0.5
            holder_growth_rate = 0.0
        
        return {
            'concentration_index': concentration_index,
            'holder_growth_rate': holder_growth_rate
        }
    
    def _gini_coefficient(self, distribution: np.ndarray) -> float:
        """Calculate Gini coefficient for holder distribution"""
        if len(distribution) == 0:
            return 0.5
        
        sorted_dist = np.sort(distribution)
        n = len(sorted_dist)
        cumulative = np.cumsum(sorted_dist)
        
        return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    
    def _compute_volatility_metrics(
        self,
        stablecoin: str,
        historical_data: Optional[Dict]
    ) -> Dict[str, float]:
        """Compute volatility and volume features"""
        if historical_data and 'prices' in historical_data:
            prices = np.array(historical_data['prices'])
            returns = np.diff(prices) / prices[:-1]
            realized_volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            volumes = historical_data.get('volumes', [])
            if len(volumes) > 1:
                volume_zscore = (volumes[-1] - np.mean(volumes)) / max(np.std(volumes), 0.001)
            else:
                volume_zscore = 0.0
        else:
            realized_volatility = 0.02  # Default 2% volatility
            volume_zscore = 0.0
        
        return {
            'realized_volatility': realized_volatility,
            'volume_zscore': volume_zscore
        }
    
    def _compute_cross_asset_metrics(
        self,
        stablecoin: str,
        historical_data: Optional[Dict]
    ) -> Dict[str, float]:
        """Compute cross-asset correlation and exchange flow"""
        if historical_data and 'cross_asset_prices' in historical_data:
            # Compute correlation with other stablecoins
            own_prices = np.array(historical_data['prices'])
            other_prices = np.array(historical_data['cross_asset_prices'])
            
            if len(own_prices) > 1 and len(other_prices) > 1:
                cross_asset_corr = np.corrcoef(own_prices, other_prices)[0, 1]
            else:
                cross_asset_corr = 0.4  # Default moderate correlation
            
            # Net exchange flow
            net_exchange_flow = historical_data.get('net_exchange_flow', 0)
        else:
            cross_asset_corr = 0.4
            net_exchange_flow = 0.0
        
        return {
            'cross_asset_corr': cross_asset_corr,
            'net_exchange_flow': net_exchange_flow
        }
    
    def _compute_transaction_features(
        self,
        value: float,
        timestamp: datetime
    ) -> Dict[str, float]:
        """Compute transaction-specific features"""
        # Value ratio (compared to typical transaction)
        typical_value = 10000  # $10k typical transaction
        tx_value_ratio = value / typical_value
        
        # Whale activity (>$1M transaction)
        whale_threshold = 1000000
        whale_activity = 1.0 if value > whale_threshold else 0.0
        
        # Temporal features (cyclical encoding)
        hour = timestamp.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_of_week = timestamp.weekday()
        
        return {
            'tx_value_ratio': tx_value_ratio,
            'whale_activity': whale_activity,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'day_of_week': float(day_of_week)
        }
    
    def _compute_macro_features(
        self,
        historical_data: Optional[Dict]
    ) -> Dict[str, float]:
        """Compute macroeconomic features from FRED data"""
        if historical_data and 'treasury_yields' in historical_data:
            yields = historical_data['treasury_yields']
            treasury_yield_2y = yields.get('DGS2', 4.5)
            treasury_yield_10y = yields.get('DGS10', 4.0)
            treasury_spread = treasury_yield_10y - treasury_yield_2y
        else:
            treasury_yield_2y = 4.5
            treasury_spread = -0.5  # Inverted yield curve default
        
        return {
            'treasury_yield_2y': treasury_yield_2y,
            'treasury_spread': treasury_spread
        }
    
    def _get_cached_features(self, cache_key: str) -> Optional[Dict]:
        """Get features from Redis cache"""
        if not self.use_cache:
            return None
        
        try:
            cached = self.redis_client.hgetall(cache_key)
            if cached:
                return {k: float(v) for k, v in cached.items()}
        except Exception:
            pass
        
        return None
    
    def _cache_features(self, cache_key: str, features: Dict[str, float]):
        """Cache features in Redis"""
        if not self.use_cache:
            return
        
        try:
            self.redis_client.hset(cache_key, mapping={k: str(v) for k, v in features.items()})
            self.redis_client.expire(cache_key, 3600)  # 1 hour TTL
        except Exception:
            pass
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return list(self.feature_definitions.keys())
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get feature descriptions"""
        return self.feature_definitions.copy()


# Example usage
if __name__ == "__main__":
    # Initialize feature engineer
    fe = FeatureEngineer(use_cache=False)
    
    # Compute sample features
    features = fe.compute_features(
        stablecoin="USDT",
        sender="0x123...",
        receiver="0x456...",
        value=500000,
        timestamp=datetime.now()
    )
    
    print("Computed features:")
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")
