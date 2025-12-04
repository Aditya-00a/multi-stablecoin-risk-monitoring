"""
Etherscan API Client for Stablecoin Transaction Data

Fetches blockchain transaction data for:
- USDT (Tether)
- USDC (USD Coin)
- DAI (MakerDAO)
- BUSD (Binance USD)

Author: Aditya Sakhale
Institution: NYU School of Professional Studies
Date: November 2025
"""

import os
import requests
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import redis


class EtherscanClient:
    """
    Client for Etherscan API V2 to fetch stablecoin transaction data.
    """
    
    # Stablecoin contract addresses (Ethereum Mainnet)
    CONTRACTS = {
        "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "DAI": "0x6B175474E89094C44Da98b954EescdeCB5BE1FBa",
        "BUSD": "0x4Fabb145d64652a948d72533023f6E7A623C7C53"
    }
    
    # Token decimals
    DECIMALS = {
        "USDT": 6,
        "USDC": 6,
        "DAI": 18,
        "BUSD": 18
    }
    
    BASE_URL = "https://api.etherscan.io/api"
    
    def __init__(self, api_key: Optional[str] = None, use_cache: bool = True):
        """
        Initialize Etherscan client.
        
        Args:
            api_key: Etherscan API key (or from env)
            use_cache: Whether to use Redis caching
        """
        self.api_key = api_key or os.getenv("ETHERSCAN_API_KEY")
        if not self.api_key:
            raise ValueError("Etherscan API key required")
        
        self.use_cache = use_cache
        self.rate_limit_delay = 0.2  # 5 requests per second
        self.last_request_time = 0
        
        # Initialize Redis cache
        if use_cache:
            try:
                self.redis_client = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", 6379)),
                    decode_responses=True
                )
                self.redis_client.ping()
            except redis.ConnectionError:
                print("Redis not available, caching disabled")
                self.use_cache = False
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, params: Dict) -> Dict:
        """Make API request with rate limiting"""
        self._rate_limit()
        
        params["apikey"] = self.api_key
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "0" and "rate limit" in data.get("message", "").lower():
                time.sleep(1)
                return self._make_request(params)
            
            return data
            
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            return {"status": "0", "message": str(e), "result": []}
    
    def get_token_transfers(
        self,
        stablecoin: str,
        address: Optional[str] = None,
        start_block: int = 0,
        end_block: int = 99999999,
        page: int = 1,
        offset: int = 100,
        sort: str = "desc"
    ) -> List[Dict]:
        """
        Get token transfer events for a stablecoin.
        
        Args:
            stablecoin: Token symbol (USDT, USDC, DAI, BUSD)
            address: Filter by address (optional)
            start_block: Starting block number
            end_block: Ending block number
            page: Page number for pagination
            offset: Number of results per page
            sort: Sort order (asc/desc)
            
        Returns:
            List of transfer events
        """
        if stablecoin not in self.CONTRACTS:
            raise ValueError(f"Unsupported stablecoin: {stablecoin}")
        
        contract = self.CONTRACTS[stablecoin]
        
        params = {
            "module": "account",
            "action": "tokentx",
            "contractaddress": contract,
            "startblock": start_block,
            "endblock": end_block,
            "page": page,
            "offset": offset,
            "sort": sort
        }
        
        if address:
            params["address"] = address
        
        result = self._make_request(params)
        
        if result.get("status") == "1":
            transfers = result.get("result", [])
            return self._parse_transfers(transfers, stablecoin)
        
        return []
    
    def _parse_transfers(self, transfers: List[Dict], stablecoin: str) -> List[Dict]:
        """Parse and normalize transfer data"""
        decimals = self.DECIMALS[stablecoin]
        parsed = []
        
        for tx in transfers:
            try:
                value = int(tx.get("value", 0)) / (10 ** decimals)
                
                parsed.append({
                    "hash": tx.get("hash"),
                    "block_number": int(tx.get("blockNumber", 0)),
                    "timestamp": datetime.fromtimestamp(int(tx.get("timeStamp", 0))),
                    "from": tx.get("from", "").lower(),
                    "to": tx.get("to", "").lower(),
                    "value": value,
                    "stablecoin": stablecoin,
                    "gas_price": int(tx.get("gasPrice", 0)),
                    "gas_used": int(tx.get("gasUsed", 0))
                })
            except (ValueError, TypeError) as e:
                continue
        
        return parsed
    
    def get_token_supply(self, stablecoin: str) -> float:
        """
        Get current total supply of a stablecoin.
        
        Args:
            stablecoin: Token symbol
            
        Returns:
            Total supply as float
        """
        if stablecoin not in self.CONTRACTS:
            raise ValueError(f"Unsupported stablecoin: {stablecoin}")
        
        # Check cache first
        if self.use_cache:
            cache_key = f"supply:{stablecoin}"
            cached = self.redis_client.get(cache_key)
            if cached:
                return float(cached)
        
        contract = self.CONTRACTS[stablecoin]
        decimals = self.DECIMALS[stablecoin]
        
        params = {
            "module": "stats",
            "action": "tokensupply",
            "contractaddress": contract
        }
        
        result = self._make_request(params)
        
        if result.get("status") == "1":
            supply = int(result.get("result", 0)) / (10 ** decimals)
            
            # Cache for 5 minutes
            if self.use_cache:
                self.redis_client.setex(f"supply:{stablecoin}", 300, str(supply))
            
            return supply
        
        return 0.0
    
    def get_token_holders(self, stablecoin: str) -> int:
        """
        Get approximate number of token holders.
        Note: Etherscan doesn't provide exact holder count via API.
        This is estimated from recent transfer unique addresses.
        
        Args:
            stablecoin: Token symbol
            
        Returns:
            Estimated holder count
        """
        # Get recent transfers
        transfers = self.get_token_transfers(stablecoin, offset=1000)
        
        # Count unique addresses
        addresses = set()
        for tx in transfers:
            addresses.add(tx["from"])
            addresses.add(tx["to"])
        
        return len(addresses)
    
    def get_whale_transactions(
        self,
        stablecoin: str,
        min_value: float = 1000000,  # $1M minimum
        hours: int = 24
    ) -> List[Dict]:
        """
        Get large transactions (whale activity).
        
        Args:
            stablecoin: Token symbol
            min_value: Minimum transaction value
            hours: Time window in hours
            
        Returns:
            List of whale transactions
        """
        transfers = self.get_token_transfers(stablecoin, offset=500)
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        whale_txs = [
            tx for tx in transfers
            if tx["value"] >= min_value and tx["timestamp"] >= cutoff_time
        ]
        
        return whale_txs
    
    def get_address_balance(self, stablecoin: str, address: str) -> float:
        """
        Get token balance for an address.
        
        Args:
            stablecoin: Token symbol
            address: Ethereum address
            
        Returns:
            Token balance
        """
        if stablecoin not in self.CONTRACTS:
            raise ValueError(f"Unsupported stablecoin: {stablecoin}")
        
        contract = self.CONTRACTS[stablecoin]
        decimals = self.DECIMALS[stablecoin]
        
        params = {
            "module": "account",
            "action": "tokenbalance",
            "contractaddress": contract,
            "address": address,
            "tag": "latest"
        }
        
        result = self._make_request(params)
        
        if result.get("status") == "1":
            return int(result.get("result", 0)) / (10 ** decimals)
        
        return 0.0


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = EtherscanClient(use_cache=False)
    
    print("Fetching USDT data...")
    
    # Get recent transfers
    transfers = client.get_token_transfers("USDT", offset=10)
    print(f"\nRecent USDT transfers: {len(transfers)}")
    for tx in transfers[:3]:
        print(f"  {tx['hash'][:16]}... ${tx['value']:,.2f}")
    
    # Get total supply
    supply = client.get_token_supply("USDT")
    print(f"\nUSDT Total Supply: ${supply:,.0f}")
    
    # Get whale transactions
    whales = client.get_whale_transactions("USDT", min_value=5000000)
    print(f"\nWhale transactions (>$5M): {len(whales)}")
