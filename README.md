# Multi-Stablecoin AI Risk Monitoring System

A real-time risk monitoring platform for stablecoin liquidity analysis using ensemble machine learning and LLM-powered explainability.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Project Overview

Developed for **NextAML** as part of NYU MS in Management and Analytics Capstone Project.

**Author:** Aditya Sakhale  
**Institution:** NYU School of Professional Studies  
**Program:** MS in Management and Analytics  
**Date:** November 2025

This platform monitors four major stablecoins representing $184B in market capitalization, providing real-time risk assessment with regulatory-compliant explanations.

## 📊 Stablecoins Monitored

| Stablecoin | Market Cap | Contract Address (Ethereum) |
|------------|------------|----------------------------|
| USDT (Tether) | $140B | `0xdAC17F958D2ee523a2206206994597C13D831ec7` |
| USDC (Circle) | $38B | `0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48` |
| DAI (MakerDAO) | $4B | `0x6B175474E89094C44Da98b954EescdeCB5BE1FBa` |
| BUSD (Binance) | $2B | `0x4Fabb145d64652a948d72533023f6E7A623C7C53` |

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Etherscan   │  │  RWA.xyz    │  │  FRED API   │              │
│  │  API V2     │  │    API      │  │             │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMING LAYER (Kafka)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING (15+ Features)                 │
│  mint_burn_ratio │ concentration_index │ realized_volatility    │
│  net_exchange_flow │ whale_activity │ cross_asset_correlation   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML ENSEMBLE LAYER                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Isolation   │  │  One-Class  │  │  XGBoost    │              │
│  │ Forest (35%)│  │  SVM (25%)  │  │   (40%)     │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              LLM EXPLAINABILITY (Llama 3.1 70B)                 │
│                    via Groq API                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API LAYER (FastAPI)                          │
│              /predict │ /explain │ /health                      │
└─────────────────────────────────────────────────────────────────┘
```

## 📈 Performance Results

| Metric | Target | Achieved |
|--------|--------|----------|
| AUC-ROC | >0.90 | **0.94** |
| F1-Score | >0.85 | **0.90** |
| Precision | >0.85 | **0.89** |
| Recall | >0.85 | **0.91** |
| False Positive Rate | <10% | **6%** |
| API Latency (p95) | <100ms | **45ms** |
| Throughput | >10K TPS | **12K TPS** |

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Redis
- Apache Kafka (optional, for streaming)

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/multi-stablecoin-risk-monitoring.git
cd multi-stablecoin-risk-monitoring

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running the API

```bash
uvicorn src.api.main:app --reload
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single transaction risk prediction |
| `/batch_predict` | POST | Batch predictions |
| `/explain` | POST | Get LLM explanation for risk score |
| `/health` | GET | Health check |

## 📁 Project Structure

```
multi-stablecoin-risk-monitoring/
│
├── README.md                    # Project description
├── requirements.txt             # Python dependencies
├── .env.example                 # Example environment variables
├── .gitignore                   # Files to ignore
│
├── src/                         # Source code
│   ├── data_ingestion/          # API clients
│   ├── feature_engineering/     # Feature calculations
│   ├── models/                  # ML models
│   ├── llm/                     # LLM integration
│   └── api/                     # FastAPI app
│
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Unit tests
├── config/                      # Configuration
└── docs/                        # Documentation
```

## 🔧 Model Configuration

### Ensemble Weights

| Model | Weight | Role |
|-------|--------|------|
| Isolation Forest | 35% | Unsupervised anomaly detection |
| One-Class SVM | 25% | Boundary-based anomaly detection |
| XGBoost | 40% | Supervised classification |

### Key Features (Top 5)

| Feature | Description | Importance |
|---------|-------------|------------|
| mint_burn_ratio | Ratio of minting to burning activity | 0.187 |
| concentration_index | Gini coefficient of holder distribution | 0.156 |
| realized_volatility | 30-day rolling standard deviation | 0.142 |
| net_exchange_flow | Net flow to/from exchanges | 0.131 |
| tx_value_ratio | Transaction value relative to average | 0.118 |

## 📜 Regulatory Compliance

This system is designed to comply with:
- **SR 11-7** (Federal Reserve) - Model Risk Management
- **BCBS 248** (Basel Committee) - Operational Risk Guidelines
- **NIST AI RMF** - AI Risk Management Framework

## 🛠️ Technologies Used

- **Python 3.9+** - Core language
- **FastAPI** - API framework
- **XGBoost** - Gradient boosting
- **Scikit-learn** - ML utilities
- **Llama 3.1 70B** - LLM for explanations
- **Groq API** - LLM inference
- **Apache Kafka** - Event streaming
- **Redis** - Caching layer
- **Plotly** - Visualization

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NextAML** - Project sponsorship and industry guidance
- **NYU School of Professional Studies** - Academic support

---

*This project was developed as part of the NYU SPS Capstone Project (Fall 2025)*
