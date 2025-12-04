"""
LLM Explainability Module for Stablecoin Risk Scores

Uses Llama 3.1 70B via Groq API to generate natural language
explanations for risk scores, with regulatory mapping to
SR 11-7 and BCBS 248 frameworks.

Author: Aditya Sakhale
Institution: NYU School of Professional Studies
Date: November 2025
"""

import os
from typing import Dict, List, Optional
from groq import Groq
from datetime import datetime


class LLMExplainer:
    """
    Generate natural language explanations for risk scores
    using Llama 3.1 70B via Groq API.
    """
    
    def __init__(self):
        """Initialize Groq client"""
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", 0.3))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", 500))
        
        # Regulatory framework references
        self.regulatory_frameworks = {
            "SR 11-7": {
                "name": "Federal Reserve SR 11-7",
                "description": "Guidance on Model Risk Management",
                "key_requirements": [
                    "Model validation and testing",
                    "Documentation of model limitations",
                    "Ongoing monitoring and outcomes analysis"
                ]
            },
            "BCBS 248": {
                "name": "Basel Committee BCBS 248",
                "description": "Principles for Effective Risk Data Aggregation",
                "key_requirements": [
                    "Accuracy and integrity of risk data",
                    "Completeness of risk capture",
                    "Timeliness of risk reporting"
                ]
            },
            "NIST AI RMF": {
                "name": "NIST AI Risk Management Framework",
                "description": "Framework for trustworthy AI systems",
                "key_requirements": [
                    "Explainability and interpretability",
                    "Bias detection and mitigation",
                    "Human oversight and control"
                ]
            }
        }
        
        # Risk factor templates
        self.risk_factor_templates = {
            'mint_burn_ratio': {
                'high': 'Elevated mint-to-burn ratio ({value:.2f}) indicates potential supply pressure',
                'normal': 'Mint-to-burn ratio ({value:.2f}) within normal range'
            },
            'concentration_index': {
                'high': 'High holder concentration (Gini: {value:.2f}) suggests whale dominance risk',
                'normal': 'Holder distribution (Gini: {value:.2f}) indicates healthy decentralization'
            },
            'realized_volatility': {
                'high': 'Elevated volatility ({value:.2%}) may precede depeg event',
                'normal': 'Price volatility ({value:.2%}) within acceptable bounds'
            },
            'net_exchange_flow': {
                'high': 'Large exchange outflow (${value:,.0f}) suggests potential selling pressure',
                'normal': 'Exchange flow (${value:,.0f}) indicates balanced market activity'
            },
            'whale_activity': {
                'high': 'Large transaction detected - potential market manipulation risk',
                'normal': 'Transaction size within typical range'
            }
        }
    
    def generate_explanation(
        self,
        stablecoin: str,
        risk_score: float,
        features: Dict[str, float],
        include_regulatory: bool = True
    ) -> Dict:
        """
        Generate natural language explanation for risk score.
        
        Args:
            stablecoin: Token symbol (USDT, USDC, DAI, BUSD)
            risk_score: Ensemble risk score (0-1)
            features: Dictionary of feature values
            include_regulatory: Include regulatory framework references
            
        Returns:
            Dictionary with explanation, risk factors, and regulatory references
        """
        # Determine risk level
        if risk_score < 0.33:
            risk_level = "LOW"
        elif risk_score < 0.67:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Identify key risk factors
        risk_factors = self._identify_risk_factors(features)
        
        # Build prompt
        prompt = self._build_prompt(
            stablecoin=stablecoin,
            risk_score=risk_score,
            risk_level=risk_level,
            features=features,
            risk_factors=risk_factors,
            include_regulatory=include_regulatory
        )
        
        # Generate explanation via Groq API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=0.9
            )
            
            explanation_text = response.choices[0].message.content
            
        except Exception as e:
            explanation_text = f"Unable to generate explanation: {str(e)}"
        
        # Build response
        result = {
            "text": explanation_text,
            "risk_factors": risk_factors,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "stablecoin": stablecoin,
            "timestamp": datetime.now().isoformat()
        }
        
        if include_regulatory:
            result["regulatory_references"] = self._get_regulatory_references(risk_factors)
        
        return result
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM"""
        return """You are a financial risk analyst AI assistant specializing in 
stablecoin liquidity risk assessment. Your role is to provide clear, concise, 
and actionable explanations of risk scores for compliance analysts and risk managers.

Guidelines:
1. Use professional financial terminology
2. Be specific about risk factors and their implications
3. Reference regulatory frameworks (SR 11-7, BCBS 248) when applicable
4. Provide actionable recommendations
5. Maintain objectivity and avoid speculation
6. Keep explanations concise (2-3 paragraphs)"""
    
    def _build_prompt(
        self,
        stablecoin: str,
        risk_score: float,
        risk_level: str,
        features: Dict[str, float],
        risk_factors: List[str],
        include_regulatory: bool
    ) -> str:
        """Build prompt for explanation generation"""
        # Format feature summary
        feature_summary = "\n".join([
            f"- {k}: {v:.4f}" for k, v in features.items()
        ])
        
        # Format risk factors
        risk_factors_text = "\n".join([f"- {rf}" for rf in risk_factors])
        
        prompt = f"""Generate a risk assessment explanation for the following transaction:

Stablecoin: {stablecoin}
Risk Score: {risk_score:.3f} ({risk_level} RISK)

Key Features:
{feature_summary}

Identified Risk Factors:
{risk_factors_text}

Please provide:
1. A brief summary of the overall risk assessment
2. Explanation of the primary risk drivers
3. {"Regulatory compliance considerations (SR 11-7, BCBS 248)" if include_regulatory else ""}
4. Recommended monitoring actions

Keep the response professional and suitable for a compliance report."""
        
        return prompt
    
    def _identify_risk_factors(self, features: Dict[str, float]) -> List[str]:
        """Identify key risk factors from feature values"""
        risk_factors = []
        
        # Define thresholds for risk identification
        thresholds = {
            'mint_burn_ratio': (0.8, 1.2),  # Normal range
            'concentration_index': (0.0, 0.6),  # Gini threshold
            'realized_volatility': (0.0, 0.05),  # 5% volatility threshold
            'net_exchange_flow': (-100000, 100000),  # $100k threshold
            'whale_activity': (0, 0.5),  # Binary threshold
            'volume_zscore': (-2, 2),  # 2 standard deviations
            'cross_asset_corr': (0.3, 0.7)  # Normal correlation range
        }
        
        for feature, value in features.items():
            if feature in thresholds:
                low, high = thresholds[feature]
                
                if value < low or value > high:
                    # Use template for risk factor
                    if feature in self.risk_factor_templates:
                        template = self.risk_factor_templates[feature]['high']
                        risk_factors.append(template.format(value=value))
        
        # Add default if no specific factors
        if not risk_factors:
            risk_factors.append("No specific risk factors identified above normal thresholds")
        
        return risk_factors
    
    def _get_regulatory_references(self, risk_factors: List[str]) -> List[str]:
        """Get relevant regulatory references based on risk factors"""
        references = []
        
        # SR 11-7 references
        if any('volatility' in rf.lower() or 'score' in rf.lower() for rf in risk_factors):
            references.append(
                "SR 11-7: Model outputs should be validated and monitored for consistency"
            )
        
        # BCBS 248 references
        if any('concentration' in rf.lower() or 'whale' in rf.lower() for rf in risk_factors):
            references.append(
                "BCBS 248: Risk concentrations should be identified and reported timely"
            )
        
        # NIST AI RMF reference
        references.append(
            "NIST AI RMF: Model explanations provided for transparency and auditability"
        )
        
        return references
    
    def get_available_models(self) -> List[str]:
        """Get list of available Groq models"""
        return [
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ]


# Example usage
if __name__ == "__main__":
    # Initialize explainer
    explainer = LLMExplainer()
    
    # Sample features
    sample_features = {
        'mint_burn_ratio': 1.5,
        'concentration_index': 0.72,
        'realized_volatility': 0.08,
        'net_exchange_flow': -250000,
        'tx_value_ratio': 3.2,
        'cross_asset_corr': 0.85,
        'whale_activity': 1.0,
        'volume_zscore': 2.3
    }
    
    # Generate explanation
    print("Generating explanation...")
    result = explainer.generate_explanation(
        stablecoin="USDT",
        risk_score=0.72,
        features=sample_features,
        include_regulatory=True
    )
    
    print("\n--- Explanation ---")
    print(result["text"])
    print("\n--- Risk Factors ---")
    for rf in result["risk_factors"]:
        print(f"  • {rf}")
    print("\n--- Regulatory References ---")
    for ref in result.get("regulatory_references", []):
        print(f"  • {ref}")
