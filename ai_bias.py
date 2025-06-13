import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from openai import OpenAI
from datetime import datetime

@dataclass
class MacroBias:
    sentiment: str  # 'bullish', 'bearish', or 'neutral'
    impact_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    confidence_adjustment: float  # -1.0 to 1.0
    explanation: str
    headlines_analyzed: List[Dict[str, str]]

class MacroBiasAnalyzer:
    def __init__(self, api_key: str):
        self.openai_client = OpenAI(api_key=api_key)
        
    def _read_headlines(self, file_path: str = 'headlines.txt') -> List[str]:
        """Read headlines from file."""
        try:
            with open(file_path, 'r') as f:
                return [line.strip() for line in f.readlines() if line.strip()][:5]  # Read up to 5 headlines
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
            return []
            
    def _analyze_headline(self, headline: str) -> Dict:
        """Analyze a single headline using OpenAI."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial market analyst. Analyze the given headline and determine its impact on financial markets."},
                    {"role": "user", "content": f"Analyze this market headline and provide a structured response:\n{headline}\n\nProvide your analysis in JSON format with these fields:\n- sentiment: 'bullish', 'bearish', or 'neutral'\n- impact_score: number between -1.0 and 1.0\n- confidence: number between 0.0 and 1.0\n- explanation: brief explanation of your analysis"}
                ],
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error analyzing headline: {str(e)}")
            return {
                "sentiment": "neutral",
                "impact_score": 0.0,
                "confidence": 0.0,
                "explanation": f"Error in analysis: {str(e)}"
            }
            
    def _calculate_asset_bias(self, analyses: List[Dict]) -> Dict:
        """Calculate aggregate market bias from multiple analyses."""
        if not analyses:
            return {
                "sentiment": "neutral",
                "impact_score": 0.0,
                "confidence": 0.0,
                "confidence_adjustment": 0.0,
                "explanation": "No headlines analyzed"
            }
            
        # Count sentiments
        sentiment_counts = {
            "bullish": 0,
            "bearish": 0,
            "neutral": 0
        }
        
        total_impact = 0.0
        total_confidence = 0.0
        
        for analysis in analyses:
            sentiment = analysis["sentiment"].lower()
            sentiment_counts[sentiment] += 1
            
            # Ensure numeric values
            try:
                impact_score = float(analysis["impact_score"])
                confidence = float(analysis["confidence"])
                total_impact += impact_score
                total_confidence += confidence
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid numeric value in analysis: {e}")
                continue
            
        # Calculate dominant sentiment and confidence adjustment
        total_headlines = len(analyses)
        dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate confidence adjustment based on sentiment proportion
        if dominant_sentiment == "bullish":
            confidence_adjustment = sentiment_counts["bullish"] / total_headlines
        elif dominant_sentiment == "bearish":
            confidence_adjustment = -sentiment_counts["bearish"] / total_headlines
        else:
            confidence_adjustment = 0.0
            
        # Calculate average impact and confidence
        avg_impact = total_impact / total_headlines if total_headlines > 0 else 0.0
        avg_confidence = total_confidence / total_headlines if total_headlines > 0 else 0.0
        
        # Generate explanation
        explanation = f"Analysis of {total_headlines} headlines:\n"
        explanation += f"- Bullish: {sentiment_counts['bullish']}\n"
        explanation += f"- Bearish: {sentiment_counts['bearish']}\n"
        explanation += f"- Neutral: {sentiment_counts['neutral']}\n"
        explanation += f"- Dominant sentiment: {dominant_sentiment}\n"
        explanation += f"- Confidence adjustment: {confidence_adjustment:.2f}"
        
        return {
            "sentiment": dominant_sentiment,
            "impact_score": avg_impact,
            "confidence": avg_confidence,
            "confidence_adjustment": confidence_adjustment,
            "explanation": explanation,
            "headlines_analyzed": analyses
        }
        
    def analyze_macro_bias(self) -> Dict[str, MacroBias]:
        """Analyze macro market bias from headlines."""
        headlines = self._read_headlines()
        if not headlines:
            return {}
            
        # Analyze each headline
        analyses = [self._analyze_headline(headline) for headline in headlines]
        
        # Calculate bias for each asset type
        biases = {}
        for asset_type in ['Crypto', 'Stocks', 'Index']:
            bias_data = self._calculate_asset_bias(analyses)
            biases[asset_type] = MacroBias(
                sentiment=bias_data["sentiment"],
                impact_score=bias_data["impact_score"],
                confidence=bias_data["confidence"],
                confidence_adjustment=bias_data["confidence_adjustment"],
                explanation=bias_data["explanation"],
                headlines_analyzed=bias_data["headlines_analyzed"]
            )
            
        # Save to file
        with open('macro_bias.json', 'w') as f:
            json.dump({
                asset_type: {
                    "sentiment": bias.sentiment,
                    "impact_score": bias.impact_score,
                    "confidence": bias.confidence,
                    "confidence_adjustment": bias.confidence_adjustment,
                    "explanation": bias.explanation,
                    "headlines_analyzed": bias.headlines_analyzed
                }
                for asset_type, bias in biases.items()
            }, f, indent=2)
            
        return biases

if __name__ == "__main__":
    # Test the macro bias analyzer
    analyzer = MacroBiasAnalyzer()
    
    # Sample headlines
    headlines = [
        "Fed Raises Interest Rates by 25 Basis Points",
        "US GDP Growth Exceeds Expectations",
        "Tech Stocks Rally on Strong Earnings"
    ]
    
    # Analyze individual headlines
    print("\nAnalyzing individual headlines:")
    for headline in headlines:
        bias = analyzer.analyze_headline(headline)
        print(f"\nHeadline: {headline}")
        print(f"Sentiment: {bias.sentiment}")
        print(f"Confidence: {bias.confidence:.0%}")
        print(f"Impact: {bias.impact_score:.2f}")
        print(f"Explanation: {bias.explanation}")
    
    # Get aggregate bias
    biases = analyzer.analyze_macro_bias()
    
    print("\nAggregate Market Bias:")
    for asset_type, bias in biases.items():
        print(f"\nAsset Type: {asset_type}")
        print(f"Sentiment: {bias.sentiment}")
        print(f"Confidence: {bias.confidence:.0%}")
        print(f"Impact: {bias.impact_score:.2f}")
        print(f"Explanation: {bias.explanation}")
        print("\nHeadlines Analyzed:")
        for h in bias.headlines_analyzed:
            print(f"Headline: {h['headline']}")
            print(f"Sentiment: {h['sentiment']}")
            print(f"Impact: {h['impact']}") 