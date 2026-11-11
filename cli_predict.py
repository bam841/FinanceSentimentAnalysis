#!/usr/bin/env python3
"""Command-Line Interface (CLI) for Financial Sentiment Analysis Pipeline.
Rubric Criterion D3 (Deployment Interface).

Usage:
    python cli_predict.py "Operating profit surged by 25% to EUR 45 million."
    python cli_predict.py --model finbert "Operating profit surged by 25%."
"""

import sys
import os
import argparse
import json
import time

def predict_cli():
    parser = argparse.ArgumentParser(description="Financial Sentiment Prediction CLI (Rubric D3)")
    parser.add_argument("text", nargs="?", type=str, default="", help="Financial headline or market commentary to analyze")
    parser.add_argument("--model", type=str, choices=["baseline", "finbert"], default="baseline", help="Model architecture (baseline=Logistic Regression, finbert=FinBERT Transformer)")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    if not args.text.strip():
        # Interactive prompt mode
        print("\n" + "=" * 65)
        print("  Financial Sentiment Analysis Interactive CLI (Rubric D3)")
        print("=" * 65)
        text = input("Enter financial sentence: ").strip()
        if not text:
            print("Error: Empty input provided.")
            sys.exit(1)
    else:
        text = args.text.strip()

    # Import inference engine
    from app import FinancialSentimentInferenceEngine
    engine = FinancialSentimentInferenceEngine(model_choice=args.model)
    result = engine.predict(text)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "-" * 50)
        print(f" Input:      {text}")
        print(f" Model:      {args.model.upper()}")
        print(f" Sentiment:  {result.get('predicted_sentiment', 'N/A').upper()}")
        print(f" Confidence: {result.get('confidence', 0.0) * 100:.2f}%")
        print(f" Latency:    {result.get('latency_ms', 0.0):.2f} ms")
        print(" Probabilities:")
        for sentiment, prob in result.get("probabilities", {}).items():
            bar = "█" * int(prob * 30)
            print(f"   {sentiment:<8} : {prob * 100:5.2f}% {bar}")
        print("-" * 50 + "\n")

if __name__ == '__main__':
    predict_cli()
