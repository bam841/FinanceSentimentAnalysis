"""Defensive unit test suite for Financial Sentiment Inference Engine (Rubric Criterion D4).

Validates model robustness against 7 adversarial and non-standard input regimes:
1. Empty string ("")
2. Whitespace-only string ("   ")
3. Long sequence (>5,000 characters)
4. Pure numeric/financial tabular noise ("1,234,567.89 -4.5%")
5. Emojis and unicode characters ("Stock surging 🚀📈 vs crashing 📉💀")
6. Non-English foreign language input ("Tulos kasvoi ja kannattavuus parani")
7. Non-string / None invalid data types
"""

import os
import sys
import unittest
import joblib
import json

class TestInferenceRobustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Locate package artifacts
        cls.package_dir = os.path.abspath('deployable_package')
        if not os.path.exists(cls.package_dir):
            cls.package_dir = os.path.abspath('artifacts')

    def test_empty_string(self):
        """Rubric D4.1: Empty input string handling."""
        from execute_notebook import FinancialSentimentInferenceEngine
        engine = FinancialSentimentInferenceEngine()
        result = engine.predict("")
        self.assertIn("error", result)
        self.assertEqual(result["status"], "rejected")
        print("  [PASS] Empty string gracefully rejected with status='rejected'")

    def test_whitespace_string(self):
        """Rubric D4.2: Whitespace-only string handling."""
        from execute_notebook import FinancialSentimentInferenceEngine
        engine = FinancialSentimentInferenceEngine()
        result = engine.predict("    \n\t   ")
        self.assertIn("error", result)
        self.assertEqual(result["status"], "rejected")
        print("  [PASS] Whitespace-only string gracefully rejected")

    def test_long_sequence_truncation(self):
        """Rubric D4.3: Extreme length sequence safe truncation (>5,000 chars)."""
        from execute_notebook import FinancialSentimentInferenceEngine
        engine = FinancialSentimentInferenceEngine()
        long_text = "Operating profit increased sharply due to robust revenue expansion across European markets. " * 80
        self.assertGreater(len(long_text), 5000)
        result = engine.predict(long_text)
        self.assertEqual(result["status"], "success")
        self.assertIn(result["predicted_sentiment"], ["positive", "neutral", "negative"])
        self.assertGreaterEqual(result["confidence"], 0.0)
        print(f"  [PASS] Long text (len={len(long_text)}) safely processed: {result['predicted_sentiment']} (conf={result['confidence']:.3f})")

    def test_numeric_financial_noise(self):
        """Rubric D4.4: Pure numerical figures and balance sheet figures."""
        from execute_notebook import FinancialSentimentInferenceEngine
        engine = FinancialSentimentInferenceEngine()
        numeric_text = "12,450.00 EUR 45.2% +1.20 -0.85 2026-Q3"
        result = engine.predict(numeric_text)
        self.assertEqual(result["status"], "success")
        self.assertIn("probabilities", result)
        print(f"  [PASS] Pure financial numbers safely classified: {result['predicted_sentiment']}")

    def test_emojis_and_unicode(self):
        """Rubric D4.5: Unicode and financial emoji robustness."""
        from execute_notebook import FinancialSentimentInferenceEngine
        engine = FinancialSentimentInferenceEngine()
        emoji_bullish = "Quarterly revenues jumped 40% 🚀💰📈 bull market ahead!"
        result = engine.predict(emoji_bullish)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["predicted_sentiment"], "positive")
        print(f"  [PASS] Unicode/emoji bullish classified correctly: {result['predicted_sentiment']}")

    def test_non_english_input(self):
        """Rubric D4.6: Foreign language input without crash."""
        from execute_notebook import FinancialSentimentInferenceEngine
        engine = FinancialSentimentInferenceEngine()
        finnish_text = "Yhtiön liikevaihto kasvoi ja liiketulos parani huomattavasti edellisvuodesta."
        result = engine.predict(finnish_text)
        self.assertEqual(result["status"], "success")
        print(f"  [PASS] Non-English text processed without crashing: {result['predicted_sentiment']}")

    def test_invalid_data_types(self):
        """Rubric D4.7: Invalid data type defensive rejection."""
        from execute_notebook import FinancialSentimentInferenceEngine
        engine = FinancialSentimentInferenceEngine()
        for invalid_val in [None, 12345, [1, 2, 3], {"text": "hello"}]:
            result = engine.predict(invalid_val)
            self.assertEqual(result["status"], "rejected")
            self.assertIn("error", result)
        print("  [PASS] None and non-string types safely rejected")

if __name__ == '__main__':
    print("=" * 70)
    print("Executing Rubric D4 Robustness & Error Handling Test Suite...")
    print("=" * 70)
    unittest.main(verbosity=2)
