"""
Unit tests for Calculator and Web Search tools.

These tests verify tool functionality in isolation (no LLM calls needed).
Run with: python -m pytest tests/test_tools.py -v
"""
import json
import pytest

from src.tools.calculator import CalculatorTools
from src.tools.web_search import WebSearchTools


class TestCalculatorTools:
    """Tests for the CalculatorTools toolkit."""

    def setup_method(self):
        """Create a fresh CalculatorTools instance for each test."""
        self.calc = CalculatorTools()

    # --- calculate() tests ---

    def test_basic_arithmetic(self):
        result = json.loads(self.calc.calculate("45 * 32 + 17"))
        assert result["status"] == "success"
        assert result["result"] == 1457

    def test_division(self):
        result = json.loads(self.calc.calculate("100 / 4"))
        assert result["status"] == "success"
        assert result["result"] == 25

    def test_exponentiation(self):
        result = json.loads(self.calc.calculate("2 ** 10"))
        assert result["status"] == "success"
        assert result["result"] == 1024

    def test_sqrt(self):
        result = json.loads(self.calc.calculate("sqrt(144)"))
        assert result["status"] == "success"
        assert result["result"] == 12

    def test_scientific_functions(self):
        result = json.loads(self.calc.calculate("sin(pi/2)"))
        assert result["status"] == "success"
        assert abs(result["result"] - 1.0) < 1e-6

    def test_log(self):
        result = json.loads(self.calc.calculate("log(100, 10)"))
        assert result["status"] == "success"
        assert result["result"] == 2

    def test_factorial(self):
        result = json.loads(self.calc.calculate("factorial(10)"))
        assert result["status"] == "success"
        assert result["result"] == 3628800

    def test_invalid_expression(self):
        result = json.loads(self.calc.calculate("not_a_function(42)"))
        assert result["status"] == "error"
        assert "error" in result
        assert "hint" in result

    def test_result_includes_simplified_form(self):
        result = json.loads(self.calc.calculate("2 + 3"))
        assert result["status"] == "success"
        assert "simplified" in result

    # --- unit_convert() tests ---

    def test_celsius_to_fahrenheit(self):
        result = json.loads(self.calc.unit_convert(100, "celsius", "fahrenheit"))
        assert result["status"] == "success"
        assert result["result"] == 212.0

    def test_fahrenheit_to_celsius(self):
        result = json.loads(self.calc.unit_convert(32, "fahrenheit", "celsius"))
        assert result["status"] == "success"
        assert result["result"] == 0.0

    def test_km_to_miles(self):
        result = json.loads(self.calc.unit_convert(10, "km", "miles"))
        assert result["status"] == "success"
        assert abs(result["result"] - 6.2137) < 0.001

    def test_kg_to_lbs(self):
        result = json.loads(self.calc.unit_convert(1, "kg", "lbs"))
        assert result["status"] == "success"
        assert abs(result["result"] - 2.2046) < 0.001

    def test_unsupported_conversion(self):
        result = json.loads(self.calc.unit_convert(1, "gallons", "liters"))
        assert result["status"] == "error"
        assert "supported_units" in result

    def test_case_insensitive_units(self):
        result = json.loads(self.calc.unit_convert(100, "Celsius", "Fahrenheit"))
        assert result["status"] == "success"
        assert result["result"] == 212.0


class TestWebSearchTools:
    """Tests for the WebSearchTools toolkit."""

    def setup_method(self):
        """Create a fresh WebSearchTools instance for each test."""
        self.search = WebSearchTools()

    def test_mock_search_matching_query(self):
        result = json.loads(self.search.search_web("What is machine learning?"))
        assert result["status"] == "success"
        assert len(result["results"]) > 0
        assert "machine learning" in result["results"][0]["title"].lower()

    def test_mock_search_python_query(self):
        result = json.loads(self.search.search_web("Tell me about Python"))
        assert result["status"] == "success"
        assert "python" in result["answer"].lower()

    def test_mock_search_unmatched_query(self):
        result = json.loads(self.search.search_web("quantum computing breakthroughs"))
        assert result["status"] == "success"
        assert "mock" in result["source"].lower()
        assert len(result["results"]) > 0

    def test_search_returns_structured_json(self):
        result = json.loads(self.search.search_web("deep learning"))
        assert "query" in result
        assert "answer" in result
        assert "results" in result
        assert "source" in result
        assert "status" in result

    def test_max_results_parameter(self):
        result = json.loads(self.search.search_web("machine learning", max_results=1))
        assert result["status"] == "success"
        assert len(result["results"]) <= 1

    def test_result_structure(self):
        result = json.loads(self.search.search_web("transformer architecture"))
        for r in result["results"]:
            assert "title" in r
            assert "url" in r
            assert "content" in r
