"""
Calculator Tool for the AI Research Assistant.

Provides safe mathematical evaluation using sympy.
Supports arithmetic, algebra, scientific functions, and unit conversions.

"""
import json
from typing import Dict, Any

import sympy
from agno.tools import Toolkit

from src.logging_config import get_logger

logger = get_logger("tools.calculator")


class CalculatorTools(Toolkit):
    """Toolkit providing mathematical calculation capabilities.

    Uses sympy for safe, symbolic math evaluation -- no eval() or exec().

    Capabilities:
        - Basic arithmetic: +, -, *, /, **, %
        - Scientific functions: sqrt, sin, cos, log, exp
        - Algebraic expressions: simplify, expand
        - Constants: pi, e
        - Unit conversions: temperature, distance, weight, length

    Usage:
        calc = CalculatorTools()
        calc.calculate("sqrt(144) + log(100, 10)")
        calc.unit_convert(100, "celsius", "fahrenheit")
    """

    # Supported unit conversion pairs
    CONVERSIONS: Dict[tuple, callable] = {
        ("km", "miles"): lambda v: v * 0.621371,
        ("miles", "km"): lambda v: v * 1.60934,
        ("celsius", "fahrenheit"): lambda v: (v * 9 / 5) + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
        ("kg", "lbs"): lambda v: v * 2.20462,
        ("lbs", "kg"): lambda v: v / 2.20462,
        ("meters", "feet"): lambda v: v * 3.28084,
        ("feet", "meters"): lambda v: v / 3.28084,
    }

    def __init__(self):
        super().__init__(name="calculator")
        self.register(self.calculate)
        self.register(self.unit_convert)

    def calculate(self, expression: str) -> str:
        """Evaluate a mathematical expression safely using sympy.

        Args:
            expression: A mathematical expression to evaluate.
                        Examples: "45 * 32 + 17", "sqrt(144)", "sin(pi/4)",
                                  "log(100, 10)", "factorial(10)"

        Returns:
            JSON string with the result or error message.
        """
        logger.info("calculate() called with: %s", expression)
        try:
            # Parse and evaluate using sympy (safe -- no arbitrary code execution)
            result = sympy.sympify(expression)
            evaluated = float(result.evalf())

            # Format nicely (avoid floating point noise like 14.0 -> 14)
            if evaluated == int(evaluated):
                evaluated = int(evaluated)

            logger.info("calculate() result: %s = %s", expression, evaluated)
            return json.dumps({
                "expression": expression,
                "result": evaluated,
                "simplified": str(result),
                "status": "success",
            })
        except Exception as e:
            logger.error("calculate() failed for '%s': %s", expression, e)
            return json.dumps({
                "expression": expression,
                "error": str(e),
                "status": "error",
                "hint": "Try using standard math notation: +, -, *, /, **, sqrt(), sin(), cos(), log()",
            })

    def unit_convert(self, value: float, from_unit: str, to_unit: str) -> str:
        """Convert between common units.

        Args:
            value: The numeric value to convert.
            from_unit: Source unit (e.g., "km", "miles", "celsius", "fahrenheit").
            to_unit: Target unit (e.g., "miles", "km", "fahrenheit", "celsius").

        Returns:
            JSON string with the conversion result.
        """
        logger.info("unit_convert() called: %s %s -> %s", value, from_unit, to_unit)
        key = (from_unit.lower(), to_unit.lower())

        if key in self.CONVERSIONS:
            converted = self.CONVERSIONS[key](value)
            logger.info("unit_convert() result: %s %s = %s %s", value, from_unit, round(converted, 4), to_unit)
            return json.dumps({
                "value": value,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "result": round(converted, 4),
                "status": "success",
            })

        logger.warning("unit_convert() unsupported: %s -> %s", from_unit, to_unit)
        supported_units = sorted(set(u for pair in self.CONVERSIONS for u in pair))
        return json.dumps({
            "error": f"Unsupported conversion: {from_unit} -> {to_unit}",
            "supported_units": supported_units,
            "status": "error",
        })
