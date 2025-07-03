#!/usr/bin/env python3
"""
Example: Macro Economic Analysis with Enhanced Data Sources

This example demonstrates how to use the TradingAgents macro analysis tools
to get comprehensive economic data including:
- Economic indicators (Fed funds, CPI, PPI, unemployment, NFP, GDP, PMI)
- Treasury yield curve analysis
- Fed calendar and policy updates
- Economic data analysis and trading implications

Note: Twitter/X integration has been removed for system simplification.
"""

import os
import sys
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.dataflows.interface import (
    get_macro_analysis,
    get_economic_indicators, 
    get_yield_curve_analysis,
)


def main():
    # Get current date for analysis
    current_date = datetime.now().strftime("%Y-%m-%d")
    print(f"📊 Running Macro Economic Analysis for {current_date}")
    print("=" * 60)
    
    try:
        # 1. Get comprehensive macro analysis
        print("🔍 Getting comprehensive macro economic analysis...")
        macro_analysis = get_macro_analysis(
            curr_date=current_date,
            lookback_days=90
        )
        print(f"✅ Retrieved macro analysis ({len(macro_analysis)} characters)")
        print(macro_analysis[:500] + "..." if len(macro_analysis) > 500 else macro_analysis)
        print("\n" + "="*60 + "\n")
        
        # 2. Get detailed economic indicators
        print("📈 Getting economic indicators report...")
        indicators = get_economic_indicators(
            curr_date=current_date,
            lookback_days=90
        )
        print(f"✅ Retrieved indicators report ({len(indicators)} characters)")
        print(indicators[:300] + "..." if len(indicators) > 300 else indicators)
        print("\n" + "="*60 + "\n")
        
        # 3. Get yield curve analysis
        print("📊 Getting yield curve analysis...")
        yield_curve = get_yield_curve_analysis(curr_date=current_date)
        print(f"✅ Retrieved yield curve analysis ({len(yield_curve)} characters)")
        print(yield_curve[:300] + "..." if len(yield_curve) > 300 else yield_curve)
        print("\n" + "="*60 + "\n")
        
        # Summary
        print("🎯 ANALYSIS SUMMARY")
        print("=" * 30)
        print("✅ Macro Economic Analysis: Complete")
        print("✅ Economic Indicators: Complete") 
        print("✅ Yield Curve Analysis: Complete")
        print("\n💡 Economic data sources successfully integrated!")
        print("📊 Ready for trading decisions with comprehensive macro intelligence")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("💡 Make sure you have the required API keys configured:")
        print("   - FRED_API_KEY for economic data")


if __name__ == "__main__":
    main() 