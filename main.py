from datetime import datetime
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-4.1-mini"  # Use a different model
config["quick_think_llm"] = "gpt-4.1-mini"  # Use a different model
config["max_debate_rounds"] = 1  # Increase debate rounds
config["online_tools"] = True  # Increase debate rounds

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# Use current date for real-time analysis
current_date = datetime.now().strftime("%Y-%m-%d")
print(f"Running real-time analysis for NVDA with current date: {current_date}")

# forward propagate
_, decision = ta.propagate("NVDA", current_date)
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
