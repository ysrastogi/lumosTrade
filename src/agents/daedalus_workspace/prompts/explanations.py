"""
Explanation and insight generation prompts
Used to generate natural language explanations of strategy results
"""

# Prompt for generating explanations of strategy results
EXPLANATION_PROMPT = """
Generate a clear, concise explanation of the following trading strategy results.
Use a professional, informative tone. Include key metrics and insights where available.

User query: {query}

Intent: {intent}

Results: {results}

Provide a 2-3 sentence explanation with the most important information.
"""

# Prompt for suggesting next actions based on results
NEXT_ACTIONS_PROMPT = """
Based on the trading strategy results below, suggest 2-3 logical next steps the user might want to take.
Return only a JSON array of strings, each representing a next action.
Example: ["Optimize the strategy parameters", "Run a Monte Carlo simulation", "Compare with benchmark"]

User query: {query}
Intent: {intent}
Results: {results}
"""

# Default explanations for each intent when LLM generation fails
DEFAULT_EXPLANATIONS = {
    "create_strategy": "Created strategy '{strategy_name}' of type {strategy_type}.",
    "simulate": "Simulated strategy '{strategy_name}' with Sharpe ratio {sharpe_ratio:.2f} and return {total_return:.1f}%.",
    "optimize": "Optimized strategy with {method} method. Best parameters: {best_parameters} with score {best_score:.2f}.",
    "analyze": "Completed {analysis_type} analysis. See results for details.",
    "forecast": "Generated Monte Carlo forecast with {n_paths} paths over {n_days} days.",
    "compare": "Compared {strategy_count} strategies. Best strategy: {best_strategy}.",
    "status": "Current state: {strategy_count} strategies, {simulation_count} simulations.",
    "generate_ideas": "Generated {idea_count} strategy ideas based on market conditions."
}