# Options Pricing, Risk and Strategy Analysis Platform

## Overview
This project involved building a Python-based analytics platform for option pricing, risk analysis, and strategy evaluation. It combined classical option pricing models with numerical and data-driven methods to analyze options portfolios and their risk behavior.

The platform was designed to mirror the structure of real-world derivatives analytics tools at a simplified but practical level.

## Objective
To develop a modular platform capable of pricing options, computing Greeks, aggregating portfolio-level risk, and performing scenario-based stress testing.

## Models & Techniques Implemented
- Black–Scholes Model for European options
- Binomial Tree Model
- Monte Carlo Simulation
- Machine learning techniques for exploratory analysis and pattern recognition

## Tools & Technologies
- Python
- NumPy, Pandas, SciPy
- QuantLib
- Scikit-learn
- Streamlit (interactive dashboards)
- FastAPI (API-based access)
- SQL (portfolio and trade storage)
- Docker (containerized deployment)

## Platform Capabilities
- Pricing of options across multiple models
- Full Greeks computation and aggregation at portfolio level
- Scenario analysis and stress testing under varying market conditions
- Interactive dashboards for strategy analysis and visualization
- Multiple access layers including web interface, API, and CLI

## System Design Notes
- The pricing engine was implemented in a modular manner to support model extensibility
- Portfolio data and trades were structured using a relational database
- Separation of analytics, data storage, and presentation layers was maintained

## Key Learnings
- Translating derivatives pricing theory into working code
- Understanding the impact of volatility, time decay, and interest rates on option values
- Designing scalable analytics workflows for financial applications
- Building end-to-end analytical platforms combining finance and software components

## Use Case
This project represents a simplified options analytics and risk management platform that could be used by analysts or traders to evaluate option strategies and portfolio risk.

