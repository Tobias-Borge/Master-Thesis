# Optimal Stock Trading Strategies: A Forecasting And Control Problem

## Overview
This repository contains my master's thesis project completed as part of my MSc in Robotics and Intelligent Systems at the University of Oslo.

Financial markets exhibit complex characteristics such as non-linearity, non-stationarity, volatility clustering, and changing market regimes, making accurate forecasting and portfolio optimization particularly challenging. Traditional statistical models often struggle to capture these dynamics, while standalone deep learning models frequently ignore important statistical properties of financial time series.

My work presents a hybrid financial forecasting and portfolio optimization framework that combines statistical modeling, deep learning, signal processing, and control theory into a single end-to-end decision-making pipeline.

Rather than relying on a single forecasting model, the framework adopts a divide-and-conquer approach. Multivariate Empirical Mode Decomposition (MEMD) first decomposes multivariate OHLCV market data into frequency-specific components, allowing different models to specialize in the characteristics they model best. Low-frequency components are modeled using ARIMA to capture long-term linear trends, while high-frequency components are augmented with GARCH volatility estimates to explicitly account for changing market risk. Residual nonlinear relationships are learned using Long Short-Term Memory (LSTM) neural networks before all forecasts are recombined into a unified prediction.

Unlike conventional forecasting systems that stop after predicting future prices, the proposed framework integrates forecasting directly with sequential decision-making. Predicted returns are continuously fed into a Model Predictive Control (MPC) optimizer, which dynamically allocates capital across multiple assets while respecting realistic investment constraints including transaction costs, portfolio diversification, turnover limits, and budget constraints.

The complete pipeline was evaluated on historical market data across multiple stocks and ETFs and compared against standalone statistical models, deep learning models, existing hybrid approaches, and traditional portfolio allocation strategies. The project includes extensive forecasting evaluation, component-wise ablation studies, statistical significance testing, portfolio backtesting, transaction cost analysis, and robustness experiments to assess both predictive accuracy and real-world investment performance.

The result is a holistic AI-driven investment framework where forecasting, uncertainty estimation, and portfolio optimization operate as an integrated adaptive system rather than as separate stages.

## Framework

![Pipeline](Hybrid Pipeline.drawio (1).png)
