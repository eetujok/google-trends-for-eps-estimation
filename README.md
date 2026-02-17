<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/2df25512-9517-46a2-be97-041bb070f56d" />

# Branded Keyword Spike Impact on Earning Surprises

0. Introduction

This project explores whether search interest in a company’s brands can impact company financial performance. Using Google Trends data, the analysis measures "relative popularity" spikes, assings them to fisical quarters and analyzes what impact the spikes have on earnings.

0.1 Starting Hypothesis

"Positive spikes in a company's branded keyword traffic is indicative of a positive earnings surprise."
For example, a sudden increase in searches for 'salomon' might predict a stronger-than-expected earnings report for Amer Sports Inc.

0.2 Key Results (TLDR)

The analysis confirms a statistically significant relationship:

    Lower Risk of Negative Surprises: Quarters with high relative popularity spikes saw a decrease in negative earnings surprises (ΔP(EPS < 0) = -0.0245).

    Higher Probability of Positive Surprises: These same quarters showed an increase in positive surprises (ΔP(EPS > 0) = +0.0361).

    Visual Confirmation: The difference in earnings distributions between spike quarters and baseline quarters is visually evident in the comparative histograms and CDF plots.

0.3 Methodology & Data

    Keyword Generation: Generated using the Gemini-2.5-Flash LLM to identify branded keywords w/ company context.

    Data Sources:

        Earnings Data: Quarterly reports for consumer discretionary stocks via EODHD.

        Search Trends: Time-series data scraped from Google Trends via Oxylabs.

    Signal Detection: Utilized STL Decomposition to separate search trends into seasonal, trend, and residual components, allowing for the isolation of specific "event" spikes.

    Statistical Validation: Employed the Kolmogorov-Smirnov (KS) test to determine if the distribution of earnings surprises during keyword spikes significantly differed from the general population.

0.4 Project Implementation

The project demonstrates core data science competencies:

    Feature Engineering: Extracting residual "spikes" from noisy time-series data.

    Statistical Analysis: Using CDF difference plots and KS tests for distribution comparison.

    Parallel Computing: Implementing CPU parallelization for large-scale data processing.

0.5 Future Improvements

    Normalization: Account for the relative volume between different keywords.

    Window Optimization: Refine the "event windows" to better capture the lead-up to earnings reports.

    Spike Consensus: Implement a filter requiring multiple keywords to spike before flagging a quarter.

Disclaimer: This project was created to demonstrate interest in data analysis and feature engineering. Some helper functions for plotting and parallelization were AI-assisted.
