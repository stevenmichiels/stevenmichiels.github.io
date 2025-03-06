---
layout: default
title: ETF Forecast
description: A simple website for ETF forecasting and analysis
---

## Welcome to ETF Forecast

This is a simple website for ETF forecasting and analysis.

### SPY ETF Forecast

<div id="forecastPlot" style="width: 100%; height: 1000px;"></div>

<script>
document.addEventListener('DOMContentLoaded', function() {
    // Generate dates for the past 5 years and 1 year into the future
    function generateDates(startYear, numYears) {
        let dates = [];
        let currentDate = new Date(startYear, 0, 1);
        
        for (let i = 0; i < numYears * 12; i++) {
            dates.push(new Date(currentDate));
            currentDate.setMonth(currentDate.getMonth() + 1);
        }
        
        return dates;
    }
    
    // Generate historical dates (5 years of past data)
    const currentYear = new Date().getFullYear();
    const historicalDates = generateDates(currentYear - 5, 5);
    
    // Generate forecast dates (1 year into the future)
    const forecastDates = generateDates(currentYear, 1);
    
    // Generate historical price data with some realistic SPY movement
    function generateHistoricalPrices(startPrice, numPoints, volatility) {
        let prices = [startPrice];
        for (let i = 1; i < numPoints; i++) {
            // Random walk with drift
            const change = (Math.random() - 0.45) * volatility; // Slight upward bias
            const newPrice = prices[i-1] * (1 + change);
            prices.push(newPrice);
        }
        return prices;
    }
    
    // Generate forecast price data with a trend
    function generateForecastPrices(lastHistoricalPrice, numPoints, trend, volatility) {
        let prices = [lastHistoricalPrice];
        for (let i = 1; i < numPoints; i++) {
            // Random walk with specified trend
            const change = trend + (Math.random() - 0.5) * volatility;
            const newPrice = prices[i-1] * (1 + change);
            prices.push(newPrice);
        }
        return prices;
    }
    
    // Generate price data
    const startPrice = 350; // Starting price for SPY 5 years ago
    const historicalPrices = generateHistoricalPrices(startPrice, historicalDates.length, 0.03);
    const forecastPrices = generateForecastPrices(
        historicalPrices[historicalPrices.length - 1], 
        forecastDates.length, 
        0.005, // Slight upward trend
        0.02  // Lower volatility for forecast
    );
    
    // Calculate returns
    function calculateReturns(prices) {
        let returns = [0];
        for (let i = 1; i < prices.length; i++) {
            returns.push((prices[i] / prices[0] - 1) * 100);
        }
        return returns;
    }
    
    // Calculate strategy returns (using a simple moving average crossover as an example)
    function calculateStrategyReturns(prices) {
        // Calculate a simple 50-day and 200-day moving average
        const shortPeriod = 3;
        const longPeriod = 10;
        
        let shortMA = [];
        let longMA = [];
        
        // Fill with nulls until we have enough data
        for (let i = 0; i < longPeriod - 1; i++) {
            shortMA.push(null);
            longMA.push(null);
        }
        
        // Calculate moving averages
        for (let i = longPeriod - 1; i < prices.length; i++) {
            let shortSum = 0;
            let longSum = 0;
            
            for (let j = 0; j < shortPeriod; j++) {
                shortSum += prices[i - j];
            }
            
            for (let j = 0; j < longPeriod; j++) {
                longSum += prices[i - j];
            }
            
            shortMA.push(shortSum / shortPeriod);
            longMA.push(longSum / longPeriod);
        }
        
        // Generate signals (1 for long, 0 for cash)
        let signals = [];
        for (let i = 0; i < prices.length; i++) {
            if (i < longPeriod - 1) {
                signals.push(0); // No signal until we have both MAs
            } else {
                signals.push(shortMA[i] > longMA[i] ? 1 : 0);
            }
        }
        
        // Calculate daily returns
        let dailyReturns = [0];
        for (let i = 1; i < prices.length; i++) {
            dailyReturns.push((prices[i] / prices[i-1] - 1) * signals[i-1]);
        }
        
        // Calculate cumulative returns
        let cumulativeReturns = [0];
        let cumProduct = 1;
        for (let i = 1; i < dailyReturns.length; i++) {
            cumProduct *= (1 + dailyReturns[i]);
            cumulativeReturns.push((cumProduct - 1) * 100);
        }
        
        return {
            signals: signals,
            returns: cumulativeReturns
        };
    }
    
    // Calculate drawdowns
    function calculateDrawdown(returns) {
        let cumulative = returns.map(r => 1 + r/100);
        let rollingMax = [cumulative[0]];
        
        for (let i = 1; i < cumulative.length; i++) {
            rollingMax.push(Math.max(rollingMax[i-1], cumulative[i]));
        }
        
        let drawdowns = [];
        for (let i = 0; i < cumulative.length; i++) {
            drawdowns.push((cumulative[i] - rollingMax[i]) / rollingMax[i] * 100);
        }
        
        return drawdowns;
    }
    
    // Combine historical and forecast data
    const allDates = [...historicalDates, ...forecastDates];
    const allPrices = [...historicalPrices, ...forecastPrices];
    
    // Calculate buy and hold returns
    const buyHoldReturns = calculateReturns(allPrices);
    
    // Calculate strategy returns
    const strategyData = calculateStrategyReturns(allPrices);
    const strategyReturns = strategyData.returns;
    const positions = strategyData.signals;
    
    // Calculate drawdowns
    const strategyDD = calculateDrawdown(strategyReturns);
    const bhDD = calculateDrawdown(buyHoldReturns);
    
    // Create the subplots
    const trace1 = {
        x: allDates,
        y: strategyReturns,
        type: 'scatter',
        mode: 'lines',
        name: 'Strategy Returns',
        line: {
            color: 'blue'
        }
    };
    
    const trace2 = {
        x: allDates,
        y: buyHoldReturns,
        type: 'scatter',
        mode: 'lines',
        name: 'Buy & Hold Returns',
        line: {
            color: 'gray'
        }
    };
    
    const trace3 = {
        x: allDates,
        y: strategyDD,
        type: 'scatter',
        mode: 'lines',
        name: 'Strategy Drawdown',
        line: {
            color: 'red'
        }
    };
    
    const trace4 = {
        x: allDates,
        y: bhDD,
        type: 'scatter',
        mode: 'lines',
        name: 'Buy & Hold Drawdown',
        line: {
            color: 'orange'
        }
    };
    
    const trace5 = {
        x: allDates,
        y: positions,
        type: 'scatter',
        mode: 'lines',
        name: 'Position (1=Long, 0=Cash)',
        line: {
            color: 'green'
        }
    };
    
    // Calculate performance metrics
    const strategyFinalReturn = strategyReturns[strategyReturns.length - 1].toFixed(1);
    const bhFinalReturn = buyHoldReturns[buyHoldReturns.length - 1].toFixed(1);
    const avgStrategyDD = (strategyDD.reduce((a, b) => a + b, 0) / strategyDD.length).toFixed(1);
    const avgBhDD = (bhDD.reduce((a, b) => a + b, 0) / bhDD.length).toFixed(1);
    
    // Create layout
    const layout = {
        title: {
            text: `SPY Portfolio Performance with Forecast Signals<br>` +
                  `Strategy Return: ${strategyFinalReturn}% | Buy&Hold Return: ${bhFinalReturn}%<br>` +
                  `Avg DD Strategy: ${avgStrategyDD}% | Avg DD Buy&Hold: ${avgBhDD}%`,
            font: {
                size: 16
            }
        },
        grid: {
            rows: 3,
            columns: 1,
            pattern: 'independent',
            roworder: 'top to bottom'
        },
        height: 1000,
        showlegend: true,
        legend: {
            orientation: 'h',
            y: 1.1
        },
        annotations: [
            {
                text: 'Cumulative Returns (%)',
                font: {
                    size: 14
                },
                showarrow: false,
                x: 0,
                y: 1.05,
                xref: 'paper',
                yref: 'paper'
            },
            {
                text: 'Drawdown (%)',
                font: {
                    size: 14
                },
                showarrow: false,
                x: 0,
                y: 0.65,
                xref: 'paper',
                yref: 'paper'
            },
            {
                text: 'Position Signal',
                font: {
                    size: 14
                },
                showarrow: false,
                x: 0,
                y: 0.3,
                xref: 'paper',
                yref: 'paper'
            }
        ]
    };
    
    // Add vertical line to separate historical from forecast data
    const shapes = [{
        type: 'line',
        x0: historicalDates[historicalDates.length - 1],
        y0: 0,
        x1: historicalDates[historicalDates.length - 1],
        y1: 1,
        yref: 'paper',
        line: {
            color: 'black',
            width: 2,
            dash: 'dash'
        }
    }];
    
    layout.shapes = shapes;
    
    // Create the plot
    Plotly.newPlot('forecastPlot', [trace1, trace2, trace3, trace4, trace5], layout);
    
    // Add annotation for forecast start
    const forecastAnnotation = {
        x: historicalDates[historicalDates.length - 1],
        y: strategyReturns[historicalDates.length - 1],
        text: 'Forecast Start',
        showarrow: true,
        arrowhead: 2,
        arrowsize: 1,
        arrowwidth: 2,
        ax: -40,
        ay: -40
    };
    
    Plotly.relayout('forecastPlot', {
        annotations: [...layout.annotations, forecastAnnotation]
    });
});
</script> 