#!/usr/bin/env python3
"""
Quick test: ensure chart_data produced by StockAnalyzer has aligned series for ECharts category xAxis.
Checks:
 - len(kline) == len(portfolio)
 - every trade_marker.date exists in kline dates
"""
from main import StockAnalyzer


def test_chart_data_alignment():
    analyzer = StockAnalyzer()
    result = analyzer.analyze_stock("600036.SS")
    assert result.get('success', False), f"Analyze failed: {result.get('error')}"

    chart = result.get('chart_data')
    assert chart is not None, "chart_data missing"

    kline = chart.get('kline', [])
    portfolio = chart.get('portfolio', [])
    trade_markers = chart.get('trade_markers', [])

    # Basic length alignment: portfolio should have same number of points as kline (resampled)
    assert len(kline) > 0, "kline is empty"
    assert len(kline) == len(portfolio), f"kline length {len(kline)} != portfolio length {len(portfolio)}"

    # All trade marker dates should be present in kline dates
    kdates = {p['date'] for p in kline}
    for t in trade_markers:
        assert t['date'] in kdates, f"trade marker date {t['date']} not in kline dates"

    print("chart_data alignment checks passed: kline, portfolio and trade_markers are aligned")


if __name__ == '__main__':
    test_chart_data_alignment()
