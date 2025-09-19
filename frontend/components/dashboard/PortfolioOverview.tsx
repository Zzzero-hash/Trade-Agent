'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { formatCurrency, formatPercentage } from '@/lib/utils';
import { TrendingUp, TrendingDown, DollarSign, PieChart } from 'lucide-react';

export function PortfolioOverview() {
  const [timeframe, setTimeframe] = useState('30d');

  const { data: portfolio, isLoading: portfolioLoading } = useQuery(
    'portfolio',
    () => apiClient.getPortfolio(),
    { refetchInterval: 30000 } // Refresh every 30 seconds
  );

  const { data: performance, isLoading: performanceLoading } = useQuery(
    ['portfolio-performance', timeframe],
    () => apiClient.getPortfolioPerformance(parseInt(timeframe.replace('d', ''))),
    { refetchInterval: 60000 } // Refresh every minute
  );

  if (portfolioLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (!portfolio) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">No portfolio data available</p>
      </div>
    );
  }

  const totalReturn = performance?.total_return || 0;
  const isPositive = totalReturn >= 0;

  return (
    <div className="space-y-6">
      {/* Portfolio Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="card">
          <div className="flex items-center">
            <DollarSign className="h-8 w-8 text-primary-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total Value</p>
              <p className="text-2xl font-bold text-gray-900">
                {formatCurrency(portfolio.total_value)}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <PieChart className="h-8 w-8 text-success-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Cash Balance</p>
              <p className="text-2xl font-bold text-gray-900">
                {formatCurrency(portfolio.cash_balance)}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            {isPositive ? (
              <TrendingUp className="h-8 w-8 text-success-600" />
            ) : (
              <TrendingDown className="h-8 w-8 text-danger-600" />
            )}
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total Return</p>
              <p className={`text-2xl font-bold ${
                isPositive ? 'text-success-600' : 'text-danger-600'
              }`}>
                {formatPercentage(totalReturn)}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <TrendingUp className="h-8 w-8 text-warning-600" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Positions</p>
              <p className="text-2xl font-bold text-gray-900">
                {Object.keys(portfolio.positions).length}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Timeframe Selector */}
      <div className="flex space-x-2">
        {['7d', '30d', '90d', '1y'].map((period) => (
          <button
            key={period}
            onClick={() => setTimeframe(period)}
            className={`px-3 py-1 rounded-md text-sm font-medium ${
              timeframe === period
                ? 'bg-primary-100 text-primary-700'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            {period}
          </button>
        ))}
      </div>

      {/* Positions Table */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Current Positions</h3>
        
        {Object.keys(portfolio.positions).length === 0 ? (
          <p className="text-gray-500 text-center py-8">No positions found</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Symbol
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Quantity
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Avg Cost
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Current Price
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Market Value
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    P&L
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {Object.entries(portfolio.positions).map(([symbol, position]) => {
                  const marketValue = Math.abs(position.quantity) * position.current_price;
                  const pnlPercent = position.unrealized_pnl / (position.avg_cost * Math.abs(position.quantity));
                  
                  return (
                    <tr key={symbol}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {symbol}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {position.quantity.toFixed(2)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatCurrency(position.avg_cost)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatCurrency(position.current_price)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatCurrency(marketValue)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <div className={`${
                          position.unrealized_pnl >= 0 ? 'text-success-600' : 'text-danger-600'
                        }`}>
                          {formatCurrency(position.unrealized_pnl)}
                          <br />
                          <span className="text-xs">
                            ({formatPercentage(pnlPercent)})
                          </span>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Performance Metrics */}
      {performance && !performanceLoading && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Performance Metrics ({timeframe})
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-gray-600">Sharpe Ratio</p>
              <p className="text-lg font-semibold">
                {performance.sharpe_ratio?.toFixed(2) || 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Max Drawdown</p>
              <p className="text-lg font-semibold text-danger-600">
                {formatPercentage(performance.max_drawdown || 0)}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Volatility</p>
              <p className="text-lg font-semibold">
                {formatPercentage(performance.volatility || 0)}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Win Rate</p>
              <p className="text-lg font-semibold text-success-600">
                {formatPercentage(performance.win_rate || 0)}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}