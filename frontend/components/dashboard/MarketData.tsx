'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { useMarketDataSubscription } from '@/lib/websocket-context';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { formatCurrency, formatPercentage } from '@/lib/utils';
import { TrendingUp, TrendingDown, BarChart3 } from 'lucide-react';

export function MarketData() {
  const [selectedSymbols, setSelectedSymbols] = useState(['AAPL', 'GOOGL', 'MSFT']);
  const [timeframe, setTimeframe] = useState('1h');

  // Real-time market data from WebSocket
  const realtimeData = useMarketDataSubscription(selectedSymbols);

  // Historical market data
  const { data: historicalData, isLoading } = useQuery(
    ['market-data', selectedSymbols[0], timeframe],
    () => apiClient.getMarketData(selectedSymbols[0], timeframe, 100),
    { 
      enabled: selectedSymbols.length > 0,
      refetchInterval: 30000 
    }
  );

  const popularSymbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX'];

  return (
    <div className="space-y-6">
      {/* Symbol Selection */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <BarChart3 className="h-5 w-5 mr-2 text-primary-600" />
          Market Data
        </h3>
        
        <div className="flex items-center space-x-4 mb-4">
          <div>
            <label className="label">Symbols</label>
            <div className="flex flex-wrap gap-2">
              {popularSymbols.map((symbol) => (
                <button
                  key={symbol}
                  onClick={() => {
                    if (selectedSymbols.includes(symbol)) {
                      setSelectedSymbols(prev => prev.filter(s => s !== symbol));
                    } else {
                      setSelectedSymbols(prev => [...prev, symbol]);
                    }
                  }}
                  className={`px-3 py-1 rounded-md text-sm font-medium ${
                    selectedSymbols.includes(symbol)
                      ? 'bg-primary-100 text-primary-700'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {symbol}
                </button>
              ))}
            </div>
          </div>
          
          <div>
            <label className="label">Timeframe</label>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              className="input w-24"
            >
              <option value="1m">1m</option>
              <option value="5m">5m</option>
              <option value="15m">15m</option>
              <option value="1h">1h</option>
              <option value="4h">4h</option>
              <option value="1d">1d</option>
            </select>
          </div>
        </div>
      </div>

      {/* Real-time Data Cards */}
      {Object.keys(realtimeData).length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {Object.entries(realtimeData).map(([symbol, data]) => (
            <div key={symbol} className="card">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-semibold text-gray-900">{symbol}</h4>
                <span className="text-xs text-gray-500">Live</span>
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Price</span>
                  <span className="font-semibold">
                    {formatCurrency(data.price || 0)}
                  </span>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Change</span>
                  <div className={`flex items-center ${
                    (data.change || 0) >= 0 ? 'text-success-600' : 'text-danger-600'
                  }`}>
                    {(data.change || 0) >= 0 ? (
                      <TrendingUp className="h-4 w-4 mr-1" />
                    ) : (
                      <TrendingDown className="h-4 w-4 mr-1" />
                    )}
                    <span className="font-semibold">
                      {formatCurrency(data.change || 0)} ({formatPercentage(data.change_percent || 0)})
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Historical Data Chart Placeholder */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Price Chart - {selectedSymbols[0]} ({timeframe})
        </h3>
        
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <LoadingSpinner size="lg" />
          </div>
        ) : historicalData ? (
          <div className="h-64 flex items-center justify-center bg-gray-50 rounded-lg">
            <div className="text-center">
              <BarChart3 className="h-12 w-12 text-gray-400 mx-auto mb-2" />
              <p className="text-gray-500">Chart visualization would go here</p>
              <p className="text-sm text-gray-400">
                {historicalData.data.length} data points available
              </p>
            </div>
          </div>
        ) : (
          <div className="h-64 flex items-center justify-center bg-gray-50 rounded-lg">
            <p className="text-gray-500">No data available</p>
          </div>
        )}
      </div>

      {/* Market Summary Table */}
      {historicalData && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Market Summary</h3>
          
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Metric
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Value
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {historicalData.data.length > 0 && (
                  <>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        Latest Price
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatCurrency(historicalData.data[historicalData.data.length - 1].close)}
                      </td>
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        High (Period)
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatCurrency(Math.max(...historicalData.data.map(d => d.high)))}
                      </td>
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        Low (Period)
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatCurrency(Math.min(...historicalData.data.map(d => d.low)))}
                      </td>
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        Volume (Latest)
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {historicalData.data[historicalData.data.length - 1].volume.toLocaleString()}
                      </td>
                    </tr>
                  </>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}