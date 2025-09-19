'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { useSignalSubscription } from '@/lib/websocket-context';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { formatDate, getActionColor, getConfidenceColor } from '@/lib/utils';
import { Brain, TrendingUp, AlertCircle, Zap } from 'lucide-react';

export function TradingSignals() {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [isGenerating, setIsGenerating] = useState(false);
  const queryClient = useQueryClient();

  // Real-time signals from WebSocket
  const realtimeSignals = useSignalSubscription();

  // Historical signals
  const { data: signalHistory, isLoading } = useQuery(
    ['signal-history', selectedSymbol],
    () => apiClient.getSignalHistory({ symbol: selectedSymbol, limit: 50 }),
    { refetchInterval: 30000 }
  );

  // Signal performance
  const { data: performance } = useQuery(
    ['signal-performance', selectedSymbol],
    () => apiClient.getSignalPerformance(selectedSymbol, 30),
    { refetchInterval: 60000 }
  );

  // Generate signal mutation
  const generateSignalMutation = useMutation(
    (symbol: string) => apiClient.generateTradingSignal(symbol),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['signal-history']);
        setIsGenerating(false);
      },
      onError: () => {
        setIsGenerating(false);
      },
    }
  );

  const handleGenerateSignal = async () => {
    setIsGenerating(true);
    generateSignalMutation.mutate(selectedSymbol);
  };

  const popularSymbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX'];

  // Combine real-time and historical signals
  const allSignals = [...realtimeSignals, ...(signalHistory || [])];
  const uniqueSignals = allSignals.filter((signal, index, self) => 
    index === self.findIndex(s => s.timestamp === signal.timestamp && s.symbol === signal.symbol)
  );

  return (
    <div className="space-y-6">
      {/* Signal Generation */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center">
            <Brain className="h-5 w-5 mr-2 text-primary-600" />
            Generate Trading Signal
          </h3>
          <div className="flex items-center space-x-2">
            <Zap className="h-4 w-4 text-warning-500" />
            <span className="text-sm text-gray-600">AI-Powered</span>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div>
            <label className="label">Select Symbol</label>
            <select
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              className="input w-32"
            >
              {popularSymbols.map((symbol) => (
                <option key={symbol} value={symbol}>
                  {symbol}
                </option>
              ))}
            </select>
          </div>

          <div className="flex-1">
            <label className="label">Custom Symbol</label>
            <input
              type="text"
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value.toUpperCase())}
              className="input"
              placeholder="Enter symbol (e.g., AAPL)"
            />
          </div>

          <div className="pt-6">
            <button
              onClick={handleGenerateSignal}
              disabled={isGenerating || !selectedSymbol}
              className="btn-primary"
            >
              {isGenerating ? (
                <>
                  <LoadingSpinner size="sm" className="mr-2" />
                  Generating...
                </>
              ) : (
                'Generate Signal'
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Performance Summary */}
      {performance && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <TrendingUp className="h-5 w-5 mr-2 text-success-600" />
            Signal Performance ({selectedSymbol})
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-gray-600">Total Signals</p>
              <p className="text-2xl font-bold text-gray-900">{performance.total_signals}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Accuracy</p>
              <p className="text-2xl font-bold text-success-600">
                {(performance.accuracy * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Total Return</p>
              <p className={`text-2xl font-bold ${
                performance.total_return >= 0 ? 'text-success-600' : 'text-danger-600'
              }`}>
                {(performance.total_return * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Sharpe Ratio</p>
              <p className="text-2xl font-bold text-primary-600">
                {performance.sharpe_ratio.toFixed(2)}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Real-time Signals Alert */}
      {realtimeSignals.length > 0 && (
        <div className="bg-primary-50 border border-primary-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertCircle className="h-5 w-5 text-primary-600 mr-2" />
            <span className="text-sm font-medium text-primary-800">
              {realtimeSignals.length} new signal{realtimeSignals.length > 1 ? 's' : ''} received in real-time
            </span>
          </div>
        </div>
      )}

      {/* Signal History */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Signals</h3>
        
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <LoadingSpinner size="md" />
          </div>
        ) : uniqueSignals.length === 0 ? (
          <div className="text-center py-8">
            <p className="text-gray-500">No signals found for {selectedSymbol}</p>
            <p className="text-sm text-gray-400 mt-1">
              Generate your first signal using the form above
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {uniqueSignals.slice(0, 10).map((signal, index) => (
              <div
                key={`${signal.timestamp}-${signal.symbol}-${index}`}
                className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div>
                      <h4 className="font-semibold text-gray-900">{signal.symbol}</h4>
                      <p className="text-sm text-gray-500">
                        {formatDate(signal.timestamp)}
                      </p>
                    </div>
                    
                    <div className={`badge ${getActionColor(signal.action)}`}>
                      {signal.action}
                    </div>
                    
                    <div>
                      <p className="text-sm text-gray-600">Confidence</p>
                      <p className={`font-semibold ${getConfidenceColor(signal.confidence)}`}>
                        {(signal.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                    
                    <div>
                      <p className="text-sm text-gray-600">Position Size</p>
                      <p className="font-semibold text-gray-900">
                        {(signal.position_size * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    {signal.target_price && (
                      <div>
                        <p className="text-sm text-gray-600">Target</p>
                        <p className="font-semibold text-success-600">
                          ${signal.target_price.toFixed(2)}
                        </p>
                      </div>
                    )}
                    {signal.stop_loss && (
                      <div className="mt-1">
                        <p className="text-sm text-gray-600">Stop Loss</p>
                        <p className="font-semibold text-danger-600">
                          ${signal.stop_loss.toFixed(2)}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
                
                <div className="mt-2 text-xs text-gray-500">
                  Model: {signal.model_version}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}