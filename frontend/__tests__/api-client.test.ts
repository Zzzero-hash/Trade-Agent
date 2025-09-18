/**
 * Tests for API client and WebSocket functionality.
 * 
 * This module tests the frontend API client, authentication,
 * and WebSocket connections for real-time data streaming.
 */

import { apiClient } from '@/lib/api-client';
import axios from 'axios';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('ApiClient', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    
    // Mock axios.create to return a mock instance
    const mockAxiosInstance = {
      get: jest.fn(),
      post: jest.fn(),
      put: jest.fn(),
      delete: jest.fn(),
      interceptors: {
        request: { use: jest.fn() },
        response: { use: jest.fn() },
      },
    };
    
    mockedAxios.create.mockReturnValue(mockAxiosInstance as any);
  });

  describe('Authentication', () => {
    it('should login successfully', async () => {
      const mockResponse = {
        data: {
          access_token: 'test-token',
          token_type: 'bearer',
          expires_in: 3600,
          user: {
            id: 'user123',
            email: 'test@example.com',
            username: 'testuser',
            role: 'premium',
            is_active: true,
            created_at: '2023-12-01T00:00:00Z',
            daily_signal_count: 0,
            daily_signal_limit: 999999,
          },
        },
      };

      const mockPost = jest.fn().mockResolvedValue(mockResponse);
      (apiClient as any).client.post = mockPost;

      const result = await apiClient.login('test@example.com', 'password');

      expect(mockPost).toHaveBeenCalledWith('/api/v1/auth/login', {
        email: 'test@example.com',
        password: 'password',
      });
      expect(result).toEqual(mockResponse.data);
    });

    it('should register successfully', async () => {
      const mockResponse = {
        data: {
          access_token: 'test-token',
          token_type: 'bearer',
          expires_in: 3600,
          user: {
            id: 'user123',
            email: 'test@example.com',
            username: 'testuser',
            role: 'trial',
            is_active: true,
            created_at: '2023-12-01T00:00:00Z',
            trial_expires_at: '2023-12-08T00:00:00Z',
            daily_signal_count: 0,
            daily_signal_limit: 5,
          },
        },
      };

      const mockPost = jest.fn().mockResolvedValue(mockResponse);
      (apiClient as any).client.post = mockPost;

      const result = await apiClient.register('test@example.com', 'testuser', 'password');

      expect(mockPost).toHaveBeenCalledWith('/api/v1/auth/register', {
        email: 'test@example.com',
        username: 'testuser',
        password: 'password',
      });
      expect(result).toEqual(mockResponse.data);
    });

    it('should get current user', async () => {
      const mockResponse = {
        data: {
          id: 'user123',
          email: 'test@example.com',
          username: 'testuser',
          role: 'premium',
          is_active: true,
          created_at: '2023-12-01T00:00:00Z',
          daily_signal_count: 5,
          daily_signal_limit: 999999,
        },
      };

      const mockGet = jest.fn().mockResolvedValue(mockResponse);
      (apiClient as any).client.get = mockGet;

      const result = await apiClient.getCurrentUser();

      expect(mockGet).toHaveBeenCalledWith('/api/v1/auth/me');
      expect(result).toEqual(mockResponse.data);
    });

    it('should handle authentication errors', async () => {
      const mockError = {
        response: {
          status: 401,
          data: { detail: 'Invalid credentials' },
        },
      };

      const mockPost = jest.fn().mockRejectedValue(mockError);
      (apiClient as any).client.post = mockPost;

      await expect(apiClient.login('test@example.com', 'wrongpassword')).rejects.toEqual(mockError);
    });
  });

  describe('Trading Signals', () => {
    it('should generate trading signal', async () => {
      const mockResponse = {
        data: {
          symbol: 'AAPL',
          action: 'BUY',
          confidence: 0.85,
          position_size: 0.1,
          target_price: 155.0,
          stop_loss: 145.0,
          timestamp: '2023-12-01T10:00:00Z',
          model_version: 'cnn-lstm-v1.0',
        },
      };

      const mockPost = jest.fn().mockResolvedValue(mockResponse);
      (apiClient as any).client.post = mockPost;

      const result = await apiClient.generateTradingSignal('AAPL');

      expect(mockPost).toHaveBeenCalledWith('/api/v1/trading/signals/generate', null, {
        params: { symbol: 'AAPL' },
      });
      expect(result).toEqual(mockResponse.data);
    });

    it('should get signal history', async () => {
      const mockResponse = {
        data: [
          {
            symbol: 'AAPL',
            action: 'BUY',
            confidence: 0.85,
            position_size: 0.1,
            timestamp: '2023-12-01T10:00:00Z',
            model_version: 'cnn-lstm-v1.0',
          },
        ],
      };

      const mockGet = jest.fn().mockResolvedValue(mockResponse);
      (apiClient as any).client.get = mockGet;

      const result = await apiClient.getSignalHistory({
        symbol: 'AAPL',
        limit: 10,
      });

      expect(mockGet).toHaveBeenCalledWith('/api/v1/trading/signals/history', {
        params: { symbol: 'AAPL', limit: 10 },
      });
      expect(result).toEqual(mockResponse.data);
    });

    it('should get signal performance', async () => {
      const mockResponse = {
        data: {
          symbol: 'AAPL',
          period_days: 30,
          total_signals: 25,
          accuracy: 0.72,
          total_return: 0.15,
          sharpe_ratio: 1.2,
          max_drawdown: -0.05,
          win_rate: 0.68,
        },
      };

      const mockGet = jest.fn().mockResolvedValue(mockResponse);
      (apiClient as any).client.get = mockGet;

      const result = await apiClient.getSignalPerformance('AAPL', 30);

      expect(mockGet).toHaveBeenCalledWith('/api/v1/trading/signals/performance', {
        params: { symbol: 'AAPL', days: 30 },
      });
      expect(result).toEqual(mockResponse.data);
    });
  });

  describe('Portfolio Management', () => {
    it('should get portfolio', async () => {
      const mockResponse = {
        data: {
          user_id: 'user123',
          positions: {
            AAPL: {
              symbol: 'AAPL',
              quantity: 100,
              avg_cost: 150.0,
              current_price: 155.0,
              unrealized_pnl: 500.0,
              realized_pnl: 0.0,
            },
          },
          cash_balance: 10000.0,
          total_value: 25500.0,
          last_updated: '2023-12-01T10:00:00Z',
        },
      };

      const mockGet = jest.fn().mockResolvedValue(mockResponse);
      (apiClient as any).client.get = mockGet;

      const result = await apiClient.getPortfolio();

      expect(mockGet).toHaveBeenCalledWith('/api/v1/trading/portfolio');
      expect(result).toEqual(mockResponse.data);
    });

    it('should rebalance portfolio', async () => {
      const mockResponse = {
        data: {
          message: 'Portfolio rebalancing initiated',
          rebalance_id: 'rebal_123',
          estimated_trades: [
            { symbol: 'AAPL', action: 'BUY', quantity: 10 },
          ],
        },
      };

      const mockPost = jest.fn().mockResolvedValue(mockResponse);
      (apiClient as any).client.post = mockPost;

      const targetAllocation = {
        AAPL: 0.3,
        GOOGL: 0.2,
        MSFT: 0.2,
        TSLA: 0.1,
        CASH: 0.2,
      };

      const result = await apiClient.rebalancePortfolio(targetAllocation);

      expect(mockPost).toHaveBeenCalledWith('/api/v1/trading/portfolio/rebalance', {
        target_allocation: targetAllocation,
      });
      expect(result).toEqual(mockResponse.data);
    });

    it('should optimize portfolio', async () => {
      const mockResponse = {
        data: {
          optimization_method: 'mean_variance',
          recommended_allocation: {
            AAPL: 0.25,
            GOOGL: 0.20,
            MSFT: 0.20,
            TSLA: 0.15,
            CASH: 0.20,
          },
          expected_return: 0.12,
          expected_risk: 0.18,
        },
      };

      const mockPost = jest.fn().mockResolvedValue(mockResponse);
      (apiClient as any).client.post = mockPost;

      const result = await apiClient.optimizePortfolio('mean_variance', 0.5);

      expect(mockPost).toHaveBeenCalledWith('/api/v1/trading/portfolio/optimize', null, {
        params: {
          optimization_method: 'mean_variance',
          risk_tolerance: 0.5,
        },
      });
      expect(result).toEqual(mockResponse.data);
    });
  });

  describe('Market Data', () => {
    it('should get market data', async () => {
      const mockResponse = {
        data: {
          symbol: 'AAPL',
          timeframe: '1h',
          data: [
            {
              timestamp: '2023-12-01T10:00:00Z',
              open: 150.0,
              high: 155.0,
              low: 149.0,
              close: 154.0,
              volume: 1000000,
            },
          ],
          timestamp: '2023-12-01T10:00:00Z',
        },
      };

      const mockGet = jest.fn().mockResolvedValue(mockResponse);
      (apiClient as any).client.get = mockGet;

      const result = await apiClient.getMarketData('AAPL', '1h', 100);

      expect(mockGet).toHaveBeenCalledWith('/api/v1/trading/market-data/AAPL', {
        params: { timeframe: '1h', limit: 100 },
      });
      expect(result).toEqual(mockResponse.data);
    });
  });

  describe('Error Handling', () => {
    it('should handle network errors', async () => {
      const mockError = new Error('Network Error');
      const mockGet = jest.fn().mockRejectedValue(mockError);
      (apiClient as any).client.get = mockGet;

      await expect(apiClient.getCurrentUser()).rejects.toThrow('Network Error');
    });

    it('should handle 401 unauthorized errors', async () => {
      const mockError = {
        response: {
          status: 401,
          data: { detail: 'Token expired' },
        },
      };

      const mockGet = jest.fn().mockRejectedValue(mockError);
      (apiClient as any).client.get = mockGet;

      await expect(apiClient.getCurrentUser()).rejects.toEqual(mockError);
    });

    it('should handle 500 server errors', async () => {
      const mockError = {
        response: {
          status: 500,
          data: { detail: 'Internal server error' },
        },
      };

      const mockPost = jest.fn().mockRejectedValue(mockError);
      (apiClient as any).client.post = mockPost;

      await expect(apiClient.generateTradingSignal('AAPL')).rejects.toEqual(mockError);
    });
  });

  describe('Token Management', () => {
    it('should set auth token', () => {
      const token = 'test-token-123';
      apiClient.setAuthToken(token);
      
      expect((apiClient as any).authToken).toBe(token);
    });

    it('should clear auth token', () => {
      apiClient.setAuthToken('test-token');
      apiClient.setAuthToken(null);
      
      expect((apiClient as any).authToken).toBeNull();
    });
  });
});