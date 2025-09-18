/**
 * Tests for WebSocket context and real-time data streaming.
 */

import React from 'react';
import { render, act, waitFor } from '@testing-library/react';
import { WebSocketProvider, useWebSocket, useMarketDataSubscription } from '@/lib/websocket-context';
import { AuthProvider } from '@/lib/auth-context';

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  constructor(public url: string) {
    // Simulate connection opening
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 10);
  }

  send(data: string) {
    // Mock send implementation
  }

  close(code?: number, reason?: string) {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent('close', { code: code || 1000, reason }));
    }
  }

  // Helper method to simulate receiving messages
  simulateMessage(data: any) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data: JSON.stringify(data) }));
    }
  }
}

// Mock the global WebSocket
(global as any).WebSocket = MockWebSocket;

// Mock auth context
const mockUser = {
  id: 'test-user-123',
  email: 'test@example.com',
  username: 'testuser',
  role: 'premium' as const,
  is_active: true,
  created_at: '2023-12-01T00:00:00Z',
  daily_signal_count: 0,
  daily_signal_limit: 999999,
};

const MockAuthProvider = ({ children }: { children: React.ReactNode }) => (
  <AuthProvider>
    {children}
  </AuthProvider>
);

// Override useAuth hook for testing
jest.mock('@/lib/auth-context', () => ({
  ...jest.requireActual('@/lib/auth-context'),
  useAuth: () => ({
    user: mockUser,
    token: 'test-token',
    isLoading: false,
    login: jest.fn(),
    register: jest.fn(),
    logout: jest.fn(),
    refreshToken: jest.fn(),
  }),
}));

describe('WebSocket Context', () => {
  let mockWebSocket: MockWebSocket;

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Intercept WebSocket constructor
    const originalWebSocket = (global as any).WebSocket;
    (global as any).WebSocket = jest.fn().mockImplementation((url: string) => {
      mockWebSocket = new originalWebSocket(url);
      return mockWebSocket;
    });
  });

  afterEach(() => {
    if (mockWebSocket) {
      mockWebSocket.close();
    }
  });

  it('should establish WebSocket connection when user is authenticated', async () => {
    const TestComponent = () => {
      const { isConnected } = useWebSocket();
      return <div data-testid="connection-status">{isConnected ? 'connected' : 'disconnected'}</div>;
    };

    const { getByTestId } = render(
      <MockAuthProvider>
        <WebSocketProvider>
          <TestComponent />
        </WebSocketProvider>
      </MockAuthProvider>
    );

    // Initially disconnected
    expect(getByTestId('connection-status')).toHaveTextContent('disconnected');

    // Wait for connection to establish
    await waitFor(() => {
      expect(getByTestId('connection-status')).toHaveTextContent('connected');
    });

    expect(global.WebSocket).toHaveBeenCalledWith(
      expect.stringContaining('/api/v1/trading/ws/test-user-123')
    );
  });

  it('should handle incoming WebSocket messages', async () => {
    const messages: any[] = [];
    
    const TestComponent = () => {
      const { subscribe, lastMessage } = useWebSocket();
      
      React.useEffect(() => {
        const unsubscribe = subscribe((message) => {
          messages.push(message);
        });
        return unsubscribe;
      }, [subscribe]);

      return <div data-testid="last-message">{JSON.stringify(lastMessage)}</div>;
    };

    render(
      <MockAuthProvider>
        <WebSocketProvider>
          <TestComponent />
        </WebSocketProvider>
      </MockAuthProvider>
    );

    // Wait for connection
    await waitFor(() => {
      expect(mockWebSocket.readyState).toBe(MockWebSocket.OPEN);
    });

    // Simulate incoming message
    const testMessage = {
      type: 'signal_generated',
      data: {
        symbol: 'AAPL',
        action: 'BUY',
        confidence: 0.85,
      },
      timestamp: '2023-12-01T10:00:00Z',
    };

    act(() => {
      mockWebSocket.simulateMessage(testMessage);
    });

    await waitFor(() => {
      expect(messages).toHaveLength(1);
      expect(messages[0]).toEqual(testMessage);
    });
  });

  it('should send messages through WebSocket', async () => {
    const sendSpy = jest.spyOn(MockWebSocket.prototype, 'send');
    
    const TestComponent = () => {
      const { sendMessage, isConnected } = useWebSocket();
      
      React.useEffect(() => {
        if (isConnected) {
          sendMessage({ type: 'subscribe_signals' });
        }
      }, [isConnected, sendMessage]);

      return <div>Test</div>;
    };

    render(
      <MockAuthProvider>
        <WebSocketProvider>
          <TestComponent />
        </WebSocketProvider>
      </MockAuthProvider>
    );

    await waitFor(() => {
      expect(sendSpy).toHaveBeenCalledWith(
        JSON.stringify({ type: 'subscribe_signals' })
      );
    });
  });

  it('should handle WebSocket disconnection and reconnection', async () => {
    const TestComponent = () => {
      const { isConnected } = useWebSocket();
      return <div data-testid="connection-status">{isConnected ? 'connected' : 'disconnected'}</div>;
    };

    const { getByTestId } = render(
      <MockAuthProvider>
        <WebSocketProvider>
          <TestComponent />
        </WebSocketProvider>
      </MockAuthProvider>
    );

    // Wait for initial connection
    await waitFor(() => {
      expect(getByTestId('connection-status')).toHaveTextContent('connected');
    });

    // Simulate disconnection
    act(() => {
      mockWebSocket.close(1006, 'Connection lost'); // Abnormal closure
    });

    await waitFor(() => {
      expect(getByTestId('connection-status')).toHaveTextContent('disconnected');
    });

    // Should attempt to reconnect (mocked WebSocket will auto-connect)
    await waitFor(() => {
      expect(getByTestId('connection-status')).toHaveTextContent('connected');
    }, { timeout: 2000 });
  });

  it('should handle JSON parsing errors gracefully', async () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
    
    const TestComponent = () => {
      const { lastMessage } = useWebSocket();
      return <div data-testid="last-message">{JSON.stringify(lastMessage)}</div>;
    };

    render(
      <MockAuthProvider>
        <WebSocketProvider>
          <TestComponent />
        </WebSocketProvider>
      </MockAuthProvider>
    );

    await waitFor(() => {
      expect(mockWebSocket.readyState).toBe(MockWebSocket.OPEN);
    });

    // Simulate invalid JSON message
    act(() => {
      if (mockWebSocket.onmessage) {
        mockWebSocket.onmessage(new MessageEvent('message', { data: 'invalid json' }));
      }
    });

    expect(consoleSpy).toHaveBeenCalledWith(
      'Failed to parse WebSocket message:',
      expect.any(Error)
    );

    consoleSpy.mockRestore();
  });
});

describe('Market Data Subscription Hook', () => {
  let mockWebSocket: MockWebSocket;

  beforeEach(() => {
    const originalWebSocket = (global as any).WebSocket;
    (global as any).WebSocket = jest.fn().mockImplementation((url: string) => {
      mockWebSocket = new originalWebSocket(url);
      return mockWebSocket;
    });
  });

  it('should subscribe to market data for specified symbols', async () => {
    const sendSpy = jest.spyOn(MockWebSocket.prototype, 'send');
    
    const TestComponent = () => {
      const marketData = useMarketDataSubscription(['AAPL', 'GOOGL']);
      return <div data-testid="market-data">{JSON.stringify(marketData)}</div>;
    };

    render(
      <MockAuthProvider>
        <WebSocketProvider>
          <TestComponent />
        </WebSocketProvider>
      </MockAuthProvider>
    );

    await waitFor(() => {
      expect(sendSpy).toHaveBeenCalledWith(
        JSON.stringify({
          type: 'subscribe_market_data',
          symbols: ['AAPL', 'GOOGL'],
        })
      );
    });
  });

  it('should update market data when receiving updates', async () => {
    const TestComponent = () => {
      const marketData = useMarketDataSubscription(['AAPL']);
      return <div data-testid="market-data">{JSON.stringify(marketData)}</div>;
    };

    const { getByTestId } = render(
      <MockAuthProvider>
        <WebSocketProvider>
          <TestComponent />
        </WebSocketProvider>
      </MockAuthProvider>
    );

    await waitFor(() => {
      expect(mockWebSocket.readyState).toBe(MockWebSocket.OPEN);
    });

    // Simulate market data update
    const marketUpdate = {
      type: 'market_data_update',
      data: {
        symbol: 'AAPL',
        price: 155.0,
        change: 2.5,
        change_percent: 0.016,
      },
    };

    act(() => {
      mockWebSocket.simulateMessage(marketUpdate);
    });

    await waitFor(() => {
      const marketDataElement = getByTestId('market-data');
      const marketData = JSON.parse(marketDataElement.textContent || '{}');
      expect(marketData.AAPL).toEqual(marketUpdate.data);
    });
  });
});

describe('WebSocket Error Handling', () => {
  it('should handle WebSocket connection errors', async () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
    
    const TestComponent = () => {
      const { isConnected } = useWebSocket();
      return <div data-testid="connection-status">{isConnected ? 'connected' : 'disconnected'}</div>;
    };

    render(
      <MockAuthProvider>
        <WebSocketProvider>
          <TestComponent />
        </WebSocketProvider>
      </MockAuthProvider>
    );

    // Simulate WebSocket error
    act(() => {
      if (mockWebSocket.onerror) {
        mockWebSocket.onerror(new Event('error'));
      }
    });

    expect(consoleSpy).toHaveBeenCalledWith('WebSocket error:', expect.any(Event));
    
    consoleSpy.mockRestore();
  });

  it('should not connect when user is not authenticated', () => {
    // Mock unauthenticated state
    jest.doMock('@/lib/auth-context', () => ({
      useAuth: () => ({
        user: null,
        token: null,
        isLoading: false,
        login: jest.fn(),
        register: jest.fn(),
        logout: jest.fn(),
        refreshToken: jest.fn(),
      }),
    }));

    const TestComponent = () => {
      const { isConnected } = useWebSocket();
      return <div data-testid="connection-status">{isConnected ? 'connected' : 'disconnected'}</div>;
    };

    const { getByTestId } = render(
      <MockAuthProvider>
        <WebSocketProvider>
          <TestComponent />
        </WebSocketProvider>
      </MockAuthProvider>
    );

    expect(getByTestId('connection-status')).toHaveTextContent('disconnected');
    expect(global.WebSocket).not.toHaveBeenCalled();
  });
});