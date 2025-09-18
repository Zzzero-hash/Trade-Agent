/**
 * Type definitions for the AI Trading Platform SDK
 */

export enum TradingAction {
  BUY = 'BUY',
  SELL = 'SELL',
  HOLD = 'HOLD'
}

export enum OrderType {
  MARKET = 'MARKET',
  LIMIT = 'LIMIT',
  STOP = 'STOP',
  STOP_LIMIT = 'STOP_LIMIT'
}

export enum AssetType {
  STOCK = 'STOCK',
  ETF = 'ETF',
  OPTION = 'OPTION',
  FOREX = 'FOREX',
  CRYPTO = 'CRYPTO'
}

export interface TradingSignal {
  id: string;
  symbol: string;
  action: TradingAction;
  confidence: number;
  priceTarget?: number;
  stopLoss?: number;
  positionSize?: number;
  reasoning?: string;
  timestamp: string;
  expiresAt?: string;
  metadata?: Record<string, any>;
}

export interface Position {
  symbol: string;
  quantity: number;
  averageCost: number;
  currentPrice: number;
  marketValue: number;
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
  assetType: AssetType;
  lastUpdated: string;
}

export interface Portfolio {
  userId: string;
  positions: Record<string, Position>;
  cashBalance: number;
  totalValue: number;
  totalPnl: number;
  totalPnlPercent: number;
  lastUpdated: string;
}

export interface MarketData {
  symbol: string;
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  vwap?: number;
  metadata?: Record<string, any>;
}

export interface RiskMetrics {
  portfolioVar: number;
  portfolioCvar: number;
  sharpeRatio: number;
  sortinoRatio: number;
  maxDrawdown: number;
  beta?: number;
  alpha?: number;
  volatility: number;
  correlationMatrix?: Record<string, Record<string, number>>;
  timestamp: string;
}

export interface Alert {
  id: string;
  severity: string;
  title: string;
  message: string;
  timestamp: string;
  modelName?: string;
  metricName?: string;
  acknowledged: boolean;
  metadata?: Record<string, any>;
}

export interface ModelStatus {
  modelName: string;
  status: string;
  healthScore: number;
  lastPrediction?: string;
  predictionsToday: number;
  accuracy?: number;
  confidence?: number;
  version: string;
  deployedAt: string;
}

export interface PerformanceMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  aucRoc?: number;
  timestamp: string;
}

export interface WebSocketMessage {
  type: string;
  data: Record<string, any>;
  timestamp: string;
  userId?: string;
}

export interface PluginConfig {
  name: string;
  version: string;
  enabled: boolean;
  config: Record<string, any>;
  dependencies: string[];
}

export interface StrategyConfig {
  name: string;
  description?: string;
  parameters: Record<string, any>;
  riskLimits: Record<string, number>;
  enabled: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface BacktestResult {
  strategyName: string;
  startDate: string;
  endDate: string;
  totalReturn: number;
  annualizedReturn: number;
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalTrades: number;
  profitFactor: number;
  metadata?: Record<string, any>;
}

export interface AuthTokens {
  accessToken: string;
  refreshToken?: string;
  expiresIn: number;
  tokenType: string;
}

export interface UserInfo {
  userId: string;
  username: string;
  email: string;
  roles: string[];
  tier: string;
  expiresAt: string;
}

export interface ClientConfig {
  baseUrl: string;
  clientId?: string;
  clientSecret?: string;
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
  rateLimitRequests?: number;
  rateLimitWindow?: number;
}

export interface RequestOptions {
  timeout?: number;
  retries?: number;
  headers?: Record<string, string>;
}

export interface PaginationOptions {
  limit?: number;
  offset?: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

export interface FilterOptions {
  symbol?: string;
  action?: TradingAction;
  startDate?: string;
  endDate?: string;
  severity?: string;
  modelName?: string;
}

// Plugin interfaces
export interface BasePlugin {
  name: string;
  version: string;
  enabled: boolean;
  config: Record<string, any>;
  
  initialize(): Promise<void>;
  cleanup(): Promise<void>;
  getInfo(): PluginConfig;
}

export interface TradingStrategy extends BasePlugin {
  generateSignal(marketData: MarketData[], currentPortfolio: Portfolio): Promise<TradingSignal | null>;
  validateSignal(signal: TradingSignal): Promise<boolean>;
  getRequiredDataPeriod(): number;
}

export interface TechnicalIndicator extends BasePlugin {
  calculate(data: MarketData[]): Promise<number[]>;
  getRequiredColumns(): string[];
}

export interface RiskManager extends BasePlugin {
  assessRisk(signal: TradingSignal, portfolio: Portfolio): Promise<Record<string, any>>;
  adjustPositionSize(signal: TradingSignal, riskAssessment: Record<string, any>): Promise<number>;
}

export interface WebhookHandler extends BasePlugin {
  handleWebhook(eventType: string, data: Record<string, any>): Promise<Record<string, any>>;
  getSupportedEvents(): string[];
}

// Event types
export interface EventMap {
  'signal': TradingSignal;
  'portfolio_update': Portfolio;
  'market_data': MarketData;
  'alert': Alert;
  'connection_status': { connected: boolean; reconnectAttempts: number };
  'error': Error;
}

export type EventCallback<T = any> = (data: T) => void;

// Webhook event types
export interface WebhookEvent {
  id: string;
  type: string;
  timestamp: string;
  data: Record<string, any>;
  source: string;
  signature?: string;
}