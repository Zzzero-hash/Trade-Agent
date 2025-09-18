/**
 * AI Trading Platform JavaScript/TypeScript SDK
 * 
 * A comprehensive SDK for interacting with the AI Trading Platform API.
 * Supports both browser and Node.js environments with full TypeScript support.
 * 
 * Requirements: 11.3, 11.4, 11.5
 */

export { TradingPlatformClient } from './client';
export { AuthManager } from './auth';
export { WebSocketClient } from './websocket';
export { PluginManager } from './plugins';

// Export types and interfaces
export * from './types';
export * from './exceptions';

// Export utilities
export { EventEmitter } from './utils/events';
export { RetryManager } from './utils/retry';
export { RateLimiter } from './utils/rateLimit';

// Version info
export const VERSION = '1.0.0';