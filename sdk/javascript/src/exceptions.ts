/**
 * Exception classes for the AI Trading Platform SDK
 */

export class TradingPlatformError extends Error {
  public statusCode?: number;
  public responseData?: Record<string, any>;

  constructor(message: string, statusCode?: number, responseData?: Record<string, any>) {
    super(message);
    this.name = 'TradingPlatformError';
    this.statusCode = statusCode;
    this.responseData = responseData;
  }
}

export class AuthenticationError extends TradingPlatformError {
  constructor(message: string, statusCode?: number, responseData?: Record<string, any>) {
    super(message, statusCode, responseData);
    this.name = 'AuthenticationError';
  }
}

export class AuthorizationError extends TradingPlatformError {
  constructor(message: string, statusCode?: number, responseData?: Record<string, any>) {
    super(message, statusCode, responseData);
    this.name = 'AuthorizationError';
  }
}

export class RateLimitError extends TradingPlatformError {
  public retryAfter?: number;

  constructor(message: string, retryAfter?: number, statusCode?: number, responseData?: Record<string, any>) {
    super(message, statusCode, responseData);
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

export class ValidationError extends TradingPlatformError {
  public validationErrors: any[];

  constructor(message: string, validationErrors: any[] = [], statusCode?: number, responseData?: Record<string, any>) {
    super(message, statusCode, responseData);
    this.name = 'ValidationError';
    this.validationErrors = validationErrors;
  }
}

export class APIError extends TradingPlatformError {
  constructor(message: string, statusCode?: number, responseData?: Record<string, any>) {
    super(message, statusCode, responseData);
    this.name = 'APIError';
  }
}

export class NetworkError extends TradingPlatformError {
  constructor(message: string, statusCode?: number, responseData?: Record<string, any>) {
    super(message, statusCode, responseData);
    this.name = 'NetworkError';
  }
}

export class TimeoutError extends TradingPlatformError {
  constructor(message: string, statusCode?: number, responseData?: Record<string, any>) {
    super(message, statusCode, responseData);
    this.name = 'TimeoutError';
  }
}

export class WebSocketError extends TradingPlatformError {
  constructor(message: string, statusCode?: number, responseData?: Record<string, any>) {
    super(message, statusCode, responseData);
    this.name = 'WebSocketError';
  }
}

export class PluginError extends TradingPlatformError {
  constructor(message: string, statusCode?: number, responseData?: Record<string, any>) {
    super(message, statusCode, responseData);
    this.name = 'PluginError';
  }
}