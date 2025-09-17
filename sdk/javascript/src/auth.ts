/**
 * Authentication manager for the AI Trading Platform SDK
 */

import axios, { AxiosInstance } from 'axios';
import * as jwt from 'jsonwebtoken';
import { AuthenticationError, AuthorizationError, NetworkError } from './exceptions';
import { AuthTokens, UserInfo } from './types';

export class AuthManager {
  private baseUrl: string;
  private clientId?: string;
  private clientSecret?: string;
  private accessToken?: string;
  private refreshToken?: string;
  private tokenExpiresAt?: Date;
  private httpClient: AxiosInstance;
  private refreshPromise?: Promise<AuthTokens>;

  constructor(baseUrl: string, clientId?: string, clientSecret?: string) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.clientId = clientId;
    this.clientSecret = clientSecret;
    
    this.httpClient = axios.create({
      baseURL: this.baseUrl,
      timeout: 30000,
    });
  }

  /**
   * Authenticate using username and password
   */
  async authenticateWithCredentials(username: string, password: string): Promise<AuthTokens> {
    try {
      const response = await this.httpClient.post('/api/v1/auth/login', {
        username,
        password,
        client_id: this.clientId,
      });

      const tokenData = response.data;
      this.storeTokens(tokenData);
      return tokenData;
    } catch (error: any) {
      if (error.response?.status === 401) {
        throw new AuthenticationError('Invalid credentials');
      }
      if (error.response) {
        throw new AuthenticationError(`Authentication failed: ${error.response.data?.message || error.response.statusText}`);
      }
      throw new NetworkError(`Network error during authentication: ${error.message}`);
    }
  }

  /**
   * Authenticate using OAuth2 authorization code
   */
  async authenticateWithOAuth(authorizationCode: string, redirectUri: string): Promise<AuthTokens> {
    try {
      const response = await this.httpClient.post('/api/v1/auth/oauth/token', {
        grant_type: 'authorization_code',
        code: authorizationCode,
        redirect_uri: redirectUri,
        client_id: this.clientId,
        client_secret: this.clientSecret,
      });

      const tokenData = response.data;
      this.storeTokens(tokenData);
      return tokenData;
    } catch (error: any) {
      if (error.response?.status === 401) {
        throw new AuthenticationError('Invalid authorization code');
      }
      if (error.response) {
        throw new AuthenticationError(`OAuth authentication failed: ${error.response.data?.message || error.response.statusText}`);
      }
      throw new NetworkError(`Network error during OAuth authentication: ${error.message}`);
    }
  }

  /**
   * Authenticate using API key
   */
  async authenticateWithApiKey(apiKey: string): Promise<AuthTokens> {
    try {
      const response = await this.httpClient.post('/api/v1/auth/api-key', {}, {
        headers: {
          'X-API-Key': apiKey,
        },
      });

      const tokenData = response.data;
      this.storeTokens(tokenData);
      return tokenData;
    } catch (error: any) {
      if (error.response?.status === 401) {
        throw new AuthenticationError('Invalid API key');
      }
      if (error.response) {
        throw new AuthenticationError(`API key authentication failed: ${error.response.data?.message || error.response.statusText}`);
      }
      throw new NetworkError(`Network error during API key authentication: ${error.message}`);
    }
  }

  /**
   * Store authentication tokens
   */
  private storeTokens(tokenData: AuthTokens): void {
    this.accessToken = tokenData.accessToken;
    this.refreshToken = tokenData.refreshToken;
    
    // Calculate expiration time
    const expiresIn = tokenData.expiresIn || 3600; // Default 1 hour
    this.tokenExpiresAt = new Date(Date.now() + expiresIn * 1000);
  }

  /**
   * Get a valid access token, refreshing if necessary
   */
  async getValidToken(): Promise<string> {
    if (!this.accessToken) {
      throw new AuthenticationError('No access token available. Please authenticate first.');
    }

    // Check if token is expired or will expire soon (5 minutes buffer)
    const bufferTime = 5 * 60 * 1000; // 5 minutes in milliseconds
    if (this.tokenExpiresAt && Date.now() + bufferTime >= this.tokenExpiresAt.getTime()) {
      await this.refreshAccessToken();
    }

    return this.accessToken;
  }

  /**
   * Refresh the access token using refresh token
   */
  private async refreshAccessToken(): Promise<void> {
    // Prevent multiple concurrent refresh attempts
    if (this.refreshPromise) {
      await this.refreshPromise;
      return;
    }

    this.refreshPromise = this.performTokenRefresh();
    
    try {
      await this.refreshPromise;
    } finally {
      this.refreshPromise = undefined;
    }
  }

  private async performTokenRefresh(): Promise<AuthTokens> {
    // Double-check if token still needs refreshing
    const bufferTime = 5 * 60 * 1000;
    if (this.tokenExpiresAt && Date.now() + bufferTime < this.tokenExpiresAt.getTime()) {
      return {
        accessToken: this.accessToken!,
        refreshToken: this.refreshToken,
        expiresIn: Math.floor((this.tokenExpiresAt.getTime() - Date.now()) / 1000),
        tokenType: 'Bearer',
      };
    }

    if (!this.refreshToken) {
      throw new AuthenticationError('No refresh token available. Please re-authenticate.');
    }

    try {
      const response = await this.httpClient.post('/api/v1/auth/refresh', {
        refresh_token: this.refreshToken,
        client_id: this.clientId,
      });

      const tokenData = response.data;
      this.storeTokens(tokenData);
      return tokenData;
    } catch (error: any) {
      if (error.response?.status === 401) {
        // Refresh token is invalid, clear all tokens
        this.clearTokens();
        throw new AuthenticationError('Refresh token expired. Please re-authenticate.');
      }
      if (error.response) {
        throw new AuthenticationError(`Token refresh failed: ${error.response.data?.message || error.response.statusText}`);
      }
      throw new NetworkError(`Network error during token refresh: ${error.message}`);
    }
  }

  /**
   * Logout and invalidate tokens
   */
  async logout(): Promise<void> {
    if (this.accessToken) {
      try {
        await this.httpClient.post('/api/v1/auth/logout', {}, {
          headers: {
            Authorization: `Bearer ${this.accessToken}`,
          },
        });
      } catch (error) {
        // Ignore network errors during logout
      }
    }

    this.clearTokens();
  }

  /**
   * Clear stored tokens
   */
  private clearTokens(): void {
    this.accessToken = undefined;
    this.refreshToken = undefined;
    this.tokenExpiresAt = undefined;
  }

  /**
   * Check if user is currently authenticated
   */
  isAuthenticated(): boolean {
    return !!(
      this.accessToken &&
      this.tokenExpiresAt &&
      Date.now() < this.tokenExpiresAt.getTime()
    );
  }

  /**
   * Extract user information from access token
   */
  getUserInfo(): UserInfo | null {
    if (!this.accessToken) {
      return null;
    }

    try {
      // Decode JWT token (without verification for user info)
      const payload = jwt.decode(this.accessToken) as any;
      
      if (!payload) {
        return null;
      }

      return {
        userId: payload.sub,
        username: payload.username,
        email: payload.email,
        roles: payload.roles || [],
        tier: payload.tier || 'free',
        expiresAt: new Date(payload.exp * 1000).toISOString(),
      };
    } catch (error) {
      return null;
    }
  }

  /**
   * Get authentication headers for API requests
   */
  async getAuthHeaders(): Promise<Record<string, string>> {
    const token = await this.getValidToken();
    return {
      Authorization: `Bearer ${token}`,
    };
  }

  /**
   * Generate OAuth2 authorization URL
   */
  getOAuthAuthorizationUrl(
    redirectUri: string,
    state?: string,
    scopes?: string[]
  ): string {
    const params = new URLSearchParams({
      response_type: 'code',
      client_id: this.clientId || '',
      redirect_uri: redirectUri,
    });

    if (state) {
      params.append('state', state);
    }

    if (scopes && scopes.length > 0) {
      params.append('scope', scopes.join(' '));
    }

    return `${this.baseUrl}/api/v1/auth/oauth/authorize?${params.toString()}`;
  }

  /**
   * Get current access token (without validation)
   */
  getCurrentToken(): string | undefined {
    return this.accessToken;
  }

  /**
   * Get token expiration time
   */
  getTokenExpiresAt(): Date | undefined {
    return this.tokenExpiresAt;
  }
}