'use client';

import { useState } from 'react';
import { useAuth } from '@/lib/auth-context';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { Brain, TrendingUp, Shield, Zap } from 'lucide-react';

export function LandingPage() {
  const { login, register } = useAuth();
  const [isLogin, setIsLogin] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    username: '',
    password: '',
  });
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      if (isLogin) {
        await login(formData.email, formData.password);
      } else {
        await register(formData.email, formData.username, formData.password);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 to-primary-100">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-primary-600 mr-2" />
              <h1 className="text-2xl font-bold text-gray-900">AI Trading Platform</h1>
            </div>
            <div className="flex space-x-4">
              <button
                onClick={() => setIsLogin(true)}
                className={`px-4 py-2 rounded-md ${
                  isLogin ? 'bg-primary-600 text-white' : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Login
              </button>
              <button
                onClick={() => setIsLogin(false)}
                className={`px-4 py-2 rounded-md ${
                  !isLogin ? 'bg-primary-600 text-white' : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Sign Up
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Hero Section */}
          <div>
            <h2 className="text-4xl font-bold text-gray-900 mb-6">
              Advanced AI-Powered Trading Platform
            </h2>
            <p className="text-xl text-gray-600 mb-8">
              Harness the power of CNN+LSTM models and reinforcement learning 
              to make intelligent trading decisions across stocks, forex, and crypto.
            </p>

            {/* Features */}
            <div className="space-y-6">
              <div className="flex items-start">
                <Brain className="h-6 w-6 text-primary-600 mt-1 mr-3" />
                <div>
                  <h3 className="font-semibold text-gray-900">AI-Driven Decisions</h3>
                  <p className="text-gray-600">
                    CNN+LSTM hybrid models for feature extraction and temporal pattern recognition
                  </p>
                </div>
              </div>
              
              <div className="flex items-start">
                <TrendingUp className="h-6 w-6 text-primary-600 mt-1 mr-3" />
                <div>
                  <h3 className="font-semibold text-gray-900">Multi-Asset Trading</h3>
                  <p className="text-gray-600">
                    Trade stocks, forex, and cryptocurrencies from a unified platform
                  </p>
                </div>
              </div>
              
              <div className="flex items-start">
                <Shield className="h-6 w-6 text-primary-600 mt-1 mr-3" />
                <div>
                  <h3 className="font-semibold text-gray-900">Risk Management</h3>
                  <p className="text-gray-600">
                    Advanced portfolio optimization and automated risk controls
                  </p>
                </div>
              </div>
              
              <div className="flex items-start">
                <Zap className="h-6 w-6 text-primary-600 mt-1 mr-3" />
                <div>
                  <h3 className="font-semibold text-gray-900">Real-Time Insights</h3>
                  <p className="text-gray-600">
                    Live market data and instant trading signal generation
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Auth Form */}
          <div className="bg-white rounded-lg shadow-lg p-8">
            <h3 className="text-2xl font-bold text-gray-900 mb-6">
              {isLogin ? 'Welcome Back' : 'Get Started Free'}
            </h3>
            
            {!isLogin && (
              <div className="bg-primary-50 border border-primary-200 rounded-md p-4 mb-6">
                <p className="text-sm text-primary-800">
                  🎉 Start with a 7-day free trial! Get 5 AI trading signals per day.
                </p>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="label">Email</label>
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleInputChange}
                  className="input"
                  required
                />
              </div>

              {!isLogin && (
                <div>
                  <label className="label">Username</label>
                  <input
                    type="text"
                    name="username"
                    value={formData.username}
                    onChange={handleInputChange}
                    className="input"
                    required
                  />
                </div>
              )}

              <div>
                <label className="label">Password</label>
                <input
                  type="password"
                  name="password"
                  value={formData.password}
                  onChange={handleInputChange}
                  className="input"
                  required
                />
              </div>

              {error && (
                <div className="bg-danger-50 border border-danger-200 rounded-md p-3">
                  <p className="text-sm text-danger-800">{error}</p>
                </div>
              )}

              <button
                type="submit"
                disabled={isLoading}
                className="w-full btn-primary"
              >
                {isLoading ? (
                  <LoadingSpinner size="sm" className="mr-2" />
                ) : null}
                {isLogin ? 'Sign In' : 'Create Account'}
              </button>
            </form>

            <div className="mt-6 text-center">
              <p className="text-sm text-gray-600">
                {isLogin ? "Don't have an account? " : "Already have an account? "}
                <button
                  onClick={() => setIsLogin(!isLogin)}
                  className="text-primary-600 hover:text-primary-500 font-medium"
                >
                  {isLogin ? 'Sign up' : 'Sign in'}
                </button>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}