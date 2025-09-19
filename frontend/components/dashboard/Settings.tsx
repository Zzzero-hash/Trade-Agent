'use client';

import { useState } from 'react';
import { useAuth } from '@/lib/auth-context';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { formatDate } from '@/lib/utils';
import { User, CreditCard, Bell, Shield, Zap } from 'lucide-react';

export function Settings() {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState('profile');

  const { data: usageStats, isLoading } = useQuery(
    'usage-stats',
    () => apiClient.getUsageStats(),
    { refetchInterval: 60000 }
  );

  const tabs = [
    { id: 'profile', name: 'Profile', icon: User },
    { id: 'billing', name: 'Billing', icon: CreditCard },
    { id: 'notifications', name: 'Notifications', icon: Bell },
    { id: 'security', name: 'Security', icon: Shield },
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'profile':
        return <ProfileSettings user={user} />;
      case 'billing':
        return <BillingSettings user={user} usageStats={usageStats} />;
      case 'notifications':
        return <NotificationSettings />;
      case 'security':
        return <SecuritySettings />;
      default:
        return <ProfileSettings user={user} />;
    }
  };

  return (
    <div className="space-y-6">
      <div className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Settings</h2>
        
        {/* Tab Navigation */}
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`py-2 px-1 border-b-2 font-medium text-sm flex items-center ${
                    activeTab === tab.id
                      ? 'border-primary-500 text-primary-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="h-4 w-4 mr-2" />
                  {tab.name}
                </button>
              );
            })}
          </nav>
        </div>
        
        {/* Tab Content */}
        <div className="mt-6">
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <LoadingSpinner size="md" />
            </div>
          ) : (
            renderTabContent()
          )}
        </div>
      </div>
    </div>
  );
}

function ProfileSettings({ user }: { user: any }) {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-4">Profile Information</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="label">Email</label>
            <input
              type="email"
              value={user?.email || ''}
              className="input"
              disabled
            />
          </div>
          
          <div>
            <label className="label">Username</label>
            <input
              type="text"
              value={user?.username || ''}
              className="input"
              disabled
            />
          </div>
          
          <div>
            <label className="label">Account Type</label>
            <div className={`badge ${
              user?.role === 'premium' ? 'badge-primary' :
              user?.role === 'free' ? 'badge-success' :
              user?.role === 'trial' ? 'badge-warning' :
              'badge-secondary'
            }`}>
              {user?.role?.toUpperCase()}
            </div>
          </div>
          
          <div>
            <label className="label">Member Since</label>
            <p className="text-sm text-gray-600">
              {user?.created_at ? formatDate(user.created_at) : 'N/A'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function BillingSettings({ user, usageStats }: { user: any; usageStats: any }) {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-4">Usage & Billing</h3>
        
        {/* Usage Stats */}
        <div className="bg-gray-50 rounded-lg p-4 mb-6">
          <h4 className="font-medium text-gray-900 mb-3">Current Usage</h4>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-gray-600">Daily Signals Used</p>
              <div className="flex items-center">
                <p className="text-lg font-semibold text-gray-900">
                  {usageStats?.daily_signal_count || 0}
                </p>
                <span className="text-sm text-gray-500 ml-1">
                  / {usageStats?.daily_signal_limit || 0}
                </span>
              </div>
              
              {/* Progress bar */}
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div
                  className="bg-primary-600 h-2 rounded-full"
                  style={{
                    width: `${Math.min(
                      ((usageStats?.daily_signal_count || 0) / (usageStats?.daily_signal_limit || 1)) * 100,
                      100
                    )}%`,
                  }}
                />
              </div>
            </div>
            
            <div>
              <p className="text-sm text-gray-600">Account Status</p>
              <p className="text-lg font-semibold text-gray-900">
                {user?.role === 'trial' ? 'Trial' : 'Active'}
              </p>
            </div>
            
            {user?.role === 'trial' && usageStats?.trial_expires_at && (
              <div>
                <p className="text-sm text-gray-600">Trial Expires</p>
                <p className="text-lg font-semibold text-warning-600">
                  {formatDate(usageStats.trial_expires_at)}
                </p>
              </div>
            )}
          </div>
        </div>
        
        {/* Upgrade Options */}
        {user?.role !== 'premium' && (
          <div className="border border-primary-200 rounded-lg p-6">
            <div className="flex items-center mb-4">
              <Zap className="h-6 w-6 text-primary-600 mr-2" />
              <h4 className="text-lg font-medium text-gray-900">Upgrade to Premium</h4>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="border border-gray-200 rounded-lg p-4">
                <h5 className="font-medium text-gray-900 mb-2">Free Plan</h5>
                <p className="text-2xl font-bold text-gray-900 mb-2">$0<span className="text-sm font-normal text-gray-500">/month</span></p>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• 5 AI signals per day</li>
                  <li>• Basic portfolio tracking</li>
                  <li>• Community support</li>
                </ul>
              </div>
              
              <div className="border border-primary-200 rounded-lg p-4 bg-primary-50">
                <h5 className="font-medium text-gray-900 mb-2">Premium Plan</h5>
                <p className="text-2xl font-bold text-gray-900 mb-2">$29<span className="text-sm font-normal text-gray-500">/month</span></p>
                <ul className="text-sm text-gray-600 space-y-1 mb-4">
                  <li>• Unlimited AI signals</li>
                  <li>• Advanced portfolio optimization</li>
                  <li>• Real-time market data</li>
                  <li>• Priority support</li>
                  <li>• Advanced analytics</li>
                </ul>
                <button className="w-full btn-primary">
                  Upgrade Now
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function NotificationSettings() {
  const [notifications, setNotifications] = useState({
    signalAlerts: true,
    portfolioUpdates: true,
    marketNews: false,
    systemUpdates: true,
  });

  const handleToggle = (key: string) => {
    setNotifications(prev => ({
      ...prev,
      [key]: !prev[key as keyof typeof prev],
    }));
  };

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-4">Notification Preferences</h3>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900">Trading Signal Alerts</p>
              <p className="text-sm text-gray-500">Get notified when new signals are generated</p>
            </div>
            <button
              onClick={() => handleToggle('signalAlerts')}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                notifications.signalAlerts ? 'bg-primary-600' : 'bg-gray-200'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  notifications.signalAlerts ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
          
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900">Portfolio Updates</p>
              <p className="text-sm text-gray-500">Receive updates about portfolio performance</p>
            </div>
            <button
              onClick={() => handleToggle('portfolioUpdates')}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                notifications.portfolioUpdates ? 'bg-primary-600' : 'bg-gray-200'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  notifications.portfolioUpdates ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
          
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900">Market News</p>
              <p className="text-sm text-gray-500">Stay updated with relevant market news</p>
            </div>
            <button
              onClick={() => handleToggle('marketNews')}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                notifications.marketNews ? 'bg-primary-600' : 'bg-gray-200'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  notifications.marketNews ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
          
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900">System Updates</p>
              <p className="text-sm text-gray-500">Important system and security updates</p>
            </div>
            <button
              onClick={() => handleToggle('systemUpdates')}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                notifications.systemUpdates ? 'bg-primary-600' : 'bg-gray-200'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  notifications.systemUpdates ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function SecuritySettings() {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-4">Security Settings</h3>
        
        <div className="space-y-6">
          <div>
            <h4 className="font-medium text-gray-900 mb-2">Change Password</h4>
            <div className="space-y-4 max-w-md">
              <div>
                <label className="label">Current Password</label>
                <input type="password" className="input" />
              </div>
              <div>
                <label className="label">New Password</label>
                <input type="password" className="input" />
              </div>
              <div>
                <label className="label">Confirm New Password</label>
                <input type="password" className="input" />
              </div>
              <button className="btn-primary">Update Password</button>
            </div>
          </div>
          
          <div className="border-t border-gray-200 pt-6">
            <h4 className="font-medium text-gray-900 mb-2">Two-Factor Authentication</h4>
            <p className="text-sm text-gray-600 mb-4">
              Add an extra layer of security to your account
            </p>
            <button className="btn-secondary">Enable 2FA</button>
          </div>
          
          <div className="border-t border-gray-200 pt-6">
            <h4 className="font-medium text-gray-900 mb-2">API Keys</h4>
            <p className="text-sm text-gray-600 mb-4">
              Manage API keys for programmatic access
            </p>
            <button className="btn-secondary">Manage API Keys</button>
          </div>
        </div>
      </div>
    </div>
  );
}