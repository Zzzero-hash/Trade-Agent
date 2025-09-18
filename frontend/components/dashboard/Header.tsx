'use client';

import { useAuth } from '@/lib/auth-context';
import { Bell, LogOut, User, Wifi, WifiOff } from 'lucide-react';
import { cn } from '@/lib/utils';

interface HeaderProps {
  user: any;
  isConnected: boolean;
}

export function Header({ user, isConnected }: HeaderProps) {
  const { logout } = useAuth();

  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Dashboard</h2>
            <p className="text-sm text-gray-600">
              Welcome back, {user?.username}
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Connection Status */}
            <div className="flex items-center space-x-2">
              {isConnected ? (
                <Wifi className="h-5 w-5 text-success-600" />
              ) : (
                <WifiOff className="h-5 w-5 text-danger-600" />
              )}
              <span className={cn(
                'text-sm font-medium',
                isConnected ? 'text-success-600' : 'text-danger-600'
              )}>
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            
            {/* User Role Badge */}
            <div className={cn(
              'badge',
              user?.role === 'premium' ? 'badge-primary' :
              user?.role === 'free' ? 'badge-success' :
              user?.role === 'trial' ? 'badge-warning' :
              'badge-secondary'
            )}>
              {user?.role?.toUpperCase()}
            </div>
            
            {/* Notifications */}
            <button className="p-2 text-gray-400 hover:text-gray-600 rounded-md">
              <Bell className="h-5 w-5" />
            </button>
            
            {/* User Menu */}
            <div className="flex items-center space-x-2">
              <button className="p-2 text-gray-400 hover:text-gray-600 rounded-md">
                <User className="h-5 w-5" />
              </button>
              
              <button
                onClick={logout}
                className="p-2 text-gray-400 hover:text-gray-600 rounded-md"
                title="Logout"
              >
                <LogOut className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}