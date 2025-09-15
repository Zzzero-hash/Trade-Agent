'use client';

import { cn } from '@/lib/utils';
import { 
  BarChart3, 
  TrendingUp, 
  Wallet, 
  Settings, 
  Brain,
  Home
} from 'lucide-react';

interface SidebarProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

const navigation = [
  { id: 'overview', name: 'Overview', icon: Home },
  { id: 'signals', name: 'Trading Signals', icon: Brain },
  { id: 'portfolio', name: 'Portfolio', icon: Wallet },
  { id: 'market', name: 'Market Data', icon: TrendingUp },
  { id: 'settings', name: 'Settings', icon: Settings },
];

export function Sidebar({ activeTab, onTabChange }: SidebarProps) {
  return (
    <div className="w-64 bg-white shadow-sm border-r border-gray-200 h-screen">
      <div className="p-6">
        <div className="flex items-center">
          <Brain className="h-8 w-8 text-primary-600 mr-2" />
          <h1 className="text-xl font-bold text-gray-900">AI Trading</h1>
        </div>
      </div>
      
      <nav className="px-3">
        <ul className="space-y-1">
          {navigation.map((item) => {
            const Icon = item.icon;
            return (
              <li key={item.id}>
                <button
                  onClick={() => onTabChange(item.id)}
                  className={cn(
                    'w-full flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors',
                    activeTab === item.id
                      ? 'bg-primary-100 text-primary-700'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  )}
                >
                  <Icon className="h-5 w-5 mr-3" />
                  {item.name}
                </button>
              </li>
            );
          })}
        </ul>
      </nav>
    </div>
  );
}