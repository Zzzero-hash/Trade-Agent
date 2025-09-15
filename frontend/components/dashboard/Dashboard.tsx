'use client';

import { useState } from 'react';
import { useAuth } from '@/lib/auth-context';
import { useWebSocket } from '@/lib/websocket-context';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { PortfolioOverview } from './PortfolioOverview';
import { TradingSignals } from './TradingSignals';
import { MarketData } from './MarketData';
import { Settings } from './Settings';

type ActiveTab = 'overview' | 'signals' | 'portfolio' | 'market' | 'settings';

export function Dashboard() {
  const { user } = useAuth();
  const { isConnected } = useWebSocket();
  const [activeTab, setActiveTab] = useState<ActiveTab>('overview');

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return <PortfolioOverview />;
      case 'signals':
        return <TradingSignals />;
      case 'portfolio':
        return <PortfolioOverview />;
      case 'market':
        return <MarketData />;
      case 'settings':
        return <Settings />;
      default:
        return <PortfolioOverview />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="flex">
        <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
        
        <div className="flex-1 flex flex-col">
          <Header user={user} isConnected={isConnected} />
          
          <main className="flex-1 p-6">
            {renderContent()}
          </main>
        </div>
      </div>
    </div>
  );
}