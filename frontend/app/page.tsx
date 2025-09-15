'use client';

import { useAuth } from '@/lib/auth-context';
import { Dashboard } from '@/components/dashboard/Dashboard';
import { LandingPage } from '@/components/landing/LandingPage';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';

export default function HomePage() {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return user ? <Dashboard /> : <LandingPage />;
}