"use client";

/**
 * Header component for the application.
 */

import * as React from 'react';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { useRefresh } from '@/hooks/use-stats';
import { ThemeToggle } from '@/components/theme';
import { RefreshCw } from 'lucide-react';

export function Header() {
  const { refreshData, loading, success, error, refreshStatus } = useRefresh();
  const pathname = usePathname();

  // Use a state to track client-side rendering
  const [isMounted, setIsMounted] = React.useState(false);

  // After component mounts, set isMounted to true
  React.useEffect(() => {
    setIsMounted(true);
  }, []);

  const isActive = (path: string) => {
    if (path === '/' && pathname !== '/') return false;
    return pathname === path || pathname.startsWith(`${path}/`);
  };

  // Use a simplified version for server-side rendering to avoid hydration errors
  const headerContent = (
    <header className="sticky top-0 z-50 w-full border-b border-border/50 bg-background/90 backdrop-blur-md supports-[backdrop-filter]:bg-background/60">
      <div className="container-centered flex h-16 items-center">
        <div className="mr-6 flex">
          <Link href="/" className="flex items-center space-x-2 group">
            <div className="flex items-center justify-center w-9 h-9 rounded-md bg-primary text-primary-foreground shadow-lg shadow-primary/20 transition-all duration-300 group-hover:shadow-primary/30">
              <span className="font-bold text-sm">2K</span>
            </div>
            <span className="font-bold text-xl gradient-text tracking-tight">2K Flash</span>
          </Link>
        </div>
        <div className="flex flex-1 items-center justify-between space-x-2 md:justify-end">
          <nav className="flex items-center space-x-2">
            <Link href="/" className={`nav-link ${isActive('/') ? 'active' : ''}`}>
              Home
            </Link>
            <Link href="/predictions" className={`nav-link ${isActive('/predictions') ? 'active' : ''}`}>
              Predictions
            </Link>
            <Link href="/live" className={`nav-link ${isActive('/live') ? 'active' : ''}`}>
              Live
            </Link>
            <Link href="/players" className={`nav-link ${isActive('/players') ? 'active' : ''}`}>
              Players
            </Link>
            <Link href="/scores" className={`nav-link ${isActive('/scores') ? 'active' : ''}`}>
              Scores
            </Link>
            <Link href="/history" className={`nav-link ${isActive('/history') ? 'active' : ''}`}>
              History
            </Link>
            <Link href="/stats" className={`nav-link ${isActive('/stats') ? 'active' : ''}`}>
              Stats
            </Link>
          </nav>          <div className="flex items-center space-x-3">
            <Button
              variant="outline"
              size="sm"
              onClick={() => refreshData()}
              disabled={loading}
              className="flex items-center gap-1.5 border-border/70 hover:border-primary/50 hover:bg-primary/5 transition-all duration-300"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              {loading ? (refreshStatus?.stage || 'Refreshing...') : 'Refresh'}
            </Button>
            
            {/* Progress status display */}
            {loading && refreshStatus && (
              <div className="flex flex-col items-start">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium text-blue-600 animate-fadeIn">
                    {refreshStatus.message}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {Math.round(refreshStatus.progress)}%
                  </span>
                </div>
                {/* Progress bar */}
                <div className="w-32 h-1 bg-gray-200 rounded-full mt-1">
                  <div 
                    className="h-1 bg-blue-500 rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${refreshStatus.progress}%` }}
                  />
                </div>
              </div>
            )}
            
            {success && (
              <span className="text-xs font-medium text-green-500 animate-fadeIn">
                Refresh successful!
              </span>
            )}
            {error && (
              <span className="text-xs font-medium text-red-500 animate-fadeIn">
                Refresh failed
              </span>
            )}
            <div className="border-l border-border/50 h-6 mx-1"></div>
            <ThemeToggle />
          </div>
        </div>
      </div>
    </header>
  );

  // Return the header content
  return headerContent;
}
