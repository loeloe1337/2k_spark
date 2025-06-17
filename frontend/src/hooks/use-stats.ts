"use client";

/**
 * Custom hook for fetching prediction statistics.
 */

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api/client';
import { useRefreshContext } from '@/contexts/refresh-context';

// Type definitions for API responses
interface StatsResponse {
  total_matches: number;
  home_wins_predicted: number;
  away_wins_predicted: number;
  avg_confidence: number;
  model_accuracy: number;
  last_updated: string;
}

interface RefreshResponse {
  status: string;
  message?: string;
}

interface RefreshStatusResponse {
  status: 'idle' | 'running' | 'completed' | 'failed';
  stage: string;
  progress: number;
  message: string;
  start_time?: string;
  end_time?: string;
  error?: string;
  duration_seconds?: number;
}

/**
 * Hook for fetching prediction statistics.
 *
 * @returns Object with statistics data, loading state, and error
 */
export function useStats() {
  const [stats, setStats] = useState<{
    total_matches: number,
    home_wins_predicted: number,
    away_wins_predicted: number,
    avg_confidence: number,
    model_accuracy: number,
    last_updated: string
  } | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const { refreshCounter } = useRefreshContext();
  useEffect(() => {
    const fetchStats = async () => {
      try {
        setLoading(true);
        const response = await apiClient.getStats();
        
        if (response.error) {
          setError(response.error);
          return;
        }
        
        const data = response.data as StatsResponse;
        setStats(data);
        setError(null);
        console.log(`Stats refreshed after refresh ${refreshCounter}`);
      } catch (err) {
        console.error('Error fetching stats:', err);
        setError('Failed to fetch statistics. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, [refreshCounter]); // Re-fetch when refreshCounter changes

  return { stats, loading, error };
}

/**
 * Hook for triggering data refresh.
 *
 * @returns Object with refresh function, loading state, and error
 */
export function useRefresh() {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<boolean>(false);
  const [refreshStatus, setRefreshStatus] = useState<RefreshStatusResponse | null>(null);
  const { triggerRefresh } = useRefreshContext();

  // Poll for refresh status
  const pollRefreshStatus = async () => {
    try {
      const response = await apiClient.getRefreshStatus();
      if (response.error) {
        console.error('Error getting refresh status:', response.error);
        return;
      }
      
      const status = response.data as RefreshStatusResponse;
      setRefreshStatus(status);
      
      // Check if refresh is completed
      if (status.status === 'completed') {
        setSuccess(true);
        setLoading(false);
        setError(null);
        
        // Trigger component refresh after completion
        setTimeout(() => {
          triggerRefresh();
          // Clear success message after 3 seconds
          setTimeout(() => {
            setSuccess(false);
            setRefreshStatus(null);
          }, 3000);
        }, 1000);
        
        return false; // Stop polling
      } else if (status.status === 'failed') {
        setError(status.error || 'Refresh failed');
        setLoading(false);
        setSuccess(false);
        setRefreshStatus(null);
        return false; // Stop polling
      }
      
      return true; // Continue polling
    } catch (err) {
      console.error('Error polling refresh status:', err);
      return true; // Continue polling despite error
    }
  };

  const refreshData = async () => {
    try {
      setLoading(true);
      setSuccess(false);
      setError(null);
      setRefreshStatus(null);

      const response = await apiClient.refreshData();

      // Check if the API call was successful (no error in ApiResponse)
      if (response.error) {
        setError(response.error);
        setLoading(false);
        return;
      }

      // Check the backend response data
      const backendResponse = response.data as RefreshResponse;
      if (backendResponse && backendResponse.status === 'success') {
        // Start polling for status updates
        const pollInterval = setInterval(async () => {
          const shouldContinue = await pollRefreshStatus();
          if (!shouldContinue) {
            clearInterval(pollInterval);
          }
        }, 1000); // Poll every second

        // Set a maximum polling time of 5 minutes
        setTimeout(() => {
          clearInterval(pollInterval);
          if (loading) {
            setLoading(false);
            setError('Refresh timed out. Please try again.');
            setRefreshStatus(null);
          }
        }, 300000); // 5 minutes
      } else {
        setError(backendResponse?.message || 'Refresh failed');
        setLoading(false);
      }
    } catch (err) {
      console.error('Error refreshing data:', err);
      setError('Failed to refresh data. Please try again later.');
      setSuccess(false);
      setLoading(false);
    }
  };

  return { refreshData, loading, error, success, refreshStatus };
}
