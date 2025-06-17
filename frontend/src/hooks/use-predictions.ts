"use client";

/**
 * Custom hook for fetching predictions.
 */

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api/client';
import { useRefreshContext } from '@/contexts/refresh-context';

/**
 * Hook for fetching match predictions.
 *
 * @returns Object with predictions data, loading state, and error
 */
export function usePredictions() {
  const [predictions, setPredictions] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const { refreshCounter } = useRefreshContext();

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        setLoading(true);
        console.log('Fetching predictions...');

        // Add a timestamp to the request to prevent caching
        const timestamp = Date.now();
        console.log(`Adding timestamp ${timestamp} to request to prevent caching`);
        const response = await apiClient.getPredictions();

        // Handle API response with error checking
        if (response.error) {
          console.error('API error fetching predictions:', response.error);
          setError(`Failed to fetch predictions: ${response.error}`);
          setPredictions([]);
          return;
        }

        // Extract data from API response
        const data = response.data;

        // Check if the data is in the expected format
        let predictionsData = [];

        if (Array.isArray(data)) {
          console.log('API Response (array format):', JSON.stringify(data).substring(0, 200) + '...');
          console.log(`Fetched ${data.length} predictions after refresh ${refreshCounter} (array format)`);

          // Debug: Log the first prediction
          if (data.length > 0) {
            console.log('First prediction:', JSON.stringify(data[0]).substring(0, 200) + '...');
          }

          predictionsData = data;
        } else if (data && typeof data === 'object' && 'predictions' in data && Array.isArray((data as any).predictions)) {
          // Handle object format with predictions array
          const typedData = data as { predictions: any[] };
          console.log('API Response (object format):', JSON.stringify(data).substring(0, 200) + '...');
          console.log(`Fetched ${typedData.predictions.length} predictions after refresh ${refreshCounter} (object format)`);

          // Debug: Log the first prediction
          if (typedData.predictions.length > 0) {
            console.log('First prediction:', JSON.stringify(typedData.predictions[0]).substring(0, 200) + '...');
          }

          predictionsData = typedData.predictions;
        } else {
          console.error('Unexpected API response format:', data);
          setPredictions([]);
          setError('Unexpected API response format');
          setLoading(false);
          return;
        }

        if (predictionsData.length === 0) {
          console.log('No predictions data received from API');
          setPredictions([]);
          setError(null);
          setLoading(false);
          return;
        }

        // Debug: Log all matches with their start times
        console.log('All matches:');
        predictionsData.forEach((match: any) => {
          const fixtureStart = new Date(match.fixtureStart);
          console.log(`Match ${match.fixtureId}: ${match.homePlayer.name} vs ${match.awayPlayer.name}, Start: ${fixtureStart.toISOString()}`);
        });

        // Show all matches for debugging
        const upcomingMatches = predictionsData;

        console.log(`Showing ${upcomingMatches.length} upcoming matches after filtering out ${predictionsData.length - upcomingMatches.length} matches that have already started`);

        // Sort by start time (earliest first)
        upcomingMatches.sort((a: any, b: any) => {
          return new Date(a.fixtureStart).getTime() - new Date(b.fixtureStart).getTime();
        });

        // Set the predictions state
        setPredictions(upcomingMatches);
        setError(null);
      } catch (err) {
        console.error('Error fetching predictions:', err);
        setError('Failed to fetch predictions. Please try again later.');
        setPredictions([]);
      } finally {
        setLoading(false);
      }
    };

    fetchPredictions();
  }, [refreshCounter]); // Re-fetch when refreshCounter changes

  return { predictions, loading, error };
}

/**
 * Hook for fetching score predictions.
 *
 * @returns Object with score predictions data, model accuracy, loading state, and error
 */
export function useScorePredictions() {
  const [predictions, setPredictions] = useState<any[]>([]);
  const [modelAccuracy, setModelAccuracy] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const { refreshCounter } = useRefreshContext();

  useEffect(() => {
    const fetchScorePredictions = async () => {
      try {
        setLoading(true);
        const response = await apiClient.getScorePredictions();

        // Handle API response with error checking
        if (response.error) {
          console.error('API error fetching score predictions:', response.error);
          setError(`Failed to fetch score predictions: ${response.error}`);
          setPredictions([]);
          return;
        }

        // Extract data from API response
        const data = response.data as any;
        
        let predictionsData: any[] = [];
        let accuracy = 0;

        if (data && typeof data === 'object') {
          if ('predictions' in data && Array.isArray(data.predictions)) {
            predictionsData = data.predictions;
          }
          if ('summary' in data && data.summary && 'model_accuracy' in data.summary) {
            accuracy = data.summary.model_accuracy;
          }
        } else if (Array.isArray(data)) {
          predictionsData = data;
        }

        console.log(`Fetched ${predictionsData.length} score predictions after refresh ${refreshCounter}`);

        // Show all matches for debugging
        const upcomingMatches = predictionsData;

        // Sort by start time (earliest first)
        upcomingMatches.sort((a: any, b: any) => {
          return new Date(a.fixtureStart).getTime() - new Date(b.fixtureStart).getTime();
        });

        console.log(`Showing ${upcomingMatches.length} upcoming score predictions after filtering`);

        setPredictions(upcomingMatches);
        setModelAccuracy(accuracy);
        setError(null);
      } catch (err) {
        console.error('Error fetching score predictions:', err);
        setError('Failed to fetch score predictions. Please try again later.');
        setPredictions([]);
      } finally {
        setLoading(false);
      }
    };

    fetchScorePredictions();
  }, [refreshCounter]); // Re-fetch when refreshCounter changes

  return { predictions, modelAccuracy, loading, error };
}

/**
 * Hook for fetching prediction history.
 *
 * @param player - Optional player filter
 * @param date - Optional date filter
 * @returns Object with prediction history data, loading state, and error
 */
export function usePredictionHistory(player?: string, date?: string) {
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const { refreshCounter } = useRefreshContext();

  useEffect(() => {
    const fetchPredictionHistory = async () => {
      try {
        setLoading(true);
        
        // Build parameters object
        const params: { date_from?: string; limit?: number } = {};
        if (date) {
          params.date_from = date;
        }
        // Add a reasonable limit to prevent huge responses
        params.limit = 1000;
        
        const response = await apiClient.getPredictionHistory(params);
        
        // Handle API response with error checking
        if (response.error) {
          console.error('API error fetching prediction history:', response.error);
          setError(`Failed to fetch prediction history: ${response.error}`);
          setHistory([]);
          return;
        }
        
        // Extract data from API response
        const data = response.data as any;
        let predictions: any[] = [];
        
        if (Array.isArray(data)) {
          predictions = data;
        } else if (data && typeof data === 'object' && 'predictions' in data && Array.isArray(data.predictions)) {
          predictions = data.predictions;
        } else {
          console.warn('Unexpected prediction history data format:', data);
          predictions = [];
        }
        
        // Filter by player if specified
        if (player) {
          predictions = predictions.filter((p: any) => 
            p.homePlayer?.name?.toLowerCase().includes(player.toLowerCase()) ||
            p.awayPlayer?.name?.toLowerCase().includes(player.toLowerCase())
          );
        }
        
        // Debug logging
        console.log(`Prediction history refreshed after refresh ${refreshCounter}`);
        console.log(`Total predictions received: ${predictions.length}`);
        
        // Check first few predictions for validation data
        const validatedPredictions = predictions.filter((p: any) => p.homeScore !== undefined && p.awayScore !== undefined);
        const unvalidatedPredictions = predictions.filter((p: any) => p.homeScore === undefined || p.awayScore === undefined);
        console.log(`Validated predictions: ${validatedPredictions.length}`);
        console.log(`Unvalidated predictions: ${unvalidatedPredictions.length}`);
        
        if (validatedPredictions.length > 0) {
          const sample = validatedPredictions[0];
          console.log('Sample validated prediction:', {
            fixtureId: sample.fixtureId,
            homeScore: sample.homeScore,
            awayScore: sample.awayScore,
            prediction_correct: sample.prediction_correct,
            homeScoreType: typeof sample.homeScore,
            awayScoreType: typeof sample.awayScore
          });
        }
        
        setHistory(predictions);
        setError(null);
      } catch (err) {
        console.error('Error fetching prediction history:', err);
        setError('Failed to fetch prediction history. Please try again later.');
        setHistory([]);
      } finally {
        setLoading(false);
      }
    };

    fetchPredictionHistory();
  }, [player, date, refreshCounter]); // Re-fetch when refreshCounter changes

  return { history, loading, error };
}

/**
 * Hook for fetching player statistics.
 *
 * @returns Object with player statistics data, loading state, and error
 */
export function usePlayerStats() {
  const [playerStats, setPlayerStats] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const { refreshCounter } = useRefreshContext();

  useEffect(() => {
    const fetchPlayerStats = async () => {
      try {
        setLoading(true);
        const response = await apiClient.getPlayerStats();
        
        // Handle API response with error checking
        if (response.error) {
          console.error('API error fetching player stats:', response.error);
          setError(`Failed to fetch player statistics: ${response.error}`);
          setPlayerStats([]);
          return;
        }
        
        // Extract data from API response and ensure it's an array
        const data = response.data;
        let statsArray: any[] = [];
        
        if (Array.isArray(data)) {
          statsArray = data;
        } else if (data && typeof data === 'object' && 'player_stats' in data && Array.isArray((data as any).player_stats)) {
          // Handle nested structure if API returns {player_stats: [...]}
          statsArray = (data as any).player_stats;
        } else if (data) {
          console.warn('Unexpected player stats data format:', data);
          statsArray = [];
        }
        
        setPlayerStats(statsArray);
        setError(null);
        console.log(`Player stats refreshed after refresh ${refreshCounter}. Loaded ${statsArray.length} players.`);
      } catch (err) {
        console.error('Error fetching player statistics:', err);
        setError('Failed to fetch player statistics. Please try again later.');
        setPlayerStats([]);
      } finally {
        setLoading(false);
      }
    };

    fetchPlayerStats();
  }, [refreshCounter]); // Re-fetch when refreshCounter changes

  return { playerStats, loading, error };
}

/**
 * Hook for fetching upcoming matches.
 *
 * @returns Object with upcoming matches data, loading state, and error
 */
export function useUpcomingMatches() {
  const [upcomingMatches, setUpcomingMatches] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const { refreshCounter } = useRefreshContext();

  useEffect(() => {
    const fetchUpcomingMatches = async () => {
      try {
        setLoading(true);
        const response = await apiClient.getUpcomingMatches();

        // Handle API response with error checking
        if (response.error) {
          console.error('API error fetching upcoming matches:', response.error);
          setError(`Failed to fetch upcoming matches: ${response.error}`);
          setUpcomingMatches([]);
          return;
        }

        // Extract data from API response
        const data = response.data;
        let matchesData: any[] = [];

        if (Array.isArray(data)) {
          matchesData = data;
        } else if (data && typeof data === 'object' && 'matches' in data && Array.isArray((data as any).matches)) {
          matchesData = (data as any).matches;
        } else {
          console.warn('Unexpected upcoming matches data format:', data);
          matchesData = [];
        }

        console.log(`Fetched ${matchesData.length} upcoming matches after refresh ${refreshCounter}`);

        // Show all matches for debugging
        const filteredMatches = matchesData;

        // Sort by start time (earliest first)
        filteredMatches.sort((a: any, b: any) => {
          return new Date(a.fixtureStart).getTime() - new Date(b.fixtureStart).getTime();
        });

        console.log(`Showing ${filteredMatches.length} upcoming matches after filtering`);

        setUpcomingMatches(filteredMatches);
        setError(null);
      } catch (err) {
        console.error('Error fetching upcoming matches:', err);
        setError('Failed to fetch upcoming matches. Please try again later.');
        setUpcomingMatches([]);
      } finally {
        setLoading(false);
      }
    };

    fetchUpcomingMatches();
  }, [refreshCounter]); // Re-fetch when refreshCounter changes

  return { upcomingMatches, loading, error };
}
