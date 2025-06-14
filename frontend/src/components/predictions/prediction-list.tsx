/**
 * Prediction list component for displaying a list of match predictions.
 */

"use client";

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api/client';
import { PredictionCard } from "./prediction-card";

// Type definition for prediction data
interface PredictionData {
  fixtureId: string;
  homePlayer: {
    id: string;
    name: string;
  };
  awayPlayer: {
    id: string;
    name: string;
  };
  homeTeam: {
    id: string;
    name: string;
  };
  awayTeam: {
    id: string;
    name: string;
  };
  fixtureStart: string;
  fetched_at?: string;
  prediction: {
    home_win_probability: number;
    away_win_probability: number;
    predicted_winner: "home" | "away";
    confidence: number;
  };
}

export function PredictionList() {
  const [predictions, setPredictions] = useState<PredictionData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        setLoading(true);

        // Make the API request with fetch directly
        const response = await fetch(`http://localhost:5000/api/predictions?timestamp=${Date.now()}`);

        if (!response.ok) {
          throw new Error(`HTTP error ${response.status}`);
        }

        const data = await response.json();
        let predictionsData = [];

        if (Array.isArray(data)) {
          predictionsData = data;
        } else if (data && typeof data === 'object' && Array.isArray(data.predictions)) {
          predictionsData = data.predictions;
        } else {
          setError('Unexpected API response format');
          setPredictions([]);
          return;
        }

        // Filter out matches that have already started
        const now = new Date();

        const upcomingMatches = predictionsData.filter((match: PredictionData) => {
          // Parse the fixture start time
          const fixtureStart = new Date(match.fixtureStart);

          // Only include matches that haven't started yet
          const isUpcoming = fixtureStart > now;

          return isUpcoming;
        });

        // Sort by start time (earliest first)
        upcomingMatches.sort((a: PredictionData, b: PredictionData) => {
          return new Date(a.fixtureStart).getTime() - new Date(b.fixtureStart).getTime();
        });

        setPredictions(upcomingMatches);

        setError(null);
      } catch (err) {
        setError(`Failed to fetch predictions. Please try again later.`);
      } finally {
        setLoading(false);
      }
    };

    fetchPredictions();
  }, []);



  if (loading) {
    return (
      <div className="flex justify-center items-center py-12">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-muted-foreground">Loading predictions...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex justify-center items-center py-12">
        <div className="text-center">
          <p className="text-red-500">{error}</p>
          <p className="mt-2 text-muted-foreground">Please try again later.</p>
        </div>
      </div>
    );
  }

  if (!predictions || predictions.length === 0) {
    return (
      <div className="flex justify-center items-center py-12">
        <div className="text-center">
          <p className="text-muted-foreground">No upcoming match predictions available.</p>
          <p className="mt-2 text-sm text-muted-foreground">
            All scheduled matches have already started or there are no matches scheduled.
          </p>
          <p className="mt-2 text-sm text-muted-foreground">
            Check back later for new upcoming matches or use the refresh button to update the data.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
      {predictions.map((prediction) => (
        <PredictionCard key={prediction.fixtureId} prediction={prediction} />
      ))}
    </div>
  );
}
