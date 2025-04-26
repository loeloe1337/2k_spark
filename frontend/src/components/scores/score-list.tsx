/**
 * Score list component for displaying a list of score predictions.
 */

"use client";

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api/client';
import { ScoreCard } from "./score-card";

export function ScoreList() {
  const [predictions, setPredictions] = useState<any[]>([]);
  const [modelAccuracy, setModelAccuracy] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchScorePredictions = async () => {
      try {
        setLoading(true);

        const data = await apiClient.getScorePredictions();

        // Filter out matches that have already started
        const now = new Date();

        const upcomingMatches = data.predictions.filter(match => {
          // Parse the fixture start time
          const fixtureStart = new Date(match.fixtureStart);

          // Only include matches that haven't started yet
          const isUpcoming = fixtureStart > now;

          return isUpcoming;
        });

        // Sort by start time (earliest first)
        upcomingMatches.sort((a, b) => {
          return new Date(a.fixtureStart).getTime() - new Date(b.fixtureStart).getTime();
        });

        setPredictions(upcomingMatches);
        setModelAccuracy(data.summary.model_accuracy);
        setError(null);
      } catch (err) {
        setError('Failed to fetch score predictions. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchScorePredictions();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center py-12">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-muted-foreground">Loading score predictions...</p>
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
          <p className="text-muted-foreground">No upcoming score predictions available.</p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-6 p-4 bg-muted rounded-lg">
        <p className="text-sm text-muted-foreground">
          <strong>Model Accuracy:</strong> The score prediction model has an average error of {modelAccuracy.toFixed(1)} points.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {predictions.map((prediction) => (
          <ScoreCard key={prediction.fixtureId} prediction={prediction} />
        ))}
      </div>
    </div>
  );
}
