/**
 * Live match list component for displaying a list of live matches.
 */

"use client";

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api/client';
import { LiveMatchCard } from "./live-match-card";

// Type definitions for the API responses
interface Player {
  name: string;
  id: string;
}

interface PredictionData {
  fixtureId: string;
  prediction?: {
    home_win_probability: number;
    away_win_probability: number;
  };
  score_prediction?: {
    home_score: number;
    away_score: number;
    total_score: number;
    score_diff: number;
  };
}

interface UpcomingMatchData {
  id: string;
  homePlayer: Player;
  awayPlayer: Player;
  fixtureStart: string;
  [key: string]: any;
}

interface ScorePredictionsResponse {
  predictions?: Array<{
    fixtureId: string;
    score_prediction?: {
      home_score: number;
      away_score: number;
      total_score: number;
      score_diff: number;
    };
  }>;
}

interface LiveMatch extends UpcomingMatchData {
  fixtureId: string;
  homeProbability: number;
  awayProbability: number;
  homeScorePrediction: string;
  awayScorePrediction: string;
  totalScore: string;
  scoreDiff: string;
  rawHomeScore: number | null;
  rawAwayScore: number | null;
  rawTotalScore: number | null;
  rawScoreDiff: number | null;
}

export function LiveMatchList() {
  const [liveMatches, setLiveMatches] = useState<LiveMatch[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchLiveMatches = async () => {
      try {
        setLoading(true);

        // Fetch both predictions and upcoming matches
        const [predictionsData, upcomingMatchesData, scorePredictionsData] = await Promise.all([
          apiClient.getPredictions(),
          apiClient.getUpcomingMatches(),
          apiClient.getScorePredictions()
        ]) as [PredictionData[], UpcomingMatchData[], ScorePredictionsResponse];

        console.log('Raw predictions data sample:', predictionsData.length > 0 ? predictionsData[0] : 'No predictions');
        console.log('Raw score predictions data:',
          scorePredictionsData && scorePredictionsData.predictions ?
          scorePredictionsData.predictions[0] : 'No score predictions');

        // Create a map of fixture IDs to score predictions
        const scorePredictionsMap = new Map();
        if (scorePredictionsData && scorePredictionsData.predictions) {
          scorePredictionsData.predictions.forEach((prediction: { fixtureId: string; score_prediction?: { home_score: number; away_score: number; total_score: number; score_diff: number; }; }) => {
            // Log each prediction to see its structure
            console.log(`Prediction for fixture ${prediction.fixtureId}:`, prediction);

            scorePredictionsMap.set(prediction.fixtureId, {
              homeScorePrediction: prediction.score_prediction ? prediction.score_prediction.home_score : null,
              awayScorePrediction: prediction.score_prediction ? prediction.score_prediction.away_score : null,
              totalScore: prediction.score_prediction ? prediction.score_prediction.total_score : null,
              scoreDiff: prediction.score_prediction ? Math.abs(prediction.score_prediction.score_diff) : null
            });
          });
        }

        // Merge predictions with upcoming matches data
        const mergedData = upcomingMatchesData.map((match: UpcomingMatchData) => {
          // Find corresponding prediction
          const prediction = predictionsData.find((p: PredictionData) => p.fixtureId === match.id);

          // Find corresponding score prediction
          const scorePrediction = scorePredictionsMap.get(match.id);

          console.log(`Processing match ${match.id}:`, {
            prediction: prediction ? 'found' : 'not found',
            scorePrediction: scorePrediction ? 'found' : 'not found'
          });

          // Extract the probability values from the prediction
          let homeProbability = 0.5;
          let awayProbability = 0.5;

          if (prediction && prediction.prediction) {
            homeProbability = prediction.prediction.home_win_probability;
            awayProbability = prediction.prediction.away_win_probability;
          }

          // Extract score predictions
          let homeScorePrediction = null;
          let awayScorePrediction = null;
          let totalScore = null;
          let scoreDiff = null;

          // Try to get score predictions from the prediction object first
          if (prediction && prediction.score_prediction) {
            console.log(`Found score prediction in prediction object for match ${match.id}:`, prediction.score_prediction);
            homeScorePrediction = prediction.score_prediction.home_score;
            awayScorePrediction = prediction.score_prediction.away_score;
            totalScore = prediction.score_prediction.total_score;
            scoreDiff = Math.abs(prediction.score_prediction.score_diff);
          }
          // If not found, try the score prediction map
          else if (scorePrediction) {
            console.log(`Found score prediction in map for match ${match.id}:`, scorePrediction);
            homeScorePrediction = scorePrediction.homeScorePrediction;
            awayScorePrediction = scorePrediction.awayScorePrediction;
            totalScore = scorePrediction.totalScore;
            scoreDiff = scorePrediction.scoreDiff;
          }

          // Convert to strings for display
          const homeScorePredictionStr = homeScorePrediction !== null ? String(homeScorePrediction) : "N/A";
          const awayScorePredictionStr = awayScorePrediction !== null ? String(awayScorePrediction) : "N/A";
          const totalScoreStr = totalScore !== null ? String(totalScore) : "N/A";
          const scoreDiffStr = scoreDiff !== null ? String(scoreDiff) : "N/A";

          return {
            ...match,
            fixtureId: match.id,
            homeProbability: homeProbability,
            awayProbability: awayProbability,
            homeScorePrediction: homeScorePredictionStr,
            awayScorePrediction: awayScorePredictionStr,
            totalScore: totalScoreStr,
            scoreDiff: scoreDiffStr,
            // Keep the raw values for debugging
            rawHomeScore: homeScorePrediction,
            rawAwayScore: awayScorePrediction,
            rawTotalScore: totalScore,
            rawScoreDiff: scoreDiff
          };
        });

        // Filter for live matches (started within the last 30 minutes)
        const now = new Date();
        const thirtyMinutesAgo = new Date(now.getTime() - 30 * 60 * 1000);

        const filteredMatches = mergedData.filter((match: LiveMatch) => {
          // Parse the fixture start time
          const fixtureStart = new Date(match.fixtureStart);

          // Only include matches that have started within the last 30 minutes
          return fixtureStart <= now && fixtureStart >= thirtyMinutesAgo;
        });

        // Debug: Log the first match data to see what we have
        if (filteredMatches.length > 0) {
          console.log('First live match data:', JSON.stringify(filteredMatches[0], null, 2));
        } else {
          console.log('No live matches found');
        }

        // Sort by most recently started first
        filteredMatches.sort((a: LiveMatch, b: LiveMatch) => {
          return new Date(b.fixtureStart).getTime() - new Date(a.fixtureStart).getTime();
        });

        setLiveMatches(filteredMatches);
        setError(null);
      } catch (err) {
        setError('Failed to fetch live matches. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchLiveMatches();

    // Set up an interval to refresh the live matches every minute
    const intervalId = setInterval(fetchLiveMatches, 60000);

    // Clean up the interval when the component unmounts
    return () => clearInterval(intervalId);
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center py-12">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-muted-foreground">Loading live matches...</p>
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

  if (!liveMatches || liveMatches.length === 0) {
    return (
      <div className="flex justify-center items-center py-12">
        <div className="text-center">
          <p className="text-muted-foreground">No live matches available right now.</p>
          <p className="mt-2 text-sm text-muted-foreground">
            Check back later for live matches or view upcoming matches.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
      {liveMatches.map((match: LiveMatch) => (
        <LiveMatchCard key={match.fixtureId} match={match} />
      ))}
    </div>
  );
}
