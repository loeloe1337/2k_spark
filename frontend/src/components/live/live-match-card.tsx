/**
 * Live match card component for displaying a live match with predictions.
 */

import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { formatDistanceToNow } from "date-fns";

interface LiveMatchCardProps {
  match: {
    fixtureId: string;
    homePlayer: { name: string; id: string };
    awayPlayer: { name: string; id: string };
    fixtureStart: string;
    homeProbability?: number;
    awayProbability?: number;
    homeWinProbability?: number;
    awayWinProbability?: number;
    homeScorePrediction?: string;
    awayScorePrediction?: string;
    totalScore?: string;
    scoreDiff?: string;
    rawHomeScore?: number | null;
    rawAwayScore?: number | null;
    rawTotalScore?: number | null;
    rawScoreDiff?: number | null;
    [key: string]: any;
  };
}

export function LiveMatchCard({ match }: LiveMatchCardProps) {
  // Format the match start time
  const matchStartTime = new Date(match.fixtureStart);
  const timeAgo = formatDistanceToNow(matchStartTime, { addSuffix: true });

  // Calculate elapsed time in minutes
  const now = new Date();
  const elapsedMinutes = Math.floor((now.getTime() - matchStartTime.getTime()) / (1000 * 60));

  // Determine the predicted winner
  const homeWinProbability = match.homeProbability || match.homeWinProbability || 0;
  const awayWinProbability = match.awayProbability || match.awayWinProbability || 0;

  const predictedWinner = homeWinProbability > awayWinProbability ? match.homePlayer : match.awayPlayer;
  const winnerConfidence = Math.max(homeWinProbability, awayWinProbability) * 100;

  // Get score predictions if available
  const homeScorePrediction = match.homeScorePrediction || "N/A";
  const awayScorePrediction = match.awayScorePrediction || "N/A";
  const totalScore = match.totalScore || "N/A";
  const scoreDiff = match.scoreDiff || "N/A";

  // Debug the score predictions
  console.log('Match score predictions:', {
    matchId: match.fixtureId,
    homeScorePrediction,
    awayScorePrediction,
    totalScore,
    scoreDiff,
    rawHomeScore: match.rawHomeScore,
    rawAwayScore: match.rawAwayScore,
    rawTotalScore: match.rawTotalScore,
    rawScoreDiff: match.rawScoreDiff
  });

  return (
    <Card className="overflow-hidden card-highlight card-hover">
      <CardContent className="p-6 pt-5">
        {/* Live Badge and Time Info */}
        <div className="flex justify-between items-center mb-6">
          <Badge variant="outline" className="bg-red-500 text-white font-medium px-3 py-1 shadow-sm shadow-red-500/20 animate-pulse">
            LIVE
          </Badge>
          <div className="text-xs font-medium bg-muted/80 px-3 py-1.5 rounded-full border border-border/30">
            Started {timeAgo} ({elapsedMinutes} min)
          </div>
        </div>

        {/* Players */}
        <div className="flex justify-between items-center mb-8">
          {/* Home Player */}
          <div className="flex flex-col items-center text-center space-y-3">
            <Avatar className="h-16 w-16 border-2 border-primary/20 shadow-lg shadow-primary/10">
              <AvatarImage src={`/avatars/${match.homePlayer.id}.png`} alt={match.homePlayer.name} />
              <AvatarFallback className="bg-primary/10 text-primary font-bold">{match.homePlayer.name.charAt(0)}</AvatarFallback>
            </Avatar>
            <div>
              <p className="font-semibold text-base">{match.homePlayer.name}</p>
              <p className="text-sm text-muted-foreground">Home</p>
            </div>
          </div>

          {/* VS */}
          <div className="flex flex-col items-center px-2">
            <div className="relative">
              <p className="text-xl font-bold bg-gradient-to-r from-primary/80 to-blue-500/80 bg-clip-text text-transparent">VS</p>
              <div className="absolute -inset-3 bg-gradient-to-r from-primary/5 to-blue-500/5 rounded-full blur-md -z-10"></div>
            </div>
            <p className="text-xs text-muted-foreground mt-1">Match #{match.fixtureId}</p>
          </div>

          {/* Away Player */}
          <div className="flex flex-col items-center text-center space-y-3">
            <Avatar className="h-16 w-16 border-2 border-primary/20 shadow-lg shadow-primary/10">
              <AvatarImage src={`/avatars/${match.awayPlayer.id}.png`} alt={match.awayPlayer.name} />
              <AvatarFallback className="bg-primary/10 text-primary font-bold">{match.awayPlayer.name.charAt(0)}</AvatarFallback>
            </Avatar>
            <div>
              <p className="font-semibold text-base">{match.awayPlayer.name}</p>
              <p className="text-sm text-muted-foreground">Away</p>
            </div>
          </div>
        </div>

        {/* Predictions */}
        <div className="space-y-5">
          {/* Winner Prediction */}
          <div className="bg-muted/70 p-4 rounded-lg border border-border/30 shadow-sm">
            <p className="text-sm font-medium mb-2 text-primary/90">Predicted Winner</p>
            <div className="flex justify-between items-center">
              <div className="flex items-center space-x-3">
                <Avatar className="h-8 w-8 border border-primary/20">
                  <AvatarImage src={`/avatars/${predictedWinner.id}.png`} alt={predictedWinner.name} />
                  <AvatarFallback className="bg-primary/10 text-primary">{predictedWinner.name.charAt(0)}</AvatarFallback>
                </Avatar>
                <span className="font-medium">{predictedWinner.name}</span>
              </div>
              <Badge variant="outline" className="bg-primary/10 border-primary/20 px-3 py-1 font-medium">
                {winnerConfidence.toFixed(0)}% confidence
              </Badge>
            </div>
          </div>

          {/* Score Prediction */}
          <div className="bg-muted/70 p-4 rounded-lg border border-border/30 shadow-sm">
            <p className="text-sm font-medium mb-3 text-primary/90">Predicted Score</p>
            <div className="flex justify-center items-center mb-4">
              <div className="flex items-center space-x-6">
                <div className="text-center">
                  <p className="text-2xl font-bold">{homeScorePrediction}</p>
                  <p className="text-xs text-muted-foreground mt-1">Home</p>
                </div>
                <span className="text-xl text-muted-foreground">-</span>
                <div className="text-center">
                  <p className="text-2xl font-bold">{awayScorePrediction}</p>
                  <p className="text-xs text-muted-foreground mt-1">Away</p>
                </div>
              </div>
            </div>

            {/* Total Score */}
            <div className="flex justify-center items-center mt-3 pt-3 border-t border-border/30">
              <div className="flex items-center space-x-10">
                <div className="text-center">
                  <p className="text-sm font-medium text-primary/80">Total</p>
                  <p className="text-lg font-bold mt-1">{totalScore}</p>
                </div>
                <div className="text-center">
                  <p className="text-sm font-medium text-primary/80">Spread</p>
                  <p className="text-lg font-bold mt-1">{scoreDiff}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
