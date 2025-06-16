/**
 * Prediction card component for displaying match predictions.
 */

import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { format } from "date-fns";
import { getPlayerPhotoPath, getPlayerInitials } from "@/lib/utils/player-photos";
import { ExternalLink } from "lucide-react";

interface PredictionCardProps {
  prediction: {
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
  };
}

export function PredictionCard({ prediction }: PredictionCardProps) {
  const {
    homePlayer,
    awayPlayer,
    homeTeam,
    awayTeam,
    fixtureStart,
    fetched_at,
    prediction: predictionData,
  } = prediction;

  const matchDate = new Date(fixtureStart);
  const formattedDate = format(matchDate, "MMM d, yyyy");
  const formattedTime = format(matchDate, "h:mm a");

  const homeWinPercentage = Math.round(predictionData.home_win_probability * 100);
  const awayWinPercentage = Math.round(predictionData.away_win_probability * 100);
  const confidencePercentage = Math.round(predictionData.confidence * 100);

  const isHomeWinner = predictionData.predicted_winner === "home";

  return (
    <Card className="overflow-hidden card-highlight card-hover border-2 border-border/30 shadow-lg hover:shadow-xl transition-all duration-300 relative">
      <CardContent className="p-6 pt-5">
        {/* Date/Time Badge - Moved to top right corner */}
        <div className="absolute top-3 right-4">
          <div className="text-xs font-medium bg-muted/80 px-3 py-1.5 rounded-full border border-border/30">
            {formattedDate} • {formattedTime}
          </div>
        </div>

        {/* Players */}
        <div className="flex justify-between items-center mb-6 mt-4">
          {/* Home Player */}
          <div className="flex flex-col items-center text-center space-y-3">
            <Avatar className="h-16 w-16 border-2 border-primary/20 shadow-lg shadow-primary/10">
              <AvatarImage src={getPlayerPhotoPath(homePlayer.name)} alt={homePlayer.name} />
              <AvatarFallback className="bg-primary/10 text-primary font-bold">{getPlayerInitials(homePlayer.name)}</AvatarFallback>
            </Avatar>
            <div>
              <p className="font-semibold text-base">{homePlayer.name}</p>
              <p className="text-sm text-muted-foreground">{homeTeam.name}</p>
              <p className="text-sm font-bold mt-1 text-primary/90">{homeWinPercentage}%</p>
            </div>
          </div>

          {/* VS */}
          <div className="flex flex-col items-center px-2">
            <div className="relative">
              <p className="text-xl font-bold bg-gradient-to-r from-primary/80 to-blue-500/80 bg-clip-text text-transparent">VS</p>
              <div className="absolute -inset-3 bg-gradient-to-r from-primary/5 to-blue-500/5 rounded-full blur-md -z-10"></div>
            </div>
          </div>

          {/* Away Player */}
          <div className="flex flex-col items-center text-center space-y-3">
            <Avatar className="h-16 w-16 border-2 border-primary/20 shadow-lg shadow-primary/10">
              <AvatarImage src={getPlayerPhotoPath(awayPlayer.name)} alt={awayPlayer.name} />
              <AvatarFallback className="bg-primary/10 text-primary font-bold">{getPlayerInitials(awayPlayer.name)}</AvatarFallback>
            </Avatar>
            <div>
              <p className="font-semibold text-base">{awayPlayer.name}</p>
              <p className="text-sm text-muted-foreground">{awayTeam.name}</p>
              <p className="text-sm font-bold mt-1 text-primary/90">{awayWinPercentage}%</p>
            </div>
          </div>
        </div>

        {/* Winner Prediction with Confidence */}
        <div className="bg-gradient-to-r from-green-50/80 to-emerald-50/80 dark:from-green-950/30 dark:to-emerald-950/30 p-4 rounded-lg border border-green-200/50 dark:border-green-800/30 shadow-sm relative overflow-hidden">
          {/* Subtle glow effect */}
          <div className="absolute inset-0 bg-gradient-to-r from-green-400/5 to-emerald-400/5 rounded-lg"></div>
          
          <div className="flex justify-between items-center relative z-10">
            <div className="flex items-center space-x-3">
              <Avatar className="h-8 w-8 border-2 border-green-300/50 dark:border-green-700/50 shadow-md">
                <AvatarImage src={getPlayerPhotoPath(isHomeWinner ? homePlayer.name : awayPlayer.name)} alt={isHomeWinner ? homePlayer.name : awayPlayer.name} />
                <AvatarFallback className="bg-green-100/80 dark:bg-green-900/50 text-green-700 dark:text-green-300 font-bold text-sm">{getPlayerInitials(isHomeWinner ? homePlayer.name : awayPlayer.name)}</AvatarFallback>
              </Avatar>
              <div>
                <p className="text-sm font-medium text-green-700 dark:text-green-300 flex items-center gap-1">
                  <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                  Predicted Winner
                </p>
                <p className="text-base font-bold text-green-800 dark:text-green-200">{isHomeWinner ? homePlayer.name : awayPlayer.name}</p>
              </div>
            </div>
            <div className="text-base font-bold bg-green-100/80 dark:bg-green-900/50 text-green-700 dark:text-green-300 px-3 py-1 rounded-full border border-green-300/50 dark:border-green-700/50 shadow-sm">
              {isHomeWinner ? homeWinPercentage : awayWinPercentage}%
            </div>
          </div>

          {fetched_at && (
            <div className="text-xs text-muted-foreground mt-3 text-right">
              Last updated: {fetched_at}
            </div>
          )}
        </div>

        {/* FanDuel Betting Button */}
        <div className="mt-4">
          <Button 
            onClick={() => window.open('https://sportsbook.fanduel.com/basketball/ebasketball-h2h-gg-league-4x5-mins', '_blank')}
            className="w-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-semibold py-2.5 px-4 rounded-lg shadow-lg hover:shadow-xl transition-all duration-300 flex items-center justify-center gap-2"
          >
            <ExternalLink className="h-4 w-4" />
            Place Bet on FanDuel
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
