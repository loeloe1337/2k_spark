/**
 * Score card component for displaying score predictions.
 */

import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { format } from "date-fns";
import { getPlayerPhotoPath, getPlayerInitials } from "@/lib/utils/player-photos";

interface ScoreCardProps {
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
    score_prediction: {
      home_score: number;
      away_score: number;
      total_score: number;
      score_diff: number;
    };
  };
}

export function ScoreCard({ prediction }: ScoreCardProps) {
  const {
    homePlayer,
    awayPlayer,
    homeTeam,
    awayTeam,
    fixtureStart,
    score_prediction,
  } = prediction;

  const matchDate = new Date(fixtureStart);
  const formattedDate = format(matchDate, "MMM d, yyyy");
  const formattedTime = format(matchDate, "h:mm a");

  const homeWinner = score_prediction.home_score > score_prediction.away_score;
  const scoreDiff = Math.abs(score_prediction.score_diff);

  return (
    <Card className="overflow-hidden border border-border/50 shadow-md hover:shadow-lg transition-all duration-300 hover:border-border/80">
      <CardContent className="p-6 pt-5">
        {/* Date/Time Badge - Moved to top right corner */}
        <div className="absolute top-3 right-4">
          <div className="text-xs font-medium bg-muted/80 px-3 py-1.5 rounded-full border border-border/30">
            {formattedDate} • {formattedTime}
          </div>
        </div>

        {/* Players with Scores */}
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
              <div className="bg-primary/10 px-4 py-1 rounded-full border border-primary/20 mt-2">
                <p className="text-2xl font-bold">{score_prediction.home_score}</p>
              </div>
            </div>
          </div>

          {/* VS */}
          <div className="flex flex-col items-center px-2">
            <div className="relative">
              <p className="text-xl font-bold bg-gradient-to-r from-blue-500/80 to-primary/80 bg-clip-text text-transparent">VS</p>
              <div className="absolute -inset-3 bg-gradient-to-r from-blue-500/5 to-primary/5 rounded-full blur-md -z-10"></div>
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
              <div className="bg-primary/10 px-4 py-1 rounded-full border border-primary/20 mt-2">
                <p className="text-2xl font-bold">{score_prediction.away_score}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Win By Information */}
        <div className="bg-muted/70 p-4 rounded-lg border border-border/30 shadow-sm mb-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <div className="bg-primary/10 h-8 w-8 rounded-full flex items-center justify-center border border-primary/20">
                <div className="text-primary font-bold text-sm">
                  {scoreDiff}
                </div>
              </div>
              <div>
                <p className="text-sm font-medium text-primary/90">Win By</p>
                <p className="text-base font-bold">{scoreDiff} points</p>
              </div>
            </div>
            <div className="text-base font-bold bg-primary/10 px-3 py-1 rounded-full border border-primary/20">
              {homeWinner ? homePlayer.name : awayPlayer.name}
            </div>
          </div>
        </div>

        {/* Winner and Total Score */}
        <div className="grid grid-cols-2 gap-4">
          {/* Winner */}
          <div className="bg-gradient-to-br from-green-50/80 to-emerald-50/60 dark:from-green-950/30 dark:to-emerald-950/20 p-4 rounded-lg border border-green-200/50 dark:border-green-800/30 shadow-sm shadow-green-500/10 relative overflow-hidden">
            <div className="absolute top-2 right-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse shadow-lg shadow-green-500/50"></div>
            </div>
            <p className="text-sm font-medium text-green-700 dark:text-green-300 mb-2">Predicted Winner</p>
            <div className="flex items-center space-x-2">
              <Avatar className="h-6 w-6 ring-2 ring-green-500/30 shadow-md shadow-green-500/20">
                <AvatarImage src={getPlayerPhotoPath(homeWinner ? homePlayer.name : awayPlayer.name)} alt={homeWinner ? homePlayer.name : awayPlayer.name} />
                <AvatarFallback className="bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300 font-bold text-xs">{getPlayerInitials(homeWinner ? homePlayer.name : awayPlayer.name)}</AvatarFallback>
              </Avatar>
              <p className="text-base font-bold text-green-800 dark:text-green-200">{homeWinner ? homePlayer.name : awayPlayer.name}</p>
            </div>
          </div>

          {/* Total Score */}
          <div className="bg-muted/70 p-4 rounded-lg border border-border/30 shadow-sm">
            <p className="text-sm font-medium text-primary/90 mb-2">Total Score</p>
            <p className="text-xl font-bold">{score_prediction.total_score} <span className="text-sm font-normal text-muted-foreground">points</span></p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
