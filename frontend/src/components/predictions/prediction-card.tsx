/**
 * Prediction card component for displaying match predictions.
 */

import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar } from "@/components/ui/avatar";
import { format } from "date-fns";

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
    <Card className={`overflow-hidden card-highlight card-hover ${isHomeWinner ? "border-l-4 border-l-green-500" : "border-r-4 border-r-green-500"}`}>
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
              <div className="flex h-full w-full items-center justify-center bg-primary/10 font-bold text-primary">
                {homePlayer.name.substring(0, 2)}
              </div>
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
              <div className="flex h-full w-full items-center justify-center bg-primary/10 font-bold text-primary">
                {awayPlayer.name.substring(0, 2)}
              </div>
            </Avatar>
            <div>
              <p className="font-semibold text-base">{awayPlayer.name}</p>
              <p className="text-sm text-muted-foreground">{awayTeam.name}</p>
              <p className="text-sm font-bold mt-1 text-primary/90">{awayWinPercentage}%</p>
            </div>
          </div>
        </div>

        {/* Winner Prediction with Confidence */}
        <div className="bg-muted/70 p-4 rounded-lg border border-border/30 shadow-sm">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <div className="bg-primary/10 h-8 w-8 rounded-full flex items-center justify-center border border-primary/20">
                <div className="text-primary font-bold text-sm">
                  {isHomeWinner ? homePlayer.name.substring(0, 2) : awayPlayer.name.substring(0, 2)}
                </div>
              </div>
              <div>
                <p className="text-sm font-medium text-primary/90">Predicted Winner</p>
                <p className="text-base font-bold">{isHomeWinner ? homePlayer.name : awayPlayer.name}</p>
              </div>
            </div>
            <div className="text-base font-bold bg-primary/10 px-3 py-1 rounded-full border border-primary/20">
              {isHomeWinner ? homeWinPercentage : awayWinPercentage}%
            </div>
          </div>

          {fetched_at && (
            <div className="text-xs text-muted-foreground mt-3 text-right">
              Last updated: {fetched_at}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
