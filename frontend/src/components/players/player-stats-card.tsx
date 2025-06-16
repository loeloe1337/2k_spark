/**
 * Player stats card component for displaying player statistics.
 */

"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Progress } from "@/components/ui/progress";
import { getPlayerPhotoPath, getPlayerInitials } from "@/lib/utils/player-photos";

interface PlayerStatsCardProps {
  player: any;
}

export function PlayerStatsCard({ player }: PlayerStatsCardProps) {
  const { player_name, id, win_rate, total_matches, avg_score, teams_used } = player;
  
  // Find most used team
  let mostUsedTeam = { team_name: "Unknown", matches: 0 };
  if (teams_used) {
    Object.entries(teams_used).forEach(([_, teamData]: [string, any]) => {
      if (teamData.matches > mostUsedTeam.matches) {
        mostUsedTeam = { 
          team_name: teamData.team_name || "Unknown", 
          matches: teamData.matches 
        };
      }
    });
  }

  // Format win rate as percentage
  const winRatePercentage = Math.round(win_rate * 100);

  return (
    <Card className="overflow-hidden">
      <CardHeader className="bg-muted/50 pb-2">
        <CardTitle className="text-sm font-medium flex justify-between items-center">
          <span>Player Stats</span>
          <span className="text-xs text-muted-foreground">
            {total_matches} matches
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-4">
        <div className="flex items-center space-x-4 mb-4">
          <Avatar className="h-16 w-16 border-2 border-primary">
            <AvatarImage src={getPlayerPhotoPath(player_name)} alt={player_name} />
            <AvatarFallback>{getPlayerInitials(player_name)}</AvatarFallback>
          </Avatar>
          <div>
            <p className="font-bold text-lg">{player_name}</p>
            <p className="text-sm text-muted-foreground">
              Favorite Team: {mostUsedTeam.team_name}
            </p>
          </div>
        </div>
        
        <div className="space-y-4">
          <div>
            <div className="flex justify-between mb-1">
              <span className="text-sm font-medium">Win Rate</span>
              <span className="text-sm font-medium">{winRatePercentage}%</span>
            </div>
            <Progress value={winRatePercentage} className="h-2" />
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-muted/30 p-3 rounded-md">
              <p className="text-xs text-muted-foreground">Avg Score</p>
              <p className="text-lg font-bold">{avg_score.toFixed(1)}</p>
            </div>
            <div className="bg-muted/30 p-3 rounded-md">
              <p className="text-xs text-muted-foreground">Total Matches</p>
              <p className="text-lg font-bold">{total_matches}</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
