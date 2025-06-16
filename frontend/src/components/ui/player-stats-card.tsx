import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { getPlayerPhotoPath, getPlayerInitials } from "@/lib/utils/player-photos"

interface PlayerStatsCardProps {
  id?: string
  name: string
  wins: number
  losses: number
  winRate: number
  avgScore: number
  rank?: number
}

export function PlayerStatsCard({ 
  id, 
  name, 
  wins, 
  losses, 
  winRate, 
  avgScore, 
  rank 
}: PlayerStatsCardProps) {
  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center space-y-0 pb-2">
        <Avatar className="h-12 w-12 mr-4">
          <AvatarImage src={getPlayerPhotoPath(name)} alt={name} />
          <AvatarFallback>{getPlayerInitials(name)}</AvatarFallback>
        </Avatar>
        <div className="flex-1">
          <CardTitle className="text-lg">{name}</CardTitle>
          {rank && (
            <Badge variant="secondary" className="text-xs">
              Rank #{rank}
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-muted-foreground">Wins</p>
            <p className="text-2xl font-bold text-green-600">{wins}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Losses</p>
            <p className="text-2xl font-bold text-red-600">{losses}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Win Rate</p>
            <p className="text-xl font-semibold">{winRate.toFixed(1)}%</p>
          </div>
          <div>
            <p className="text-muted-foreground">Avg Score</p>
            <p className="text-xl font-semibold">{avgScore.toFixed(1)}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
