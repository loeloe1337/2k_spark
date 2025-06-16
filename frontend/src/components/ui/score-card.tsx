import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Calendar, Trophy } from "lucide-react"
import { getPlayerPhotoPath, getPlayerInitials } from "@/lib/utils/player-photos"

interface ScoreCardProps {
  match: {
    id: string
    homePlayer: {
      id: string
      name: string
      score: number
    }
    awayPlayer: {
      id: string
      name: string
      score: number
    }
    winner: {
      id: string
      name: string
    }
    date: string
    status: 'completed' | 'live' | 'upcoming'
  }
}

export function ScoreCard({ match }: ScoreCardProps) {
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    })
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'live': return 'destructive'
      case 'completed': return 'secondary'
      default: return 'outline'
    }
  }

  const isWinner = (playerId: string) => {
    return playerId === match.winner.id
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Match Result</CardTitle>
          <Badge variant={getStatusColor(match.status)}>
            {match.status.toUpperCase()}
          </Badge>
        </div>
        <div className="flex items-center text-sm text-muted-foreground">
          <Calendar className="h-4 w-4 mr-1" />
          {formatDate(match.date)}
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Players and Scores */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <Avatar className="h-12 w-12">
                  <AvatarImage src={getPlayerPhotoPath(match.homePlayer.name)} alt={match.homePlayer.name} />
                  <AvatarFallback>{getPlayerInitials(match.homePlayer.name)}</AvatarFallback>
                </Avatar>
                {isWinner(match.homePlayer.id) && (
                  <Trophy className="h-4 w-4 text-yellow-500 absolute -top-1 -right-1" />
                )}
              </div>
              <div>
                <p className={`font-medium ${isWinner(match.homePlayer.id) ? 'text-yellow-600' : ''}`}>
                  {match.homePlayer.name}
                </p>
                <p className="text-2xl font-bold">{match.homePlayer.score}</p>
              </div>
            </div>
            
            <div className="text-center">
              <p className="text-muted-foreground font-medium">VS</p>
            </div>
            
            <div className="flex items-center space-x-3">
              <div className="text-right">
                <p className={`font-medium ${isWinner(match.awayPlayer.id) ? 'text-yellow-600' : ''}`}>
                  {match.awayPlayer.name}
                </p>
                <p className="text-2xl font-bold">{match.awayPlayer.score}</p>
              </div>
              <div className="relative">
                <Avatar className="h-12 w-12">
                  <AvatarImage src={getPlayerPhotoPath(match.awayPlayer.name)} alt={match.awayPlayer.name} />
                  <AvatarFallback>{getPlayerInitials(match.awayPlayer.name)}</AvatarFallback>
                </Avatar>
                {isWinner(match.awayPlayer.id) && (
                  <Trophy className="h-4 w-4 text-yellow-500 absolute -top-1 -right-1" />
                )}
              </div>
            </div>
          </div>
          
          {/* Winner */}
          {match.status === 'completed' && (
            <div className="border-t pt-4">
              <div className="flex items-center justify-center space-x-2">
                <Trophy className="h-5 w-5 text-yellow-500" />
                <span className="text-sm text-muted-foreground">Winner:</span>
                <Avatar className="h-6 w-6">
                  <AvatarImage src={getPlayerPhotoPath(match.winner.name)} alt={match.winner.name} />
                  <AvatarFallback className="text-xs">{getPlayerInitials(match.winner.name)}</AvatarFallback>
                </Avatar>
                <span className="font-medium text-yellow-600">{match.winner.name}</span>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
