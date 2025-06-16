import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Clock } from "lucide-react"
import { getPlayerPhotoPath, getPlayerInitials } from "@/lib/utils/player-photos"

interface LiveMatchCardProps {
  match: {
    id: string
    homePlayer: {
      id: string
      name: string
    }
    awayPlayer: {
      id: string
      name: string
    }
    status: string
    startTime: string
    predictedWinner?: {
      id: string
      name: string
      confidence: number
    }
  }
}

export function LiveMatchCard({ match }: LiveMatchCardProps) {
  const formatTime = (timeString: string) => {
    return new Date(timeString).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Live Match</CardTitle>
          <Badge variant={match.status === 'live' ? 'destructive' : 'secondary'}>
            {match.status === 'live' ? 'LIVE' : match.status.toUpperCase()}
          </Badge>
        </div>
        <div className="flex items-center text-sm text-muted-foreground">
          <Clock className="h-4 w-4 mr-1" />
          {formatTime(match.startTime)}
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Players */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Avatar className="h-10 w-10">
                <AvatarImage src={getPlayerPhotoPath(match.homePlayer.name)} alt={match.homePlayer.name} />
                <AvatarFallback>{getPlayerInitials(match.homePlayer.name)}</AvatarFallback>
              </Avatar>
              <span className="font-medium">{match.homePlayer.name}</span>
            </div>
            <span className="text-muted-foreground font-medium">VS</span>
            <div className="flex items-center space-x-3">
              <span className="font-medium">{match.awayPlayer.name}</span>
              <Avatar className="h-10 w-10">
                <AvatarImage src={getPlayerPhotoPath(match.awayPlayer.name)} alt={match.awayPlayer.name} />
                <AvatarFallback>{getPlayerInitials(match.awayPlayer.name)}</AvatarFallback>
              </Avatar>
            </div>
          </div>
          
          {/* Prediction */}
          {match.predictedWinner && (
            <div className="border-t pt-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Predicted Winner</span>
                <div className="flex items-center space-x-2">
                  <Avatar className="h-6 w-6">
                    <AvatarImage src={getPlayerPhotoPath(match.predictedWinner.name)} alt={match.predictedWinner.name} />
                    <AvatarFallback className="text-xs">{getPlayerInitials(match.predictedWinner.name)}</AvatarFallback>
                  </Avatar>
                  <span className="font-medium">{match.predictedWinner.name}</span>
                  <Badge variant="outline">
                    {(match.predictedWinner.confidence * 100).toFixed(0)}%
                  </Badge>
                </div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
