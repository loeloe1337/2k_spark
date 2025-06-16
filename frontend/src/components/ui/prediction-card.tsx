import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { TrendingUp, TrendingDown } from "lucide-react"
import { getPlayerPhotoPath, getPlayerInitials } from "@/lib/utils/player-photos"

interface PredictionCardProps {
  prediction: {
    id: string
    homePlayer: {
      id: string
      name: string
    }
    awayPlayer: {
      id: string
      name: string
    }
    predictedWinner: {
      id: string
      name: string
      confidence: number
    }
    predictedScore: {
      home: number
      away: number
    }
    matchDate: string
    status: 'pending' | 'completed' | 'live'
  }
}

export function PredictionCard({ prediction }: PredictionCardProps) {
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'live': return 'destructive'
      case 'completed': return 'secondary'
      default: return 'outline'
    }
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Match Prediction</CardTitle>
          <Badge variant={getStatusColor(prediction.status)}>
            {prediction.status.toUpperCase()}
          </Badge>
        </div>
        <p className="text-sm text-muted-foreground">
          {formatDate(prediction.matchDate)}
        </p>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Players and Score */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Avatar className="h-12 w-12">
                <AvatarImage src={getPlayerPhotoPath(prediction.homePlayer.name)} alt={prediction.homePlayer.name} />
                <AvatarFallback>{getPlayerInitials(prediction.homePlayer.name)}</AvatarFallback>
              </Avatar>
              <div>
                <p className="font-medium">{prediction.homePlayer.name}</p>
                <p className="text-2xl font-bold">{prediction.predictedScore.home}</p>
              </div>
            </div>
            
            <div className="text-center">
              <p className="text-muted-foreground font-medium">VS</p>
            </div>
            
            <div className="flex items-center space-x-3">
              <div className="text-right">
                <p className="font-medium">{prediction.awayPlayer.name}</p>
                <p className="text-2xl font-bold">{prediction.predictedScore.away}</p>
              </div>
              <Avatar className="h-12 w-12">
                <AvatarImage src={getPlayerPhotoPath(prediction.awayPlayer.name)} alt={prediction.awayPlayer.name} />
                <AvatarFallback>{getPlayerInitials(prediction.awayPlayer.name)}</AvatarFallback>
              </Avatar>
            </div>
          </div>
          
          {/* Predicted Winner */}
          <div className="border-t pt-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Predicted Winner</span>
              <div className="flex items-center space-x-2">
                {prediction.predictedWinner.confidence > 0.6 ? (
                  <TrendingUp className="h-4 w-4 text-green-600" />
                ) : (
                  <TrendingDown className="h-4 w-4 text-yellow-600" />
                )}
                <Avatar className="h-6 w-6">
                  <AvatarImage src={getPlayerPhotoPath(prediction.predictedWinner.name)} alt={prediction.predictedWinner.name} />
                  <AvatarFallback className="text-xs">{getPlayerInitials(prediction.predictedWinner.name)}</AvatarFallback>
                </Avatar>
                <span className="font-medium">{prediction.predictedWinner.name}</span>
                <Badge variant="outline">
                  {(prediction.predictedWinner.confidence * 100).toFixed(0)}%
                </Badge>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
