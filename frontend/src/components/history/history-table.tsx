/**
 * History table component for displaying prediction history.
 */

"use client";

import { usePredictionHistory } from "@/hooks/use-predictions";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { format } from "date-fns";
import { useState } from "react";
import { HistoryFilters } from "./history-filters";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Avatar } from "@/components/ui/avatar";
import {
  Calendar,
  Clock,
  TrendingUp,
  Users,
  BarChart3,
  History,
  CheckCircle2,
  AlertCircle,
  XCircle,
  Target
} from "lucide-react";

export function HistoryTable() {
  const [playerFilter, setPlayerFilter] = useState("");
  const [dateFilter, setDateFilter] = useState("");

  const { history, loading, error } = usePredictionHistory(playerFilter, dateFilter);

  const handleFilterChange = (player: string, date: string) => {
    setPlayerFilter(player);
    setDateFilter(date);
  };

  // Calculate summary statistics
  const getSummaryStats = () => {
    if (!history || history.length === 0) return null;

    const totalPredictions = history.length;
    const highConfidencePredictions = history.filter(
      item => Math.round(item.prediction.confidence * 100) >= 70
    ).length;

    // Calculate correct predictions - only include predictions with results (exclude pending)
    const predictionsWithResults = history.filter(
      item => item.prediction_correct !== undefined
    );
    const correctPredictions = history.filter(
      item => item.prediction_correct === true
    ).length;
    const accuracy = predictionsWithResults.length > 0 ? Math.round((correctPredictions / predictionsWithResults.length) * 100) : 0;

    const uniquePlayers = new Set();
    history.forEach(item => {
      uniquePlayers.add(item.homePlayer.name);
      uniquePlayers.add(item.awayPlayer.name);
    });

    const uniqueTeams = new Set();
    history.forEach(item => {
      uniqueTeams.add(item.homeTeam.name);
      uniqueTeams.add(item.awayTeam.name);
    });

    return {
      totalPredictions,
      highConfidencePredictions,
      correctPredictions,
      accuracy,
      predictionsWithResults: predictionsWithResults.length,
      uniquePlayers: uniquePlayers.size,
      uniqueTeams: uniqueTeams.size,
    };
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center py-12">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-muted-foreground">Loading prediction history...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        <HistoryFilters onFilterChange={handleFilterChange} />
        <div className="flex justify-center items-center py-12">
          <div className="text-center">
            <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <p className="text-red-500 font-medium">{error}</p>
            <p className="mt-2 text-muted-foreground">Please try again later.</p>
          </div>
        </div>
      </div>
    );
  }

  if (!history || history.length === 0) {
    return (
      <div className="space-y-6">
        <HistoryFilters onFilterChange={handleFilterChange} />
        <div className="flex justify-center items-center py-12">
          <div className="text-center">
            <History className="h-12 w-12 text-muted-foreground mx-auto mb-4 opacity-50" />
            <p className="text-muted-foreground">No prediction history available.</p>
            <p className="text-sm text-muted-foreground/70 mt-2">Try adjusting your filters or check back later.</p>
          </div>
        </div>
      </div>
    );
  }

  const stats = getSummaryStats();

  return (
    <div className="space-y-8">
      <HistoryFilters onFilterChange={handleFilterChange} />

      {/* Summary Statistics */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="bg-primary/5 border-primary/20">
            <CardContent className="p-4 flex items-center">
              <div className="bg-primary/10 p-2 rounded-full mr-4">
                <BarChart3 className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Total Predictions</p>
                <p className="text-2xl font-bold">{stats.totalPredictions}</p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-blue-500/5 border-blue-500/20">
            <CardContent className="p-4 flex items-center">
              <div className="bg-blue-500/10 p-2 rounded-full mr-4">
                <TrendingUp className="h-5 w-5 text-blue-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">High Confidence</p>
                <p className="text-2xl font-bold">{stats.highConfidencePredictions}</p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-green-500/5 border-green-500/20">
            <CardContent className="p-4 flex items-center">
              <div className="bg-green-500/10 p-2 rounded-full mr-4">
                <Users className="h-5 w-5 text-green-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Unique Players</p>
                <p className="text-2xl font-bold">{stats.uniquePlayers}</p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-purple-500/5 border-purple-500/20">
            <CardContent className="p-4 flex items-center">
              <div className="bg-purple-500/10 p-2 rounded-full mr-4">
                <Users className="h-5 w-5 text-purple-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Unique Teams</p>
                <p className="text-2xl font-bold">{stats.uniqueTeams}</p>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Accuracy Statistics */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="bg-emerald-500/5 border-emerald-500/20">
            <CardContent className="p-4 flex items-center">
              <div className="bg-emerald-500/10 p-2 rounded-full mr-4">
                <Target className="h-5 w-5 text-emerald-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Prediction Accuracy</p>
                <p className="text-2xl font-bold">{stats.accuracy}%</p>
                <p className="text-xs text-muted-foreground">{stats.correctPredictions} of {stats.predictionsWithResults} correct</p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-orange-500/5 border-orange-500/20">
            <CardContent className="p-4 flex items-center">
              <div className="bg-orange-500/10 p-2 rounded-full mr-4">
                <BarChart3 className="h-5 w-5 text-orange-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">High Confidence Accuracy</p>
                <p className="text-2xl font-bold">
                  {(() => {
                    const highConfidenceWithResults = history.filter(item => 
                      Math.round(item.prediction.confidence * 100) >= 70 && item.prediction_correct !== undefined
                    ).length;
                    const highConfidenceCorrect = history.filter(item => 
                      Math.round(item.prediction.confidence * 100) >= 70 && item.prediction_correct === true
                    ).length;
                    return highConfidenceWithResults > 0 ? Math.round((highConfidenceCorrect / highConfidenceWithResults) * 100) : 0;
                  })()}%
                </p>
                <p className="text-xs text-muted-foreground">
                  {history.filter(item => 
                    Math.round(item.prediction.confidence * 100) >= 70 && item.prediction_correct === true
                  ).length} of {history.filter(item => 
                    Math.round(item.prediction.confidence * 100) >= 70 && item.prediction_correct !== undefined
                  ).length} correct
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* History Table */}
      <Card className="border border-border/50 shadow-sm overflow-hidden">
        <CardHeader className="bg-muted/30 pb-0 pt-4 px-6">
          <div className="flex justify-between items-center">
            <CardTitle className="text-lg flex items-center">
              <History className="h-4 w-4 mr-2 text-primary" />
              Prediction History
            </CardTitle>
            <div className="flex items-center text-xs text-muted-foreground">
              <Calendar className="h-3.5 w-3.5 mr-1.5" />
              <span>Sorted by match date (newest first)</span>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <div className="rounded-md">
            <Table>
              <TableHeader>
                <TableRow className="bg-muted/50 hover:bg-muted/50">
                  <TableHead className="w-[180px]">
                    <div className="flex items-center text-xs font-medium text-muted-foreground">
                      <Calendar className="h-3.5 w-3.5 mr-1.5" />
                      Match Date
                    </div>
                  </TableHead>
                  <TableHead className="w-[250px]">
                    <div className="flex items-center text-xs font-medium text-muted-foreground">
                      <Users className="h-3.5 w-3.5 mr-1.5" />
                      Match
                    </div>
                  </TableHead>
                  <TableHead>
                    <div className="flex items-center text-xs font-medium text-muted-foreground">
                      <TrendingUp className="h-3.5 w-3.5 mr-1.5" />
                      Prediction
                    </div>
                  </TableHead>
                  <TableHead className="w-[100px]">
                    <div className="flex items-center text-xs font-medium text-muted-foreground">
                      <BarChart3 className="h-3.5 w-3.5 mr-1.5" />
                      Confidence
                    </div>
                  </TableHead>
                  <TableHead className="w-[120px]">
                    <div className="flex items-center text-xs font-medium text-muted-foreground">
                      <CheckCircle2 className="h-3.5 w-3.5 mr-1.5" />
                      Score
                    </div>
                  </TableHead>
                  <TableHead className="w-[120px]">
                    <div className="flex items-center text-xs font-medium text-muted-foreground">
                      <Target className="h-3.5 w-3.5 mr-1.5" />
                      Result
                    </div>
                  </TableHead>
                  <TableHead className="w-[180px]">
                    <div className="flex items-center text-xs font-medium text-muted-foreground">
                      <Clock className="h-3.5 w-3.5 mr-1.5" />
                      Saved At
                    </div>
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {[...history]
                  // Sort by match date (fixtureStart) in descending order (most recent first)
                  .sort((a, b) => {
                    // Parse match dates for sorting
                    const matchDateA = a.fixtureStart ? new Date(a.fixtureStart) : new Date(0);
                    const matchDateB = b.fixtureStart ? new Date(b.fixtureStart) : new Date(0);

                    // Compare match dates for descending order (newest first)
                    return matchDateB.getTime() - matchDateA.getTime();
                  })
                  .map((item) => {
                    const matchDate = new Date(item.fixtureStart);
                    const formattedDate = format(matchDate, "MMM d, yyyy h:mm a");

                  // Handle missing saved_at date
                  let formattedSavedDate = "N/A";
                  if (item.saved_at) {
                    const savedDate = new Date(item.saved_at);
                    formattedSavedDate = format(savedDate, "MMM d, yyyy h:mm a");
                  } else if (item.generated_at) {
                    const generatedDate = new Date(item.generated_at);
                    formattedSavedDate = format(generatedDate, "MMM d, yyyy h:mm a");
                  }

                  const homeWinner = item.prediction.predicted_winner === "home" || item.prediction.predicted_winner === "home_win";
                  const confidencePercentage = Math.round(item.prediction.confidence * 100);

                  // Determine confidence level for styling
                  let confidenceLevel = "low";
                  if (confidencePercentage >= 70) confidenceLevel = "high";
                  else if (confidencePercentage >= 50) confidenceLevel = "medium";

                  return (
                    <TableRow key={`${item.fixtureId}-${item.saved_at}`} className="hover:bg-muted/30">
                      <TableCell className="py-3">
                        <div className="text-sm font-medium">{format(matchDate, "MMM d, yyyy")}</div>
                        <div className="text-xs text-muted-foreground">{format(matchDate, "h:mm a")}</div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center justify-between px-2">
                          <div className="flex flex-col items-center text-center max-w-[100px]">
                            <Avatar className="h-8 w-8 mb-1">
                              <div className="flex h-full w-full items-center justify-center bg-primary/10 font-bold text-primary text-xs">
                                {item.homePlayer.name.substring(0, 2)}
                              </div>
                            </Avatar>
                            <div className="text-xs font-medium truncate w-full">{item.homePlayer.name}</div>
                            <div className="text-[10px] text-muted-foreground/70 truncate w-full">{item.homeTeam.name}</div>
                          </div>
                          <div className="text-xs font-medium text-muted-foreground px-1">vs</div>
                          <div className="flex flex-col items-center text-center max-w-[100px]">
                            <Avatar className="h-8 w-8 mb-1">
                              <div className="flex h-full w-full items-center justify-center bg-primary/10 font-bold text-primary text-xs">
                                {item.awayPlayer.name.substring(0, 2)}
                              </div>
                            </Avatar>
                            <div className="text-xs font-medium truncate w-full">{item.awayPlayer.name}</div>
                            <div className="text-[10px] text-muted-foreground/70 truncate w-full">{item.awayTeam.name}</div>
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center space-x-2">
                          <Avatar className="h-6 w-6">
                            <div className="flex h-full w-full items-center justify-center bg-primary/10 font-bold text-primary text-xs">
                              {(homeWinner ? item.homePlayer.name : item.awayPlayer.name).substring(0, 2)}
                            </div>
                          </Avatar>
                          <span className="font-medium">
                            {homeWinner ? item.homePlayer.name : item.awayPlayer.name}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant="outline"
                          className={`
                            ${confidenceLevel === 'high' ? 'bg-green-500/10 text-green-500 border-green-500/30' :
                              confidenceLevel === 'medium' ? 'bg-blue-500/10 text-blue-500 border-blue-500/30' :
                              'bg-orange-500/10 text-orange-500 border-orange-500/30'}
                          `}
                        >
                          {confidencePercentage}%
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <div className="space-y-1">
                          <div className="text-xs text-muted-foreground">Predicted</div>
                          <div className="font-medium">
                            {item.score_prediction.home_score} - {item.score_prediction.away_score}
                          </div>
                          {(item.homeScore !== undefined && item.awayScore !== undefined) && (
                            <>
                              <div className="text-xs text-muted-foreground">Actual</div>
                              <div className="font-medium text-sm">
                                {item.homeScore} - {item.awayScore}
                              </div>
                            </>
                          )}
                        </div>
                      </TableCell>
                      <TableCell>
                        {item.prediction_correct !== undefined ? (
                          <div className="flex items-center space-x-2">
                            {item.prediction_correct ? (
                              <>
                                <CheckCircle2 className="h-4 w-4 text-green-500" />
                                <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/30">
                                  Correct
                                </Badge>
                              </>
                            ) : (
                              <>
                                <XCircle className="h-4 w-4 text-red-500" />
                                <Badge variant="outline" className="bg-red-500/10 text-red-500 border-red-500/30">
                                  Incorrect
                                </Badge>
                              </>
                            )}
                          </div>
                        ) : (
                          <div className="flex items-center space-x-2">
                            <AlertCircle className="h-4 w-4 text-yellow-500" />
                            <Badge variant="outline" className="bg-yellow-500/10 text-yellow-500 border-yellow-500/30">
                              Pending
                            </Badge>
                          </div>
                        )}
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {formattedSavedDate}
                      </TableCell>
                    </TableRow>
                  );
                  })}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
