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
    const pendingPredictions = history.filter(
      item => item.prediction_correct === undefined
    ).length;
    const averageConfidence = history.length > 0 
      ? Math.round((history.reduce((sum, item) => sum + item.prediction.confidence, 0) / history.length) * 100)
      : 0;

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
      pendingPredictions,
      averageConfidence,
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

      {/* Enhanced Summary Statistics */}
      {stats && (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-5 mb-8">
          <Card className="border border-border/50 shadow-lg hover:shadow-xl transition-all duration-300 bg-gradient-to-br from-blue-500/5 to-blue-600/10">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-3xl font-bold text-blue-600">{stats.totalPredictions}</p>
                  <p className="text-sm font-medium text-muted-foreground mt-1">Total Predictions</p>
                </div>
                <div className="bg-blue-500/15 p-3 rounded-xl">
                  <BarChart3 className="h-6 w-6 text-blue-600" />
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card className="border border-border/50 shadow-lg hover:shadow-xl transition-all duration-300 bg-gradient-to-br from-green-500/5 to-green-600/10">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-3xl font-bold text-green-600">{stats.correctPredictions}</p>
                  <p className="text-sm font-medium text-muted-foreground mt-1">Correct Predictions</p>
                </div>
                <div className="bg-green-500/15 p-3 rounded-xl">
                  <CheckCircle2 className="h-6 w-6 text-green-600" />
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card className="border border-border/50 shadow-lg hover:shadow-xl transition-all duration-300 bg-gradient-to-br from-purple-500/5 to-purple-600/10">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-3xl font-bold text-purple-600">{stats.highConfidencePredictions}</p>
                  <p className="text-sm font-medium text-muted-foreground mt-1">High Confidence</p>
                </div>
                <div className="bg-purple-500/15 p-3 rounded-xl">
                  <TrendingUp className="h-6 w-6 text-purple-600" />
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card className="border border-border/50 shadow-lg hover:shadow-xl transition-all duration-300 bg-gradient-to-br from-orange-500/5 to-orange-600/10">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-3xl font-bold text-orange-600">{stats.pendingPredictions}</p>
                  <p className="text-sm font-medium text-muted-foreground mt-1">Pending Results</p>
                </div>
                <div className="bg-orange-500/15 p-3 rounded-xl">
                  <Clock className="h-6 w-6 text-orange-600" />
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card className="border border-border/50 shadow-lg hover:shadow-xl transition-all duration-300 bg-gradient-to-br from-indigo-500/5 to-indigo-600/10">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-3xl font-bold text-indigo-600">{stats.averageConfidence}%</p>
                  <p className="text-sm font-medium text-muted-foreground mt-1">Avg Confidence</p>
                </div>
                <div className="bg-indigo-500/15 p-3 rounded-xl">
                  <Target className="h-6 w-6 text-indigo-600" />
                </div>
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
      <Card className="border border-border/50 shadow-xl bg-gradient-to-br from-background to-muted/20">
        <CardHeader className="pb-6 bg-gradient-to-r from-primary/5 to-primary/10 rounded-t-lg">
          <CardTitle className="text-2xl flex items-center">
            <div className="bg-primary/15 p-3 rounded-xl mr-4 shadow-lg">
              <Clock className="h-6 w-6 text-primary" />
            </div>
            <div>
              <h2 className="font-bold">Prediction History</h2>
              <p className="text-sm text-muted-foreground font-normal mt-1">Complete record of your NBA 2K25 eSports predictions</p>
            </div>
            <div className="ml-auto flex items-center space-x-3">
              <Badge variant="secondary" className="bg-primary/10 text-primary border-primary/20 px-3 py-1">
                {history.length} total predictions
              </Badge>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="border-b border-border/50 bg-muted/30">
                  <TableHead className="w-[140px] py-4">
                    <div className="flex items-center text-sm font-semibold text-foreground">
                      <Calendar className="h-4 w-4 mr-2 text-primary" />
                      Match Date
                    </div>
                  </TableHead>
                  <TableHead className="w-[280px] py-4">
                    <div className="flex items-center text-sm font-semibold text-foreground">
                      <Users className="h-4 w-4 mr-2 text-primary" />
                      Match Details
                    </div>
                  </TableHead>
                  <TableHead className="py-4">
                    <div className="flex items-center text-sm font-semibold text-foreground">
                      <TrendingUp className="h-4 w-4 mr-2 text-primary" />
                      Prediction
                    </div>
                  </TableHead>
                  <TableHead className="w-[120px] py-4">
                    <div className="flex items-center text-sm font-semibold text-foreground">
                      <BarChart3 className="h-4 w-4 mr-2 text-primary" />
                      Confidence
                    </div>
                  </TableHead>
                  <TableHead className="w-[140px] py-4">
                    <div className="flex items-center text-sm font-semibold text-foreground">
                      <CheckCircle2 className="h-4 w-4 mr-2 text-primary" />
                      Score
                    </div>
                  </TableHead>
                  <TableHead className="w-[140px] py-4">
                    <div className="flex items-center text-sm font-semibold text-foreground">
                      <Target className="h-4 w-4 mr-2 text-primary" />
                      Result
                    </div>
                  </TableHead>
                  <TableHead className="w-[180px] py-4">
                    <div className="flex items-center text-sm font-semibold text-foreground">
                      <Clock className="h-4 w-4 mr-2 text-primary" />
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
                    <TableRow key={`${item.fixtureId}-${item.saved_at}`} className="hover:bg-gradient-to-r hover:from-primary/5 hover:to-primary/10 transition-all duration-300 border-b border-border/30">
                      <TableCell className="py-4">
                        <div className="bg-muted/50 rounded-lg p-3 text-center">
                          <div className="text-sm font-bold text-foreground">{format(matchDate, "MMM d")}</div>
                          <div className="text-xs text-muted-foreground">{format(matchDate, "yyyy")}</div>
                          <div className="text-xs text-primary font-medium mt-1">{format(matchDate, "h:mm a")}</div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="bg-gradient-to-r from-muted/30 to-muted/50 rounded-xl p-4">
                          <div className="flex items-center justify-between">
                            <div className="flex flex-col items-center text-center max-w-[110px]">
                              <Avatar className="h-10 w-10 mb-2 ring-2 ring-primary/20 shadow-lg">
                                <div className="flex h-full w-full items-center justify-center bg-gradient-to-br from-primary/20 to-primary/30 font-bold text-primary text-sm">
                                  {item.homePlayer.name.substring(0, 2)}
                                </div>
                              </Avatar>
                              <div className="text-sm font-semibold truncate w-full">{item.homePlayer.name}</div>
                              <div className="text-xs text-muted-foreground truncate w-full">{item.homeTeam.name}</div>
                            </div>
                            <div className="text-sm font-bold text-primary bg-primary/10 px-3 py-1 rounded-full">VS</div>
                            <div className="flex flex-col items-center text-center max-w-[110px]">
                              <Avatar className="h-10 w-10 mb-2 ring-2 ring-primary/20 shadow-lg">
                                <div className="flex h-full w-full items-center justify-center bg-gradient-to-br from-primary/20 to-primary/30 font-bold text-primary text-sm">
                                  {item.awayPlayer.name.substring(0, 2)}
                                </div>
                              </Avatar>
                              <div className="text-sm font-semibold truncate w-full">{item.awayPlayer.name}</div>
                              <div className="text-xs text-muted-foreground truncate w-full">{item.awayTeam.name}</div>
                            </div>
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="bg-gradient-to-r from-green-500/10 to-green-600/20 rounded-lg p-3 border border-green-500/20">
                          <div className="flex items-center space-x-3">
                            <Avatar className="h-8 w-8 ring-2 ring-green-500/30">
                              <div className="flex h-full w-full items-center justify-center bg-gradient-to-br from-green-500/20 to-green-600/30 font-bold text-green-600 text-xs">
                                {(homeWinner ? item.homePlayer.name : item.awayPlayer.name).substring(0, 2)}
                              </div>
                            </Avatar>
                            <div>
                              <div className="font-semibold text-green-700">
                                {homeWinner ? item.homePlayer.name : item.awayPlayer.name}
                              </div>
                              <div className="text-xs text-green-600/80">Predicted Winner</div>
                            </div>
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="text-center">
                          <Badge
                            variant="outline"
                            className={`
                              text-lg font-bold px-4 py-2 shadow-lg
                              ${confidenceLevel === 'high' ? 'bg-gradient-to-r from-green-500/20 to-green-600/30 text-green-600 border-green-500/40 shadow-green-500/20' :
                                confidenceLevel === 'medium' ? 'bg-gradient-to-r from-blue-500/20 to-blue-600/30 text-blue-600 border-blue-500/40 shadow-blue-500/20' :
                                'bg-gradient-to-r from-orange-500/20 to-orange-600/30 text-orange-600 border-orange-500/40 shadow-orange-500/20'}
                            `}
                          >
                            {confidencePercentage}%
                          </Badge>
                          <div className="text-xs text-muted-foreground mt-1">
                            {confidenceLevel === 'high' ? 'High' : confidenceLevel === 'medium' ? 'Medium' : 'Low'} Confidence
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="bg-muted/50 rounded-lg p-3">
                          <div className="space-y-2">
                            <div>
                              <div className="text-xs text-muted-foreground mb-1">Predicted</div>
                              <div className="font-bold text-lg text-primary">
                                {item.score_prediction.home_score} - {item.score_prediction.away_score}
                              </div>
                            </div>
                            {(item.homeScore !== undefined && item.awayScore !== undefined) && (
                              <div className="border-t border-border/50 pt-2">
                                <div className="text-xs text-muted-foreground mb-1">Actual</div>
                                <div className="font-bold text-lg">
                                  {item.homeScore} - {item.awayScore}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="text-center">
                          {item.prediction_correct !== undefined ? (
                            <div className="space-y-2">
                              {item.prediction_correct ? (
                                <>
                                  <div className="bg-green-500/10 p-3 rounded-lg border border-green-500/20">
                                    <CheckCircle2 className="h-6 w-6 text-green-500 mx-auto mb-1" />
                                    <Badge variant="outline" className="bg-green-500/20 text-green-600 border-green-500/40 font-semibold">
                                      Correct
                                    </Badge>
                                  </div>
                                </>
                              ) : (
                                <>
                                  <div className="bg-red-500/10 p-3 rounded-lg border border-red-500/20">
                                    <XCircle className="h-6 w-6 text-red-500 mx-auto mb-1" />
                                    <Badge variant="outline" className="bg-red-500/20 text-red-600 border-red-500/40 font-semibold">
                                      Incorrect
                                    </Badge>
                                  </div>
                                </>
                              )}
                            </div>
                          ) : (
                            <div className="bg-yellow-500/10 p-3 rounded-lg border border-yellow-500/20">
                              <AlertCircle className="h-6 w-6 text-yellow-500 mx-auto mb-1" />
                              <Badge variant="outline" className="bg-yellow-500/20 text-yellow-600 border-yellow-500/40 font-semibold">
                                Pending
                              </Badge>
                            </div>
                          )}
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="bg-muted/50 rounded-lg p-3 text-center">
                          <div className="text-sm font-medium">{formattedSavedDate}</div>
                        </div>
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
