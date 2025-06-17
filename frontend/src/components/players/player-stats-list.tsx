/**
 * Player stats list component for displaying a list of player statistics.
 */

"use client";

import { usePlayerStats } from "@/hooks/use-predictions";
import { PlayerStatsCard } from "./player-stats-card";
import { useState } from "react";
import { Input } from "@/components/ui/input";

export function PlayerStatsList() {
  const { playerStats, loading, error } = usePlayerStats();
  const [searchQuery, setSearchQuery] = useState("");

  if (loading) {
    return (
      <div className="flex justify-center items-center py-12">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-muted-foreground">Loading player statistics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex justify-center items-center py-12">
        <div className="text-center">
          <p className="text-red-500">{error}</p>
          <p className="mt-2 text-muted-foreground">Please try again later.</p>
        </div>
      </div>
    );
  }
  if (!playerStats || !Array.isArray(playerStats) || playerStats.length === 0) {
    return (
      <div className="flex justify-center items-center py-12">
        <div className="text-center">
          <p className="text-muted-foreground">No player statistics available.</p>
        </div>
      </div>
    );
  }

  // Filter players based on search query - extra safety check
  const filteredPlayers = Array.isArray(playerStats) 
    ? playerStats.filter(player => 
        player?.player_name?.toLowerCase()?.includes(searchQuery.toLowerCase()) || false
      )
    : [];

  // Sort players by win rate (descending)
  const sortedPlayers = [...filteredPlayers].sort((a, b) => b.win_rate - a.win_rate);

  return (
    <div className="space-y-6">
      <div className="max-w-md mx-auto mb-8">
        <Input
          type="text"
          placeholder="Search players..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full"
        />
      </div>

      {filteredPlayers.length === 0 ? (
        <div className="text-center py-8">
          <p className="text-muted-foreground">No players found matching "{searchQuery}"</p>
        </div>
      ) : (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {sortedPlayers.map((player) => (
            <PlayerStatsCard key={player.id} player={player} />
          ))}
        </div>
      )}
    </div>
  );
}
