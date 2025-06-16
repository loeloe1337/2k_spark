/**
 * Utility functions for handling player photos and avatars
 */

/**
 * Get the path to a player's photo based on their name
 * @param playerName - The name of the player
 * @returns The path to the player's photo or default photo if not found
 */
export function getPlayerPhotoPath(playerName: string): string {
  if (!playerName) {
    return '/H2HGGL-Player-Photos/default-player-profile.webp';
  }
  
  // Convert player name to uppercase and replace spaces with nothing to match file naming
  const fileName = playerName.toUpperCase().replace(/\s+/g, '');
  return `/H2HGGL-Player-Photos/${fileName}.webp`;
}

/**
 * Get player initials for fallback display
 * @param playerName - The name of the player
 * @returns The first two characters of the player's name in uppercase
 */
export function getPlayerInitials(playerName: string): string {
  if (!playerName) {
    return 'UN';
  }
  
  return playerName.substring(0, 2).toUpperCase();
}
