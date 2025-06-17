/**
 * API Client for 2K Spark Backend
 * Handles all HTTP requests to the backend API with error handling and retry logic
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

export interface ApiResponse<T> {
  data?: T;
  error?: string;
  status: number;
}

class ApiClient {
  private baseUrl: string;
  private defaultHeaders: Record<string, string>;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    };
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseUrl}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          ...this.defaultHeaders,
          ...options.headers,
        },
      });

      const isJson = response.headers.get('content-type')?.includes('application/json');
      const data = isJson ? await response.json() : await response.text();

      if (!response.ok) {
        return {
          error: data.error || data || `HTTP ${response.status}: ${response.statusText}`,
          status: response.status,
        };
      }

      return {
        data,
        status: response.status,
      };
    } catch (error) {
      console.error('API Request failed:', error);
      return {
        error: error instanceof Error ? error.message : 'Network error occurred',
        status: 0,
      };
    }
  }

  // GET request
  async get<T>(endpoint: string): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      method: 'GET',
    });
  }

  // POST request
  async post<T>(endpoint: string, data?: any): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  // PUT request
  async put<T>(endpoint: string, data?: any): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  // DELETE request
  async delete<T>(endpoint: string): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      method: 'DELETE',
    });
  }

  // Specific API methods for 2K Spark endpoints
  async getPredictions() {
    return this.get('/api/predictions');
  }

  async getScorePredictions() {
    return this.get('/api/score-predictions');
  }

  async getStats() {
    return this.get('/api/stats');
  }

  async getPlayerStats() {
    return this.get('/api/player-stats');
  }

  async getPredictionHistory(params?: {
    limit?: number;
    offset?: number;
    date_from?: string;
    date_to?: string;
  }) {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString());
        }
      });
    }
    
    const endpoint = `/api/prediction-history${searchParams.toString() ? `?${searchParams.toString()}` : ''}`;
    return this.get(endpoint);
  }

  async getUpcomingMatches() {
    return this.get('/api/upcoming-matches');
  }

  async getLiveScores() {
    return this.get('/api/live-scores');
  }

  async getLiveMatchesWithPredictions() {
    return this.get('/api/live-matches-with-predictions');
  }

  async refreshData() {
    return this.post('/api/refresh');
  }

  async getRefreshStatus() {
    return this.get('/api/refresh/status');
  }
}

// Export singleton instance
export const apiClient = new ApiClient();

// Export the class for testing or custom instances
export { ApiClient };
