// LAYER: Service — Backend API Client
// PURPOSE: Single source of truth for all HTTP calls to the FastAPI backend.
//          Components never call HttpClient directly — they go through this service.
//          The authInterceptor automatically attaches the JWT to every request here.

import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface GenerateRequest {
  route_type: 'regular' | 'hilly' | 'historic' | 'novel' | 'coverage';
  distance_mi: number;
  tolerance: number;
  start_lat: number;
  start_lng: number;
  hilly_factor?: number;
  downtown_radius?: number;
  novelty_factor?: number;
}

export interface HistoricSite {
  name: string;
  lat: number;
  lng: number;
}

export interface RouteResult {
  id: string;
  polyline: string;
  route_segments?: string;   // JSON string, only for novel routes
  distance_mi: number;
  elevation_ft: number;
  predicted_score: number;
  novelty_pct?: number;
  historic_sites?: HistoricSite[];
  elevation_profile?: number[][];   // [distance_mi, elevation_ft] points
  city_key: string;
  route_type: string;
  user_rating?: number | null;   // Joined from route_feedback; null if not yet rated
}

export interface Ride {
  id: number;
  name: string;
  date: string | null;
  distance_mi: number | null;
  elevation_ft: number | null;
  moving_time_min: number | null;
  avg_speed_mph: number | null;
  polyline: string | null;
  user_rating: number | null;
}

export interface SyncSummary {
  new_rides: number;
  total_rides: number;
}

export interface RatingStats {
  total_ratings: number;
  model_version: number;
  ratings_until_next_retrain: number;
  model_trained_at: string | null;
}

export interface PendingCounts {
  unrated_rides: number;
  unrated_routes: number;
}

export interface GraphStatus {
  city_key: string;
  status: 'ready' | 'building' | 'not_found' | 'error';
  progress_pct: number;
  node_count: number | null;
  edge_count: number | null;
}

export interface PresetCity {
  name: string;
  lat: number;
  lng: number;
  city_key: string;
  status: string;
}

export interface CoverageSegment {
  polyline: string;
  ridden: boolean;
}

export type CoverageActivity = 'walk' | 'ride' | 'both';

export interface CoverageResponse {
  city_key: string;
  center_lat: number;
  center_lng: number;
  radius_mi: number;
  activity: CoverageActivity;
  coverage_pct: number;
  total_mi: number;
  ridden_mi: number;
  segments: CoverageSegment[];
}

// ── Service ───────────────────────────────────────────────────────────────────

@Injectable({ providedIn: 'root' })
export class RouteApiService {
  private base = environment.apiUrl;

  constructor(private http: HttpClient) {}

  // Route generation
  generateRoutes(params: GenerateRequest): Observable<RouteResult[]> {
    return this.http.post<RouteResult[]>(`${this.base}/api/routes/generate`, params);
  }

  downloadGpx(routeId: string): Observable<Blob> {
    return this.http.get(`${this.base}/api/routes/gpx/${routeId}`, { responseType: 'blob' });
  }

  getRouteHistory(): Observable<RouteResult[]> {
    return this.http.get<RouteResult[]>(`${this.base}/api/routes/history`);
  }

  // Rides
  syncRides(): Observable<SyncSummary> {
    return this.http.post<SyncSummary>(`${this.base}/api/rides/sync`, {});
  }

  getRides(): Observable<Ride[]> {
    return this.http.get<Ride[]>(`${this.base}/api/rides`);
  }

  // Ratings
  rateRide(rideId: number, rating: number): Observable<RatingStats> {
    return this.http.post<RatingStats>(`${this.base}/api/ratings/ride`, { ride_id: rideId, rating });
  }

  rateGeneratedRoute(routeId: string, rating: number): Observable<RatingStats> {
    return this.http.post<RatingStats>(`${this.base}/api/ratings/route`, { route_id: routeId, rating });
  }

  getRatingStats(): Observable<RatingStats> {
    return this.http.get<RatingStats>(`${this.base}/api/ratings/stats`);
  }

  getPendingCounts(): Observable<PendingCounts> {
    return this.http.get<PendingCounts>(`${this.base}/api/ratings/pending-counts`);
  }

  // Graph / cities
  getGraphStatus(cityKey: string): Observable<GraphStatus> {
    return this.http.get<GraphStatus>(`${this.base}/api/graph/status/${cityKey}`);
  }

  requestGraph(lat: number, lng: number): Observable<{ city_key: string }> {
    return this.http.post<{ city_key: string }>(`${this.base}/api/graph/request`, { lat, lng });
  }

  getPresetCities(): Observable<PresetCity[]> {
    return this.http.get<PresetCity[]>(`${this.base}/api/graph/presets`);
  }

  // Coverage
  setHomeBase(lat: number, lng: number): Observable<{ home_lat: number; home_lng: number; city_key: string }> {
    return this.http.put<{ home_lat: number; home_lng: number; city_key: string }>(
      `${this.base}/api/coverage/home`, { lat, lng });
  }

  getCoverage(radiusMi: number, activity: CoverageActivity): Observable<CoverageResponse> {
    return this.http.get<CoverageResponse>(`${this.base}/api/coverage`, {
      params: { radius_mi: radiusMi, activity },
    });
  }

  generateCoverageRoute(radiusMi: number, activity: CoverageActivity, distanceMi: number): Observable<RouteResult[]> {
    return this.http.post<RouteResult[]>(`${this.base}/api/coverage/route`, {
      radius_mi: radiusMi, activity, distance_mi: distanceMi,
    });
  }
}
