// LAYER: Service — UI State Store
// PURPOSE: RxJS BehaviorSubject-based state store. Components subscribe to these
//          observables instead of prop-drilling or duplicating state. This is the
//          single source of truth for routes, loading status, and selected city.

import { Injectable, signal } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { RouteResult } from './route-api.service';

export interface ActiveCity {
  lat: number;
  lng: number;
  city_key: string;
  display_name: string;
}

@Injectable({ providedIn: 'root' })
export class RouteStateService {
  // Current generated routes
  private _routes = new BehaviorSubject<RouteResult[]>([]);
  readonly routes$ = this._routes.asObservable();

  // Loading state (route generation in progress)
  private _loading = new BehaviorSubject<boolean>(false);
  readonly loading$ = this._loading.asObservable();

  // The route the user clicked on the map
  private _selectedRoute = new BehaviorSubject<RouteResult | null>(null);
  readonly selectedRoute$ = this._selectedRoute.asObservable();

  // The currently active city / start location
  private _activeCity = new BehaviorSubject<ActiveCity | null>(null);
  readonly activeCity$ = this._activeCity.asObservable();

  // Graph build status (for custom pin locations)
  private _graphBuilding = new BehaviorSubject<boolean>(false);
  readonly graphBuilding$ = this._graphBuilding.asObservable();

  setRoutes(routes: RouteResult[]): void {
    this._routes.next(routes);
    this._selectedRoute.next(routes[0] ?? null);
  }

  setLoading(loading: boolean): void {
    this._loading.next(loading);
  }

  selectRoute(route: RouteResult | null): void {
    this._selectedRoute.next(route);
  }

  setActiveCity(city: ActiveCity): void {
    this._activeCity.next(city);
  }

  setGraphBuilding(building: boolean): void {
    this._graphBuilding.next(building);
  }

  clearRoutes(): void {
    this._routes.next([]);
    this._selectedRoute.next(null);
  }
}
