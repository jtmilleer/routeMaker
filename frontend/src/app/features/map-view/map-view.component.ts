// LAYER: Component — Interactive Leaflet Map
// PURPOSE: Renders the Leaflet map and draws polylines for each generated route.
//          Watches route-state.service for changes and redraws accordingly.
//          Novel routes use route_segments to color ridden vs new roads differently.
//          Supports a draggable start pin for custom location selection.

import {
  Component, OnInit, OnDestroy, AfterViewInit,
  ElementRef, ViewChild
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';
import * as L from 'leaflet';

import { RouteStateService, ActiveCity } from '../../core/services/route-state.service';
import { RouteResult } from '../../core/services/route-api.service';

// Fix Leaflet default marker icon path issue with bundlers
const iconDefault = L.icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41], iconAnchor: [12, 41],
});
L.Marker.prototype.options.icon = iconDefault;

const ROUTE_COLORS = ['#fc4c02', '#3b82f6', '#8b5cf6', '#10b981', '#f59e0b'];

// Decode a Google encoded polyline string to [lat, lng] pairs
function decodePolyline(encoded: string): [number, number][] {
  const points: [number, number][] = [];
  let idx = 0, lat = 0, lng = 0;
  while (idx < encoded.length) {
    let b, shift = 0, result = 0;
    do { b = encoded.charCodeAt(idx++) - 63; result |= (b & 0x1f) << shift; shift += 5; } while (b >= 0x20);
    lat += (result & 1) ? ~(result >> 1) : (result >> 1);
    shift = 0; result = 0;
    do { b = encoded.charCodeAt(idx++) - 63; result |= (b & 0x1f) << shift; shift += 5; } while (b >= 0x20);
    lng += (result & 1) ? ~(result >> 1) : (result >> 1);
    points.push([lat / 1e5, lng / 1e5]);
  }
  return points;
}

@Component({
  selector: 'app-map-view',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="map-wrap">
      <div #mapEl id="leaflet-map" class="map"></div>
      <div class="idle-state" *ngIf="!(hasRoutes)">
        <div class="idle-icon">🚴</div>
        <h2>Ready to Explore</h2>
        <p>Select a city, configure your route, and hit Generate.</p>
      </div>
    </div>
  `,
  styles: [`
    .map-wrap { position: relative; width: 100%; height: 100%; }
    .map { width: 100%; height: 100%; z-index: 1; }
    .idle-state {
      position: absolute; inset: 0; z-index: 2;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      background: rgba(15,15,26,0.85); backdrop-filter: blur(4px);
      pointer-events: none;
      color: #9ca3af; text-align: center;
    }
    .idle-icon { font-size: 3.5rem; margin-bottom: 1rem; }
    .idle-state h2 { color: #fff; margin: 0 0 0.5rem; font-size: 1.5rem; }
    .idle-state p { margin: 0; font-size: 0.95rem; }
  `]
})
export class MapViewComponent implements AfterViewInit, OnDestroy {
  @ViewChild('mapEl', { static: true }) mapEl!: ElementRef<HTMLDivElement>;

  private map!: L.Map;
  private routeLayers: L.Layer[] = [];
  private startMarker?: L.Marker;
  private subs = new Subscription();
  hasRoutes = false;

  constructor(private state: RouteStateService) {}

  ngAfterViewInit(): void {
    this.map = L.map(this.mapEl.nativeElement, {
      center: [41.6543, -91.5267],
      zoom: 11,
      zoomControl: true,
    });

    // Dark-style tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '© OpenStreetMap contributors © CARTO',
      maxZoom: 19,
    }).addTo(this.map);

    // Watch routes changes
    this.subs.add(
      this.state.routes$.subscribe(routes => this.renderRoutes(routes))
    );

    // Watch city changes to center map
    this.subs.add(
      this.state.activeCity$.subscribe(city => {
        if (city) this.centerOnCity(city);
      })
    );

    // Watch selected route to highlight it
    this.subs.add(
      this.state.selectedRoute$.subscribe(route => this.highlightRoute(route))
    );
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
    this.map?.remove();
  }

  private centerOnCity(city: ActiveCity): void {
    this.map.setView([city.lat, city.lng], 11, { animate: true });

    // Update/create start marker
    if (this.startMarker) {
      this.startMarker.setLatLng([city.lat, city.lng]);
    } else {
      this.startMarker = L.marker([city.lat, city.lng], {
        title: 'Start Location',
        draggable: false,
      }).addTo(this.map).bindPopup('Start Location');
    }
  }

  private renderRoutes(routes: RouteResult[]): void {
    // Clear existing layers
    this.routeLayers.forEach(l => this.map.removeLayer(l));
    this.routeLayers = [];
    this.hasRoutes = routes.length > 0;

    routes.forEach((route, i) => {
      const color = ROUTE_COLORS[i % ROUTE_COLORS.length];

      if (route.route_segments) {
        // Novel routes: color ridden vs new sections differently
        try {
          const segments: { polyline: string; is_new: boolean }[] = JSON.parse(route.route_segments);
          segments.forEach(seg => {
            const points = decodePolyline(seg.polyline);
            const layer = L.polyline(points, {
              color: seg.is_new ? color : '#555',
              weight: seg.is_new ? 5 : 2,
              opacity: seg.is_new ? 0.9 : 0.5,
            }).addTo(this.map);
            layer.bindTooltip(`#${i + 1} | ${route.predicted_score}/10 | ${route.distance_mi}mi | ${route.elevation_ft}ft`);
            layer.on('click', () => this.state.selectRoute(route));
            this.routeLayers.push(layer);
          });
        } catch {
          this._addSimplePolyline(route, i, color);
        }
      } else {
        this._addSimplePolyline(route, i, color);
      }
    });

    // Fit map to first route
    if (routes.length > 0 && routes[0].polyline) {
      const points = decodePolyline(routes[0].polyline);
      if (points.length) {
        this.map.fitBounds(L.latLngBounds(points), { padding: [20, 20], animate: true });
      }
    }
  }

  private _addSimplePolyline(route: RouteResult, i: number, color: string): void {
    const points = decodePolyline(route.polyline);
    const layer = L.polyline(points, { color, weight: 4, opacity: 0.85 }).addTo(this.map);
    layer.bindTooltip(`#${i + 1} | ${route.predicted_score}/10 | ${route.distance_mi}mi | ${route.elevation_ft}ft`);
    layer.on('click', () => this.state.selectRoute(route));
    this.routeLayers.push(layer);
  }

  private highlightRoute(selected: RouteResult | null): void {
    // Visual highlight is handled by route card; map re-renders on route change
  }
}
