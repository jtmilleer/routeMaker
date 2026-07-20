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
import { decodePolyline } from '../../shared/utils/polyline';

// Fix Leaflet default marker icon path issue with bundlers
const iconDefault = L.icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41], iconAnchor: [12, 41],
});
L.Marker.prototype.options.icon = iconDefault;

const ROUTE_COLORS = ['#c8915a', '#7a9a6d', '#a67c52', '#6b8f71', '#b8976a'];

@Component({
  selector: 'app-map-view',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="map-wrap">
      <div #mapEl id="leaflet-map" class="map"></div>
      <div class="idle-state" *ngIf="!hasRoutes && !(state.loading$ | async)">
        <h2>Pick a starting point</h2>
        <p>Select a city, configure your route, and generate.</p>
      </div>
    </div>
  `,
  styles: [`
    .map-wrap { position: relative; width: 100%; height: 100%; }
    .map { width: 100%; height: 100%; z-index: 1; }
    .idle-state {
      position: absolute; inset: 0; z-index: 2;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      background: rgba(18, 26, 19, 0.85); backdrop-filter: blur(4px);
      pointer-events: none;
      color: var(--text-muted); text-align: center;
    }
    .idle-state h2 { color: var(--text-primary); margin: 0 0 0.5rem; font-size: 1.5rem; }
    .idle-state p { margin: 0; font-size: 0.95rem; }
  `]
})
export class MapViewComponent implements AfterViewInit, OnDestroy {
  @ViewChild('mapEl', { static: true }) mapEl!: ElementRef<HTMLDivElement>;

  private map!: L.Map;
  private routeLayers: { route: RouteResult; layers: L.Polyline[] }[] = [];
  private siteMarkers: L.Marker[] = [];
  private startMarker?: L.Marker;
  private subs = new Subscription();
  hasRoutes = false;

  constructor(public state: RouteStateService) {}

  ngAfterViewInit(): void {
    this.map = L.map(this.mapEl.nativeElement, {
      center: [41.6543, -91.5267],
      zoom: 11,
      zoomControl: true,
    });

    // Dark-style tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
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
    this.routeLayers.forEach(rl => rl.layers.forEach(l => this.map.removeLayer(l)));
    this.routeLayers = [];
    this.siteMarkers.forEach(m => this.map.removeLayer(m));
    this.siteMarkers = [];
    this.hasRoutes = routes.length > 0;

    const historicIcon = L.divIcon({
      className: 'historic-marker',
      html: '<div style="background:#6b8f71;color:#fff;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-size:14px;border:2px solid #c4bfb4;box-shadow:0 2px 6px rgba(0,0,0,0.4);">&#9733;</div>',
      iconSize: [24, 24],
      iconAnchor: [12, 12],
    });
    routes.forEach(route => {
      if (route.historic_sites) {
        route.historic_sites.forEach(site => {
          const marker = L.marker([site.lat, site.lng], { icon: historicIcon })
            .addTo(this.map)
            .bindTooltip(site.name, { permanent: false, direction: 'top', offset: [0, -14] });
          this.siteMarkers.push(marker);
        });
      }
    });

    routes.forEach((route, i) => {
      const color = ROUTE_COLORS[i % ROUTE_COLORS.length];
      const layers: L.Polyline[] = [];
      const tooltip = `#${i + 1} | ${route.predicted_score}/10 | ${route.distance_mi}mi | ${route.elevation_ft}ft`;

      if (route.route_segments) {
        try {
          const segments: { polyline: string; is_new: boolean }[] = JSON.parse(route.route_segments);
          segments.forEach(seg => {
            const points = decodePolyline(seg.polyline);
            const layer = L.polyline(points, {
              color: seg.is_new ? color : '#555',
              weight: seg.is_new ? 5 : 2,
              opacity: seg.is_new ? 0.9 : 0.5,
            }).addTo(this.map);
            layer.bindTooltip(tooltip);
            layer.on('click', () => this.state.selectRoute(route));
            layers.push(layer);
          });
        } catch {
          layers.push(this._makePolyline(route, i, color, tooltip));
        }
      } else {
        layers.push(this._makePolyline(route, i, color, tooltip));
      }

      this.routeLayers.push({ route, layers });
    });

    if (routes.length > 0 && routes[0].polyline) {
      const points = decodePolyline(routes[0].polyline);
      if (points.length) {
        this.map.fitBounds(L.latLngBounds(points), { padding: [20, 20], animate: true });
      }
    }
  }

  private _makePolyline(route: RouteResult, i: number, color: string, tooltip: string): L.Polyline {
    const points = decodePolyline(route.polyline);
    const layer = L.polyline(points, { color, weight: 4, opacity: 0.85 }).addTo(this.map);
    layer.bindTooltip(tooltip);
    layer.on('click', () => this.state.selectRoute(route));
    return layer;
  }

  private highlightRoute(selected: RouteResult | null): void {
    this.routeLayers.forEach(rl => {
      const isSelected = selected && rl.route.id === selected.id;
      rl.layers.forEach(layer => {
        layer.setStyle({
          opacity: isSelected || !selected ? 0.85 : 0.15,
          weight: isSelected ? 6 : (selected ? 2 : 4),
        });
        if (isSelected) {
          layer.bringToFront();
        }
      });
    });

    if (selected) {
      const points = decodePolyline(selected.polyline);
      if (points.length) {
        this.map.fitBounds(L.latLngBounds(points), { padding: [30, 30], animate: true });
      }
    }
  }
}
