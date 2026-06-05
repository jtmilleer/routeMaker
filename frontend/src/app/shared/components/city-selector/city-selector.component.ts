// LAYER: Component — City/Start Location Selector
// PURPOSE: Dropdown for the 3 pre-seeded cities, plus a "Drop custom pin" mode
//          that shows the Leaflet map for location picking. On custom selection,
//          calls requestGraph() and transitions to the map-building-banner state.

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouteApiService, PresetCity } from '../../../core/services/route-api.service';
import { RouteStateService, ActiveCity } from '../../../core/services/route-state.service';

const CITY_DISPLAY_NAMES: Record<string, string> = {
  iowa_city: 'Iowa City, IA',
  madison: 'Madison, WI',
  des_moines: 'Des Moines, IA',
};

@Component({
  selector: 'app-city-selector',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="city-panel">
      <label class="label">Start Location</label>
      <div class="city-grid">
        <button
          *ngFor="let city of presets"
          class="city-btn"
          [class.active]="activeKey === city.city_key"
          [class.building]="city.status === 'building'"
          (click)="selectPreset(city)"
          [id]="'city-' + city.name"
          [disabled]="city.status === 'building'"
        >
          <span class="city-icon">{{ cityIcon(city.name) }}</span>
          <span class="city-name">{{ displayName(city.name) }}</span>
          <span class="city-status" *ngIf="city.status !== 'ready'">⏳</span>
        </button>

        <button
          class="city-btn custom-btn"
          [class.active]="customMode"
          (click)="toggleCustomMode()"
          id="city-custom-pin"
        >
          <span class="city-icon">📍</span>
          <span class="city-name">Custom Pin</span>
        </button>
      </div>

      <div class="custom-inputs" *ngIf="customMode">
        <input
          id="custom-lat-input"
          type="number" placeholder="Latitude" step="0.0001"
          class="coord-input" [(ngModel)]="customLat"
        >
        <input
          id="custom-lng-input"
          type="number" placeholder="Longitude" step="0.0001"
          class="coord-input" [(ngModel)]="customLng"
        >
        <button class="set-btn" (click)="setCustomPin()" id="set-custom-pin-btn">Set Location</button>
        <p class="hint">Tip: right-click anywhere on Google Maps to copy coordinates.</p>
      </div>
    </div>
  `,
  styles: [`
    .city-panel { display: flex; flex-direction: column; gap: 0.5rem; }
    .label { font-size: 0.875rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #6b7280; }
    .city-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.375rem; }
    .city-btn {
      display: flex; align-items: center; gap: 0.375rem;
      background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
      border-radius: 8px; color: #9ca3af; padding: 0.5rem 0.625rem; font-size: 0.78rem;
      cursor: pointer; transition: all 0.15s; text-align: left;
    }
    .city-btn:hover:not(:disabled) { background: rgba(255,255,255,0.09); color: #d1d5db; }
    .city-btn.active { background: rgba(252,76,2,0.12); border-color: rgba(252,76,2,0.4); color: #fc4c02; font-weight: 600; }
    .city-btn.building { opacity: 0.6; cursor: wait; }
    .city-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .city-icon { font-size: 1rem; }
    .city-name { flex: 1; }
    .custom-btn { grid-column: 1 / -1; }

    .custom-inputs { display: flex; flex-direction: column; gap: 0.375rem; }
    .coord-input {
      width: 100%; padding: 0.5rem 0.75rem;
      background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12);
      border-radius: 8px; color: #e5e7eb; font-size: 0.85rem;
    }
    .coord-input:focus { outline: none; border-color: #fc4c02; }
    .set-btn {
      padding: 0.5rem; background: rgba(252,76,2,0.15); border: 1px solid rgba(252,76,2,0.4);
      border-radius: 8px; color: #fc4c02; font-size: 0.82rem; font-weight: 600; cursor: pointer;
    }
    .hint { font-size: 0.7rem; color: #6b7280; margin: 0; }
  `]
})
export class CitySelectorComponent implements OnInit {
  presets: PresetCity[] = [];
  activeKey = '';
  customMode = false;
  customLat = 41.6543;
  customLng = -91.5267;

  constructor(private api: RouteApiService, public state: RouteStateService) {}

  ngOnInit(): void {
    this.api.getPresetCities().subscribe({
      next: cities => {
        this.presets = cities;
        // Auto-select Iowa City if ready
        const iowa = cities.find(c => c.name === 'iowa_city' && c.status === 'ready');
        if (iowa) this.selectPreset(iowa);
      }
    });
  }

  selectPreset(city: PresetCity): void {
    this.activeKey = city.city_key;
    this.customMode = false;
    this.state.setActiveCity({
      lat: city.lat,
      lng: city.lng,
      city_key: city.city_key,
      display_name: this.displayName(city.name),
    });
  }

  toggleCustomMode(): void {
    this.customMode = !this.customMode;
    if (this.customMode) this.activeKey = '';
  }

  setCustomPin(): void {
    if (!this.customLat || !this.customLng) return;

    // Request graph build and start polling
    this.api.requestGraph(this.customLat, this.customLng).subscribe({
      next: ({ city_key }) => {
        this.activeKey = city_key;
        this.state.setActiveCity({
          lat: this.customLat,
          lng: this.customLng,
          city_key,
          display_name: `${this.customLat.toFixed(3)}, ${this.customLng.toFixed(3)}`,
        });
        this.state.setGraphBuilding(true);
        this.customMode = false;
      }
    });
  }

  displayName(name: string): string {
    return CITY_DISPLAY_NAMES[name] ?? name;
  }

  cityIcon(name: string): string {
    const icons: Record<string, string> = {
      iowa_city: '🌽', madison: '🧀', des_moines: '🌪️',
    };
    return icons[name] ?? '🏙️';
  }
}
