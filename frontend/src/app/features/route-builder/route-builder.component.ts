// LAYER: Component — Route Builder (Main App Page)
// PURPOSE: Assembles the city-selector, route-form sidebar, map-view, and
//          route-results panel for the main route generation experience.
//          Nav/chrome now lives in AppShellComponent — this component only
//          owns the builder's own layout.

import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouteStateService } from '../../core/services/route-state.service';
import { CitySelectorComponent } from '../../shared/components/city-selector/city-selector.component';
import { RouteFormComponent } from './route-form/route-form.component';
import { RouteResultsComponent } from './route-results/route-results.component';
import { MapViewComponent } from '../map-view/map-view.component';
import { LoadingOverlayComponent } from '../../shared/components/loading-overlay/loading-overlay.component';

@Component({
  selector: 'app-route-builder',
  standalone: true,
  imports: [
    CommonModule,
    CitySelectorComponent,
    RouteFormComponent,
    RouteResultsComponent,
    MapViewComponent,
    LoadingOverlayComponent,
  ],
  template: `
    <div class="main-layout">
      <aside class="sidebar">
        <app-city-selector></app-city-selector>
        <app-route-form></app-route-form>
        <app-route-results></app-route-results>
      </aside>

      <main class="map-area">
        <app-loading-overlay *ngIf="state.loading$ | async"></app-loading-overlay>
        <app-map-view></app-map-view>
      </main>
    </div>
  `,
  styles: [`
    :host { display: flex; flex: 1; min-height: 0; overflow: hidden; }
    .main-layout { display: flex; flex: 1; overflow: hidden; }

    .sidebar {
      width: 320px; min-width: 320px;
      background: rgba(60, 46, 30, 0.03);
      border-right: 1px solid var(--border);
      display: flex; flex-direction: column; overflow-y: auto;
      padding: 1rem;
      gap: 1rem;
    }
    .map-area { flex: 1; position: relative; }
  `]
})
export class RouteBuilderComponent {
  constructor(public state: RouteStateService) {}
}
