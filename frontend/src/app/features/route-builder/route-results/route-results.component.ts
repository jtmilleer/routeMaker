// LAYER: Component — Route Results List
// PURPOSE: Displays the list of generated route cards in the sidebar.
//          Clicking a card selects that route (highlighted on the map).

import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouteStateService } from '../../../core/services/route-state.service';
import { RouteCardComponent } from '../../../shared/components/route-card/route-card.component';
import { RouteResult } from '../../../core/services/route-api.service';

@Component({
  selector: 'app-route-results',
  standalone: true,
  imports: [CommonModule, RouteCardComponent],
  template: `
    <div class="results" *ngIf="(state.routes$ | async)?.length">
      <h3 class="section-title">Generated Routes</h3>
      <app-route-card
        *ngFor="let route of (state.routes$ | async); let i = index"
        [route]="route"
        [rank]="i + 1"
        [selected]="(state.selectedRoute$ | async)?.id === route.id"
        (click)="state.selectRoute(route)"
      ></app-route-card>
    </div>
  `,
  styles: [`
    .results { display: flex; flex-direction: column; gap: 0.5rem; }
    .section-title { font-size: 0.875rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-dim); margin: 0 0 0.25rem; }
  `]
})
export class RouteResultsComponent {
  constructor(public state: RouteStateService) {}
}
