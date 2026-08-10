// LAYER: Component — Rate Section Shell
// PURPOSE: Owns the Rides/Generated sub-nav for the merged "Rate" section and
//          hosts the router-outlet for its two child views.

import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-rate-shell',
  standalone: true,
  imports: [CommonModule, RouterLink, RouterLinkActive, RouterOutlet],
  template: `
    <div class="rate-shell">
      <header class="rate-header">
        <h1 class="rate-title">Rate</h1>
        <div class="sub-nav">
          <a routerLink="rides" routerLinkActive="active" class="sub-tab" id="rate-tab-rides">Rides</a>
          <a routerLink="generated" routerLinkActive="active" class="sub-tab" id="rate-tab-generated">Generated Routes</a>
        </div>
      </header>
      <div class="rate-body">
        <router-outlet></router-outlet>
      </div>
    </div>
  `,
  styles: [`
    :host { display: flex; flex-direction: column; flex: 1; min-height: 0; overflow: hidden; }
    .rate-shell { display: flex; flex-direction: column; height: 100%; }
    .rate-header {
      display: flex; align-items: center; gap: 1.5rem;
      padding: 1.25rem 2rem 0;
    }
    .rate-title { font-family: var(--font-display); font-size: 1.5rem; font-weight: 800; margin: 0; color: var(--text-primary); }
    .sub-nav { display: flex; gap: 0.5rem; border-bottom: 1px solid var(--border); flex: 1; }
    .sub-tab {
      padding: 0.6rem 1rem; text-decoration: none; color: var(--text-muted);
      font-size: 0.9rem; font-weight: 600; border-bottom: 2px solid transparent;
      margin-bottom: -1px;
    }
    .sub-tab:hover { color: var(--text-primary); }
    .sub-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
    .rate-body { flex: 1; min-height: 0; overflow-y: auto; }
  `]
})
export class RateShellComponent {}
