// LAYER: Component — App Shell
// PURPOSE: Persistent top-level chrome for every authenticated /app page —
//          nav (Builder/Rate/Coverage), user info, sign out, and the
//          graph-build banner. Hosts the router-outlet for the active section.

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { AuthService } from '../../../core/services/auth.service';
import { PendingRatingsService } from '../../../core/services/pending-ratings.service';
import { MapBuildingBannerComponent } from '../map-building-banner/map-building-banner.component';

@Component({
  selector: 'app-shell',
  standalone: true,
  imports: [CommonModule, RouterLink, RouterLinkActive, RouterOutlet, MapBuildingBannerComponent],
  template: `
    <div class="app-shell">
      <nav class="topnav">
        <div class="nav-brand">
          <span class="brand-name">RouteMaker</span>
        </div>
        <div class="nav-links">
          <a routerLink="/app" routerLinkActive="active" [routerLinkActiveOptions]="{ exact: true }" class="nav-link" id="nav-builder">Builder</a>
          <a routerLink="/app/rate" routerLinkActive="active" class="nav-link" id="nav-rate">
            Rate
            <span class="nav-badge" *ngIf="pending.pendingCount()">{{ pending.pendingCount() }}</span>
          </a>
          <a routerLink="/app/coverage" routerLinkActive="active" class="nav-link" id="nav-coverage">Coverage</a>
          <div class="nav-user" *ngIf="auth.currentUser() as user">
            <img *ngIf="user.profile_pic_url" [src]="user.profile_pic_url" class="avatar" alt="Profile">
            <span class="user-name">{{ user.name }}</span>
            <span class="model-badge" title="Preference model version">v{{ user.model_version }}</span>
          </div>
          <button class="logout-btn" (click)="auth.logout()" id="logout-btn">Sign Out</button>
        </div>
      </nav>

      <app-map-building-banner></app-map-building-banner>

      <router-outlet></router-outlet>
    </div>
  `,
  styles: [`
    :host { display: block; height: 100vh; overflow: hidden; }
    .app-shell { display: flex; flex-direction: column; height: 100vh; background: var(--bg-primary); color: var(--text-primary); font-family: var(--font-primary); }

    .topnav {
      display: flex; align-items: center; justify-content: space-between;
      padding: 0 1.5rem; height: 56px; min-height: 56px;
      background: rgba(224, 213, 184, 0.95);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(12px);
      z-index: 100;
    }
    .nav-brand { display: flex; align-items: center; gap: 0.5rem; }
    .brand-name { font-size: 1.25rem; font-weight: 700; color: var(--text-primary); font-family: var(--font-display); }

    .nav-links { display: flex; align-items: center; gap: 1rem; }
    .nav-link { color: var(--text-muted); text-decoration: none; font-size: 0.875rem; font-weight: 500; transition: color 0.15s; display: flex; align-items: center; }
    .nav-link:hover { color: var(--text-primary); }
    .nav-link.active { color: var(--accent); font-weight: 700; }
    .nav-badge {
      display: inline-block; margin-left: 0.4rem; background: var(--accent); color: var(--on-accent);
      font-size: 0.7rem; font-weight: 700; border-radius: 999px; padding: 0.05rem 0.4rem;
      font-family: var(--font-mono); line-height: 1.4;
    }
    .nav-user { display: flex; align-items: center; gap: 0.5rem; }
    .avatar { width: 28px; height: 28px; border-radius: 50%; }
    .user-name { font-size: 0.875rem; color: var(--text-secondary); }
    .model-badge { font-size: 0.7rem; font-family: var(--font-mono); background: var(--surface-active); color: var(--accent); border-radius: 3px; padding: 2px 6px; font-weight: 600; }
    .logout-btn { background: var(--bg-surface); border: 1px solid var(--border); border-radius: 4px; color: var(--text-muted); padding: 0.375rem 0.75rem; font-size: 0.8rem; font-family: var(--font-primary); cursor: pointer; transition: all 0.15s; }
    .logout-btn:hover { background: var(--surface-hover); color: var(--text-primary); }
  `]
})
export class AppShellComponent implements OnInit {
  constructor(public auth: AuthService, public pending: PendingRatingsService) {}

  ngOnInit(): void {
    this.pending.refresh();
  }
}
