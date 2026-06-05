// LAYER: Component — App Shell
// PURPOSE: Root component. Renders the router-outlet that hosts all pages.
//          Handles the global loading state during auth initialization.

import { Component, inject } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { CommonModule } from '@angular/common';
import { AuthService } from './core/services/auth.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, CommonModule],
  template: `
    <!-- Global auth init loader — shown for < 1s on cold load -->
    <div class="init-loader" *ngIf="auth.isLoading()">
      <div class="init-spinner"></div>
    </div>
    <router-outlet *ngIf="!auth.isLoading()"></router-outlet>
  `,
  styles: [`
    .init-loader {
      display: flex; align-items: center; justify-content: center;
      height: 100vh; background: #0f0f1a;
    }
    .init-spinner {
      width: 40px; height: 40px;
      border: 3px solid rgba(252,76,2,0.2);
      border-top-color: #fc4c02;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  `]
})
export class App {
  auth = inject(AuthService);
}
