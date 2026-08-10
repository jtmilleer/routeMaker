import { Routes } from '@angular/router';
import { authGuard } from './core/guards/auth.guard';

export const routes: Routes = [
  // Public: landing / login page
  {
    path: '',
    loadComponent: () =>
      import('./features/landing/landing.component').then(m => m.LandingComponent),
  },
  // OAuth callback — reads JWT from URL fragment and stores it
  {
    path: 'auth/callback',
    loadComponent: () =>
      import('./features/landing/landing.component').then(m => m.LandingComponent),
  },
  // Protected: app shell (persistent nav) + its three sections
  {
    path: 'app',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./shared/components/app-shell/app-shell.component').then(m => m.AppShellComponent),
    children: [
      // Main route builder (default landing page under /app)
      {
        path: '',
        loadComponent: () =>
          import('./features/route-builder/route-builder.component').then(m => m.RouteBuilderComponent),
      },
      // Rate section: sub-nav shell + Rides / Generated Routes children
      {
        path: 'rate',
        loadComponent: () =>
          import('./features/rate/rate-shell.component').then(m => m.RateShellComponent),
        children: [
          { path: '', redirectTo: 'rides', pathMatch: 'full' },
          {
            path: 'rides',
            loadComponent: () =>
              import('./features/rate/rate-rides/rate-rides.component').then(m => m.RateRidesComponent),
          },
          {
            path: 'generated',
            loadComponent: () =>
              import('./features/rate/rate-generated/rate-generated.component').then(m => m.RateGeneratedComponent),
          },
        ],
      },
      // Street coverage map
      {
        path: 'coverage',
        loadComponent: () =>
          import('./features/coverage/coverage.component').then(m => m.CoverageComponent),
      },
    ],
  },
  // Fallback
  { path: '**', redirectTo: '' },
];
