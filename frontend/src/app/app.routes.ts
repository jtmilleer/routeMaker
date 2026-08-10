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
  // Protected: main route builder
  {
    path: 'app',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./features/route-builder/route-builder.component').then(m => m.RouteBuilderComponent),
  },
  // Protected: rate your Strava rides
  {
    path: 'app/rate-rides',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./features/rate/rate-rides/rate-rides.component').then(m => m.RateRidesComponent),
  },
  // Protected: rate generated routes after riding them
  {
    path: 'app/rate-generated',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./features/rate-generated/rate-generated.component').then(m => m.RateGeneratedComponent),
  },
  // Protected: street coverage map
  {
    path: 'app/coverage',
    canActivate: [authGuard],
    loadComponent: () =>
      import('./features/coverage/coverage.component').then(m => m.CoverageComponent),
  },
  // Fallback
  { path: '**', redirectTo: '' },
];
