// LAYER: Service — Pending Ratings Count
// PURPOSE: Tracks how many rides + generated routes are still unrated, for the
//          "Rate" nav badge. Fetched once by AppShellComponent; decremented
//          locally by the rate sub-components on a successful (not skipped)
//          rating so the badge updates without a refetch.

import { Injectable, signal } from '@angular/core';
import { RouteApiService } from './route-api.service';

@Injectable({ providedIn: 'root' })
export class PendingRatingsService {
  readonly pendingCount = signal<number | null>(null);

  constructor(private api: RouteApiService) {}

  refresh(): void {
    this.api.getPendingCounts().subscribe({
      next: counts => this.pendingCount.set(counts.unrated_rides + counts.unrated_routes),
      error: () => { /* non-critical — badge just stays hidden */ },
    });
  }

  decrement(): void {
    const current = this.pendingCount();
    if (current != null && current > 0) {
      this.pendingCount.set(current - 1);
    }
  }
}
