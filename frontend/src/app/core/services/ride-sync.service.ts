// LAYER: Service — Strava Sync
// PURPOSE: One shared Strava sync for the whole app. AppShellComponent runs it
//          once on load; the "Sync from Strava" buttons on the rate and coverage
//          pages call it too, and a click while a sync is in flight is ignored
//          instead of starting a second one. Pages subscribe to `synced$` to
//          reload their data when a sync brings in new activities.

import { Injectable, computed, signal } from '@angular/core';
import { Subject } from 'rxjs';
import { RouteApiService, SyncSummary } from './route-api.service';
import { PendingRatingsService } from './pending-ratings.service';

@Injectable({ providedIn: 'root' })
export class RideSyncService {
  readonly syncing = signal(false);
  private readonly lastSync = signal<SyncSummary | null>(null);
  private readonly failed = signal(false);

  /** Emits after each successful sync that added at least one activity. */
  readonly synced$ = new Subject<SyncSummary>();

  readonly message = computed(() => {
    if (this.failed()) return 'Sync failed — try again.';
    const s = this.lastSync();
    return s ? `${s.new_rides} new activities synced (${s.total_rides} total)` : '';
  });

  constructor(private api: RouteApiService, private pending: PendingRatingsService) {}

  sync(): void {
    if (this.syncing()) return;
    this.syncing.set(true);
    this.failed.set(false);
    this.api.syncRides().subscribe({
      next: s => {
        this.syncing.set(false);
        this.lastSync.set(s);
        if (s.new_rides > 0) {
          this.pending.refresh();
          this.synced$.next(s);
        }
      },
      error: () => {
        this.syncing.set(false);
        this.failed.set(true);
      },
    });
  }
}
