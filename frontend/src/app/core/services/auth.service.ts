// LAYER: Service — Authentication
// PURPOSE: Manages the JWT lifecycle in localStorage. Exposes the current
//          user as an observable. Redirects to the backend Strava OAuth init
//          endpoint to start the login flow. Reads the token from the URL
//          fragment after OAuth callback.

import { Injectable, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { firstValueFrom, timeout } from 'rxjs';
import { environment } from '../../../environments/environment';

export interface UserProfile {
  athlete_id: number;
  name: string;
  profile_pic_url: string | null;
  model_version: number;
  total_ratings: number;
  model_trained_at: string | null;
  home_lat: number | null;
  home_lng: number | null;
}

const TOKEN_KEY = 'rm_jwt';

@Injectable({ providedIn: 'root' })
export class AuthService {
  // Reactive signal — components read this directly in templates
  readonly currentUser = signal<UserProfile | null>(null);
  readonly isLoading = signal(true);

  constructor(private http: HttpClient, private router: Router) {}

  /** Call on app start — tries to load the user from a stored JWT. */
  async initialize(): Promise<void> {
    const token = this.getToken();
    if (!token) {
      this.isLoading.set(false);
      return;
    }
    try {
      const user = await firstValueFrom(
        this.http.get<UserProfile>(`${environment.apiUrl}/auth/me`).pipe(
          timeout(3000)  // Give up after 3s if backend is unreachable
        )
      );
      this.currentUser.set(user ?? null);
      // Slide the session window forward so an open app doesn't expire mid-use.
      this.refreshSession();
    } catch {
      // Token expired, invalid, or backend unreachable — clear and continue
      localStorage.removeItem(TOKEN_KEY);
      this.currentUser.set(null);
    } finally {
      this.isLoading.set(false);
    }
  }

  /** Quietly re-issue a fresh JWT so the session keeps rolling while in use. */
  private async refreshSession(): Promise<void> {
    try {
      const resp = await firstValueFrom(
        this.http.post<{ access_token: string }>(`${environment.apiUrl}/auth/refresh`, {}).pipe(
          timeout(3000)
        )
      );
      if (resp?.access_token) {
        localStorage.setItem(TOKEN_KEY, resp.access_token);
      }
    } catch {
      // Non-fatal — the existing token is still valid for now.
    }
  }

  /** Redirect browser to Strava OAuth — backend handles the rest. */
  async login(): Promise<void> {
    const resp = await firstValueFrom(
      this.http.get<{ url: string }>(`${environment.apiUrl}/auth/strava/init`)
    );
    if (resp?.url) {
      window.location.href = resp.url;
    }
  }

  /**
   * Called by the /auth/callback route after Strava redirects back.
   * Reads the token from the URL fragment (#token=...) and stores it.
   */
  async handleCallback(fragment: string): Promise<void> {
    const params = new URLSearchParams(fragment);
    const token = params.get('token');
    if (!token) {
      this.router.navigate(['/']);
      return;
    }
    localStorage.setItem(TOKEN_KEY, token);
    await this.initialize();
    this.router.navigate(['/app']);
  }

  logout(): void {
    localStorage.removeItem(TOKEN_KEY);
    this.currentUser.set(null);
    this.router.navigate(['/']);
  }

  getToken(): string | null {
    return localStorage.getItem(TOKEN_KEY);
  }

  isAuthenticated(): boolean {
    return !!this.getToken() && !!this.currentUser();
  }
}
