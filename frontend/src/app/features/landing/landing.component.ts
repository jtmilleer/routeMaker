// LAYER: Component — Landing / Login Page
// PURPOSE: Shown to unauthenticated users. Displays the app pitch and a
//          "Connect with Strava" button. Also handles the OAuth callback
//          route (/auth/callback) — reads the JWT from the URL fragment.

import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { AuthService } from '../../core/services/auth.service';

const AUTH_ERROR_MESSAGES: Record<string, string> = {
  denied: 'Strava authorization was cancelled. Connect to continue.',
  access_denied: 'Strava authorization was cancelled. Connect to continue.',
  invalid_state: 'Your login session expired. Please try connecting again.',
  not_authorized: 'This app is in private beta and your Strava account is not on the list yet.',
  no_athlete: 'Could not read your Strava profile. Please try again.',
};

@Component({
  selector: 'app-landing',
  standalone: true,
  template: `
    <div class="landing">
      <div class="hero">
        <div class="wordmark">RouteMaker</div>
        <p class="tagline">Custom cycling routes, built from your ride history.</p>

        <div class="features">
          <div class="feature">
            <span class="feature-marker"></span>
            <span>Lollipop loops, hilly climbs, historic tours, novel roads</span>
          </div>
          <div class="feature">
            <span class="feature-marker"></span>
            <span>Routes improve as you ride and rate</span>
          </div>
          <div class="feature">
            <span class="feature-marker"></span>
            <span>Iowa City, Madison, Des Moines — or drop a custom pin</span>
          </div>
        </div>

        @if (errorMessage) {
          <div class="auth-error">{{ errorMessage }}</div>
        }

        <button
          id="connect-strava-btn"
          class="strava-btn"
          (click)="login()"
          [disabled]="loading"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
            <path d="M15.387 17.944l-2.089-4.116h-3.065L15.387 24l5.15-10.172h-3.066m-7.008-5.599l2.836 5.598h4.172L10.463 0l-7 13.828h4.917"/>
          </svg>
          {{ loading ? 'Connecting...' : 'Connect with Strava' }}
        </button>

        <p class="disclaimer">
          Only your activity data is used — never shared. Tokens are stored securely on your device.
        </p>
      </div>
    </div>
  `,
  styles: [`
    .landing {
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(145deg, var(--bg-primary) 0%, #182118 45%, #1a2418 100%);
      padding: 2rem;
    }
    .hero {
      text-align: center;
      max-width: 520px;
    }
    .wordmark {
      font-size: 3.5rem;
      font-weight: 800;
      color: var(--text-primary);
      letter-spacing: -1px;
      margin-bottom: 0.75rem;
      font-family: var(--font-primary);
    }
    .tagline {
      font-size: 1.15rem;
      color: var(--text-muted);
      margin-bottom: 2.5rem;
      line-height: 1.6;
    }
    .features {
      display: flex;
      flex-direction: column;
      gap: 0.875rem;
      margin-bottom: 2.5rem;
      text-align: left;
    }
    .feature {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0.875rem 1.25rem;
      color: var(--text-secondary);
      font-size: 0.95rem;
    }
    .feature-marker {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent);
      flex-shrink: 0;
    }
    .strava-btn {
      display: inline-flex;
      align-items: center;
      gap: 0.625rem;
      background: var(--strava-orange);
      color: #fff;
      border: none;
      border-radius: 12px;
      padding: 1rem 2rem;
      font-size: 1.05rem;
      font-weight: 700;
      font-family: var(--font-primary);
      cursor: pointer;
      transition: transform 0.15s, box-shadow 0.15s;
      box-shadow: 0 4px 24px rgba(252, 76, 2, 0.35);
      width: 100%;
      justify-content: center;
    }
    .strava-btn:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 8px 32px rgba(252, 76, 2, 0.5);
    }
    .strava-btn:disabled { opacity: 0.6; cursor: not-allowed; }
    .auth-error {
      background: rgba(200, 80, 60, 0.12);
      border: 1px solid rgba(200, 80, 60, 0.35);
      color: #e08a78;
      border-radius: 10px;
      padding: 0.75rem 1rem;
      margin-bottom: 1rem;
      font-size: 0.9rem;
    }
    .disclaimer {
      margin-top: 1.25rem;
      font-size: 0.8rem;
      color: var(--text-dim);
    }
  `]
})
export class LandingComponent implements OnInit {
  loading = false;
  errorMessage = '';

  constructor(private auth: AuthService, private route: ActivatedRoute) {}

  ngOnInit(): void {
    // Handle OAuth callback — JWT is in the URL fragment
    const fragment = this.route.snapshot.fragment ?? '';
    if (fragment.includes('token=')) {
      this.loading = true;
      this.auth.handleCallback(fragment);
      return;
    }

    // Handle OAuth failure — backend redirects to /?auth_error=<reason>
    const reason = this.route.snapshot.queryParamMap.get('auth_error');
    if (reason) {
      this.errorMessage = AUTH_ERROR_MESSAGES[reason] ?? 'Login failed. Please try again.';
    }
  }

  login(): void {
    this.loading = true;
    this.auth.login();
  }
}
