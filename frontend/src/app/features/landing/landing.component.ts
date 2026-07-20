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
        <div class="eyebrow">⟡ Field Notes · Custom Routes ⟡</div>
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
      background: radial-gradient(ellipse at 70% 15%, #e8ddc0 0%, var(--bg-primary) 55%);
      padding: 2rem;
    }
    .hero {
      text-align: center;
      max-width: 520px;
    }
    .eyebrow {
      font-family: var(--font-mono);
      font-size: 0.7rem;
      font-weight: 600;
      letter-spacing: 0.15em;
      color: var(--text-muted);
      margin-bottom: 0.875rem;
    }
    .wordmark {
      font-size: 3.5rem;
      font-weight: 900;
      color: var(--text-primary);
      letter-spacing: -1px;
      margin-bottom: 0.75rem;
      font-family: var(--font-display);
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
      border-radius: 3px;
      padding: 0.875rem 1.25rem;
      color: var(--text-secondary);
      font-size: 0.95rem;
    }
    .feature-marker {
      font-family: var(--font-mono);
      font-weight: 700;
      font-size: 0.7rem;
      color: var(--accent);
      flex-shrink: 0;
    }
    .feature-marker::before { content: '✕'; }
    .strava-btn {
      display: inline-flex;
      align-items: center;
      gap: 0.625rem;
      background: var(--strava-orange);
      color: #fff;
      border: none;
      border-radius: 4px;
      padding: 1rem 2rem;
      font-size: 1.05rem;
      font-weight: 700;
      font-family: var(--font-primary);
      cursor: pointer;
      transition: transform 0.15s, box-shadow 0.15s;
      box-shadow: 3px 3px 0 rgba(60, 46, 30, 0.25);
      width: 100%;
      justify-content: center;
    }
    .strava-btn:hover:not(:disabled) {
      transform: translate(-1px, -1px);
      box-shadow: 4px 4px 0 rgba(60, 46, 30, 0.3);
    }
    .strava-btn:disabled { opacity: 0.6; cursor: not-allowed; }
    .auth-error {
      background: rgba(178, 58, 46, 0.1);
      border: 1px solid rgba(178, 58, 46, 0.35);
      color: #8a3626;
      border-radius: 4px;
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
