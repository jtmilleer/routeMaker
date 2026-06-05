// LAYER: Component — Landing / Login Page
// PURPOSE: Shown to unauthenticated users. Displays the app pitch and a
//          "Connect with Strava" button. Also handles the OAuth callback
//          route (/auth/callback) — reads the JWT from the URL fragment.

import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { AuthService } from '../../core/services/auth.service';

@Component({
  selector: 'app-landing',
  standalone: true,
  template: `
    <div class="landing">
      <div class="hero">
        <div class="logo">🚴</div>
        <h1 class="title">Route<span class="accent">Maker</span></h1>
        <p class="tagline">AI-powered cycling routes, personalized to your preferences.</p>

        <div class="features">
          <div class="feature">
            <span class="feature-icon">🗺️</span>
            <span>Lollipop loops, hilly climbs, historic tours, novel roads</span>
          </div>
          <div class="feature">
            <span class="feature-icon">🤖</span>
            <span>ML model learns from your Strava rides and ratings</span>
          </div>
          <div class="feature">
            <span class="feature-icon">📍</span>
            <span>Iowa City, Madison, Des Moines — or drop a custom pin</span>
          </div>
        </div>

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
      background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
      padding: 2rem;
    }
    .hero {
      text-align: center;
      max-width: 520px;
    }
    .logo { font-size: 4rem; margin-bottom: 0.5rem; }
    .title {
      font-size: 3.5rem;
      font-weight: 800;
      color: #fff;
      margin: 0 0 0.75rem;
      letter-spacing: -1px;
    }
    .accent { color: #fc4c02; }
    .tagline {
      font-size: 1.15rem;
      color: #9ca3af;
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
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 12px;
      padding: 0.875rem 1.25rem;
      color: #d1d5db;
      font-size: 0.95rem;
    }
    .feature-icon { font-size: 1.25rem; flex-shrink: 0; }
    .strava-btn {
      display: inline-flex;
      align-items: center;
      gap: 0.625rem;
      background: #fc4c02;
      color: #fff;
      border: none;
      border-radius: 12px;
      padding: 1rem 2rem;
      font-size: 1.05rem;
      font-weight: 700;
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
    .disclaimer {
      margin-top: 1.25rem;
      font-size: 0.8rem;
      color: #6b7280;
    }
  `]
})
export class LandingComponent implements OnInit {
  loading = false;

  constructor(private auth: AuthService, private route: ActivatedRoute) {}

  ngOnInit(): void {
    // Handle OAuth callback — JWT is in the URL fragment
    const fragment = this.route.snapshot.fragment ?? '';
    if (fragment.includes('token=')) {
      this.loading = true;
      this.auth.handleCallback(fragment);
    }
  }

  login(): void {
    this.loading = true;
    this.auth.login();
  }
}
