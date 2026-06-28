// LAYER: Component — Loading Overlay
// PURPOSE: Full-screen translucent overlay shown during route generation.
//          Sits above the map and provides animated feedback during the 30-60s wait.

import { Component } from '@angular/core';

@Component({
  selector: 'app-loading-overlay',
  standalone: true,
  template: `
    <div class="overlay">
      <div class="spinner-ring"></div>
      <h3>Generating Routes</h3>
      <p>Running route search — this takes 30-60 seconds.</p>
      <div class="progress-dots">
        <span></span><span></span><span></span>
      </div>
    </div>
  `,
  styles: [`
    .overlay {
      position: absolute; inset: 0; z-index: 10;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      background: rgba(18, 26, 19, 0.88); backdrop-filter: blur(6px);
      color: var(--text-primary); text-align: center; gap: 1rem;
    }
    .spinner-ring {
      width: 56px; height: 56px;
      border: 4px solid rgba(200, 145, 90, 0.2);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.9s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    h3 { margin: 0; font-size: 1.25rem; color: var(--text-primary); }
    p { margin: 0; color: var(--text-muted); font-size: 0.875rem; }
    .progress-dots { display: flex; gap: 6px; }
    .progress-dots span {
      width: 8px; height: 8px; border-radius: 50%; background: var(--accent); opacity: 0.3;
      animation: pulse 1.2s ease-in-out infinite;
    }
    .progress-dots span:nth-child(2) { animation-delay: 0.2s; }
    .progress-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes pulse { 0%,100% { opacity: 0.3; } 50% { opacity: 1; } }
  `]
})
export class LoadingOverlayComponent {}
