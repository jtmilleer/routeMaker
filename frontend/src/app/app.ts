// LAYER: Component — App Shell
// PURPOSE: Root component. Renders the router-outlet that hosts all pages.
//          Shows a brief overlay spinner while auth is initializing.

import { Component, inject } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { AuthService } from './core/services/auth.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  auth = inject(AuthService);
}
