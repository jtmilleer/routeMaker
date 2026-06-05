// LAYER: Route Guard
// PURPOSE: Redirects unauthenticated users to the landing page.
//          Waits for auth initialization before making a decision
//          (avoids false redirects during cold load while the JWT is being verified).

import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { AuthService } from '../services/auth.service';

export const authGuard: CanActivateFn = () => {
  const auth = inject(AuthService);
  const router = inject(Router);

  if (auth.isAuthenticated()) {
    return true;
  }
  return router.createUrlTree(['/']);
};
