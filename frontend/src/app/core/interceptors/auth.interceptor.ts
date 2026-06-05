// LAYER: HTTP Interceptor
// PURPOSE: Attaches the JWT Authorization header to every outgoing request.
//          On a 401 response, logs the user out (token expired or invalid).
//          This is the ONLY place the JWT is read from storage for HTTP calls —
//          route functions never touch localStorage directly.

import { HttpInterceptorFn, HttpRequest, HttpHandlerFn, HttpErrorResponse } from '@angular/common/http';
import { inject } from '@angular/core';
import { catchError, throwError } from 'rxjs';
import { AuthService } from '../services/auth.service';

export const authInterceptor: HttpInterceptorFn = (
  req: HttpRequest<unknown>,
  next: HttpHandlerFn
) => {
  const auth = inject(AuthService);
  const token = auth.getToken();

  // Attach bearer token if we have one
  const authReq = token
    ? req.clone({ setHeaders: { Authorization: `Bearer ${token}` } })
    : req;

  return next(authReq).pipe(
    catchError((err: HttpErrorResponse) => {
      if (err.status === 401) {
        // Token is expired or invalid — clear state and redirect to login
        auth.logout();
      }
      return throwError(() => err);
    })
  );
};
