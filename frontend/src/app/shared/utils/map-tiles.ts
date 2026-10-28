// LAYER: Shared — Map configuration
// PURPOSE: Single basemap tile URL (CARTO Voyager, Field Guide theme) for every Leaflet map.
//          CARTO requires an API key; without one every tile is an "API KEY REQUIRED" watermark.
import { CARTO_API_KEY } from '../../../environments/carto-key';

export const BASEMAP_TILE_URL =
  `https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png?key=${CARTO_API_KEY}`;
