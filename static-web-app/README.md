# FloodCast Static Web App — Deployment Guide

Single-page demo dashboard dengan Azure Maps choropleth visualization.

## Features

- **Interactive map** dengan 15 kelurahan pilot menggunakan Azure Maps SDK
- **Multi-horizon view** (6h / 12h / 24h) dengan tombol switcher
- **Risk-coded bubbles** (Aman/Waspada/Siaga/Awas) per kelurahan
- **Detail panel** menampilkan probabilitas, top 5 SHAP factors, dan generative advisory
- **Multi-audience advisory** tab — generate pesan berbeda untuk Warga / BPBD / Perencana
- **Smart fallback** — kalau Azure Function API tidak tersedia, pakai data lokal `data/predictions.json`

## Local Testing

```bash
cd static-web-app/public
python -m http.server 8080
# Buka http://localhost:8080
```

## Configuration

### Azure Maps subscription key

Ubah `localStorage` di browser console, atau hardcode di `app.js`:

```js
window.localStorage.setItem("azure_maps_key", "YOUR_KEY_HERE");
```

Atau edit baris di `app.js`:
```js
azureMapsKey: window.localStorage.getItem("azure_maps_key") || "YOUR_KEY_HERE",
```

Cara dapat key: Azure Portal → Create Azure Maps account (free tier: 1.000 transaksi/bulan) → copy primary key.

### API endpoint (optional — kalau Azure Function sudah deploy)

```js
window.localStorage.setItem("floodcast_api", "https://floodcast-api.azurewebsites.net");
```

## Deploy ke Azure Static Web Apps

### Opsi 1: Via Azure CLI

```bash
# 1. Login
az login

# 2. Create static web app (Free tier)
az staticwebapp create \
  --name floodcast-jakarta \
  --resource-group floodcast-rg \
  --location southeastasia \
  --sku Free

# 3. Get deployment token
az staticwebapp secrets list \
  --name floodcast-jakarta \
  --query properties.apiKey -o tsv

# 4. Deploy via SWA CLI
npm install -g @azure/static-web-apps-cli
swa deploy public --deployment-token <TOKEN_FROM_STEP_3>
```

### Opsi 2: Via GitHub Actions (recommended)

1. Push folder `static-web-app/public/` ke repo GitHub
2. Azure Portal → Create Static Web App → connect ke repo
3. Set:
   - **Build Presets**: Custom
   - **App location**: `/static-web-app/public`
   - **Output location**: (kosongkan)
4. Auto-generated GitHub Action akan deploy setiap push ke `main`

URL deployed app akan jadi `https://<random-name>.azurestaticapps.net`.

## File Structure

```
public/
├── index.html              # Main UI
├── app.js                  # Map logic, API calls, rendering
├── staticwebapp.config.json # SWA routing config
└── data/
    └── predictions.json    # Demo data (15 kelurahan, 3 horizons)
```

## Demo Mode vs Live API Mode

Status ditampilkan di top-right header:

- **DEMO DATA** (biru) — fetching dari `data/predictions.json` lokal
- **API LIVE** (hijau) — connected ke Azure Function endpoint

Demo mode tetap demo-able penuh — semua interaksi dan generative advisory masih bekerja via fallback templates.
