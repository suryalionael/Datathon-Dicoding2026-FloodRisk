# FloodCast Azure Function — Deployment Guide

REST API yang membungkus model XGBoost flood prediction menjadi serverless endpoint di Azure Functions.

## Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/api/health` | Service health + model availability |
| `GET` | `/api/predict?kelurahan=Kampung+Melayu&horizon=24` | Single kelurahan prediction |
| `GET` | `/api/predict?all=true&horizon=12` | All 15 pilot kelurahan |
| `POST` | `/api/advisory` | Multi-audience advisory (Azure OpenAI) |

## Local Development

```bash
# Prerequisites
# - Python 3.11
# - Azure Functions Core Tools v4: npm i -g azure-functions-core-tools@4

cd azure-function
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Copy and edit settings
cp local.settings.json.example local.settings.json

# Place trained models in ../models/ and feature snapshot in ../data/
# Required files:
#   ../models/flood_xgb_6h.joblib
#   ../models/flood_xgb_12h.joblib
#   ../models/flood_xgb_24h.joblib
#   ../data/latest_features.parquet  (or .json)

# Run locally
func start
```

Test:
```bash
curl http://localhost:7071/api/health
curl "http://localhost:7071/api/predict?kelurahan=Kampung%20Melayu&horizon=24"
curl -X POST http://localhost:7071/api/advisory \
  -H "Content-Type: application/json" \
  -d '{"kelurahan":"Kampung Melayu","probability":0.78,"horizon_hours":12,"risk_level":"Siaga","top_factors":[],"audience":"warga"}'
```

## Deploy to Azure (Free Consumption Plan)

```bash
# 1. Login
az login

# 2. Create resource group
az group create --name floodcast-rg --location southeastasia

# 3. Create storage account (required for Functions)
az storage account create \
  --name floodcaststorage$RANDOM \
  --location southeastasia \
  --resource-group floodcast-rg \
  --sku Standard_LRS

# 4. Create function app (Consumption plan = free tier)
az functionapp create \
  --resource-group floodcast-rg \
  --consumption-plan-location southeastasia \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --name floodcast-api \
  --storage-account <name-from-step-3> \
  --os-type linux

# 5. Configure Azure OpenAI env vars (after AOAI resource is created)
az functionapp config appsettings set \
  --name floodcast-api \
  --resource-group floodcast-rg \
  --settings \
    AZURE_OPENAI_ENDPOINT="https://<your-aoai>.openai.azure.com/" \
    AZURE_OPENAI_KEY="<key>" \
    AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini"

# 6. Deploy
func azure functionapp publish floodcast-api

# 7. Test deployed endpoint
curl https://floodcast-api.azurewebsites.net/api/health
```

## Notes

- **Cold start**: 2–4s on first request after idle. Acceptable for MVP demo.
- **Model loading**: Models are loaded lazily and cached in memory between requests on the same warm instance.
- **CORS**: Configured to allow `*` for demo; restrict to your Static Web App domain in production.
- **Azure OpenAI fallback**: If env vars not set, advisory endpoint uses template-based fallback messages (still demonstrates the multi-audience concept).
