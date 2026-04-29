"""
Azure OpenAI advisory endpoint.

Endpoint: POST /api/advisory
Body: {
  "kelurahan": "Kampung Melayu",
  "probability": 0.78,
  "horizon_hours": 12,
  "risk_level": "Siaga",
  "top_factors": [...],
  "audience": "warga" | "bpbd" | "perencana"
}

Returns Azure OpenAI generated advisory message tailored to audience.
"""
import json
import logging
import os
from datetime import datetime, timezone

import azure.functions as func

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None


SYSTEM_PROMPTS = {
    "warga": """Kamu adalah asisten peringatan banjir untuk warga Jakarta. Tulis pesan singkat (max 4 kalimat) dalam bahasa Indonesia yang sederhana, tanpa jargon teknis. Berikan rekomendasi tindakan konkret dan spesifik berdasarkan level risiko. Jangan menakut-nakuti, tapi jelaskan urgensi sesuai data.""",
    "bpbd": """Kamu adalah asisten operasional untuk BPBD DKI Jakarta. Tulis briefing singkat (max 5 kalimat) dalam bahasa Indonesia formal-operasional. Sebutkan: tingkat prioritas, rekomendasi pre-positioning sumber daya konkret (perahu karet, posko, evakuasi dini), dan estimasi window waktu untuk bertindak. Gunakan terminologi BPBD.""",
    "perencana": """Kamu adalah asisten analitik untuk perencana kota Bappeda DKI Jakarta. Tulis ringkasan analitik singkat (max 4 kalimat) dalam bahasa Indonesia formal. Fokus pada: identifikasi pola risiko, faktor struktural pendorong dari data fitur, dan implikasi untuk perencanaan jangka panjang (RTH, drainase, normalisasi)."""
}


def build_user_message(payload: dict) -> str:
    factors = payload.get("top_factors", [])
    factors_str = "\n".join([
        f"- {f['feature']}: nilai {f['value']:.2f} (kontribusi {f['importance']:.3f})"
        for f in factors[:5]
    ])
    return f"""Data prediksi banjir:
- Kelurahan: {payload['kelurahan']}
- Probabilitas banjir: {payload['probability']*100:.1f}%
- Horizon waktu: {payload['horizon_hours']} jam ke depan
- Level risiko: {payload['risk_level']}
- Faktor utama pendorong:
{factors_str}

Tulis pesan peringatan."""


def fallback_message(payload: dict, audience: str) -> str:
    """Template-based fallback when Azure OpenAI is not configured."""
    kel = payload['kelurahan']
    prob = payload['probability'] * 100
    horizon = payload['horizon_hours']
    risk = payload['risk_level']

    if audience == "warga":
        if risk in ("Siaga", "Awas"):
            return (f"Wilayah {kel} berisiko TINGGI banjir ({prob:.0f}%) dalam {horizon} jam ke depan. "
                    f"Segera pindahkan kendaraan ke tempat tinggi, siapkan tas darurat berisi dokumen penting "
                    f"dan obat-obatan, dan pantau ketinggian air pintu air Manggarai. "
                    f"Jika air mulai naik di sekitar rumah, segera evakuasi ke titik aman terdekat.")
        elif risk == "Waspada":
            return (f"Wilayah {kel} dalam status WASPADA banjir ({prob:.0f}%) dalam {horizon} jam ke depan. "
                    f"Pantau perkembangan cuaca dan ketinggian air, siapkan rencana evakuasi keluarga, "
                    f"dan simpan barang berharga di tempat yang lebih tinggi.")
        else:
            return f"Wilayah {kel} dalam status AMAN ({prob:.0f}% risiko) untuk {horizon} jam ke depan. Tetap pantau update."

    elif audience == "bpbd":
        return (f"PRIORITAS {risk.upper()}: {kel}, probabilitas banjir {prob:.0f}% dalam {horizon} jam. "
                f"Rekomendasi: pre-positioning 2 perahu karet di pos siaga terdekat, "
                f"koordinasi dengan kelurahan untuk identifikasi lansia/disabilitas yang perlu evakuasi dini, "
                f"siapkan posko logistik dengan kapasitas 200 jiwa. Window aksi: {max(2, horizon-4)} jam.")
    else:
        return (f"Analisis {kel}: probabilitas banjir {prob:.0f}% horizon {horizon}h didorong oleh faktor "
                f"hidrologis dan tutupan lahan. Pola ini konsisten dengan kelurahan dengan elevasi rendah "
                f"dan akumulasi curah hujan tinggi. Implikasi: prioritas program RTH dan normalisasi drainase "
                f"di wilayah ini untuk mitigasi jangka panjang.")


def call_azure_openai(payload: dict, audience: str) -> str:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not (endpoint and api_key and AzureOpenAI):
        logging.warning("Azure OpenAI not configured, using fallback template")
        return fallback_message(payload, audience)

    try:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[audience]},
                {"role": "user", "content": build_user_message(payload)},
            ],
            temperature=0.4,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Azure OpenAI call failed: {e}")
        return fallback_message(payload, audience)


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Advisory endpoint hit")

    try:
        payload = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON body"}),
            status_code=400,
            mimetype="application/json",
        )

    required = {"kelurahan", "probability", "horizon_hours", "risk_level"}
    missing = required - set(payload.keys())
    if missing:
        return func.HttpResponse(
            json.dumps({"error": f"Missing fields: {sorted(missing)}"}),
            status_code=400,
            mimetype="application/json",
        )

    audience = payload.get("audience", "warga")
    if audience not in SYSTEM_PROMPTS:
        return func.HttpResponse(
            json.dumps({"error": f"audience must be one of {list(SYSTEM_PROMPTS.keys())}"}),
            status_code=400,
            mimetype="application/json",
        )

    message = call_azure_openai(payload, audience)

    return func.HttpResponse(
        json.dumps({
            "audience": audience,
            "kelurahan": payload["kelurahan"],
            "message": message,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }),
        status_code=200,
        mimetype="application/json",
        headers={"Access-Control-Allow-Origin": "*"},
    )
