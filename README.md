# ParaSight Backend (FastAPI + Keras + Grad-CAM)

## Struktur
```
parasight-backend/
├── app/
│   ├── main.py            # entry point, lifespan, CORS, error handler
│   ├── config.py          # settings (env var)
│   ├── schemas.py         # response model Pydantic
│   ├── errors.py          # AppError
│   ├── gradcam.py         # file Grad-CAM Anda (tidak diubah)
│   ├── api/routes.py      # /health & /predict
│   └── services/
│       ├── image_utils.py # validasi & decode gambar
│       └── inference.py   # prediksi + Grad-CAM + encode base64
├── models/malaria_model.keras
├── requirements.txt
└── .env.example
```

## Menjalankan di Windows (PowerShell)
```powershell
cd parasight-backend
python -m venv venv
.\venv\Scripts\Activate.ps1      # jika diblokir: Set-ExecutionPolicy -Scope Process RemoteSigned
pip install -r requirements.txt

copy .env.example .env
# letakkan model di models\malaria_model.keras (atau ubah MODEL_PATH di .env)

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Dokumentasi interaktif: http://localhost:8000/docs

## Konfigurasi (.env)
| Variabel | Default | Keterangan |
|---|---|---|
| MODEL_PATH | models/malaria_model.keras | Lokasi model |
| MAX_UPLOAD_MB | 5 | Batas ukuran upload |
| THRESHOLD | 0.5 | Ambang klasifikasi |
| LAST_CONV_LAYER_NAME | out_relu | Layer konvolusi terakhir untuk Grad-CAM |
| GRADCAM_ALPHA | 0.4 | Opasitas heatmap |
| ALLOWED_ORIGINS | localhost:3000,5173 | Pisahkan dengan koma |

## Contoh request
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@sel.png;type=image/png"
```

```python
import requests, base64

r = requests.post("http://localhost:8000/predict",
                  files={"file": ("sel.png", open("sel.png", "rb"), "image/png")})
data = r.json()
print(data["prediction"], data["confidence"])
with open("gradcam.png", "wb") as f:
    f.write(base64.b64decode(data["gradcam_base64"].split(",", 1)[1]))
```

## Kode error
| Status | code | Penyebab |
|---|---|---|
| 400 | INVALID_IMAGE / EMPTY_FILE | File rusak, bukan gambar, atau kosong |
| 413 | FILE_TOO_LARGE | Melebihi batas ukuran |
| 415 | UNSUPPORTED_MEDIA_TYPE / UNSUPPORTED_EXTENSION | MIME/ekstensi/isi file tidak valid |
| 422 | VALIDATION_ERROR | Field `file` tidak ada |
| 500 | INFERENCE_FAILED / GRADCAM_FAILED / INTERNAL_ERROR | Kesalahan server (detail hanya di log) |
| 503 | MODEL_NOT_READY | Model gagal dimuat |
