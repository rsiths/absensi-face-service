# Face service absensi

Service internal kecil untuk membuat dan memverifikasi embedding wajah. Keputusan akhir absensi tetap dibuat oleh API CI3.

## Menjalankan

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 127.0.0.1 --port 8010
```

Model InsightFace akan diunduh oleh runtime saat pertama kali dipakai. Pastikan server dapat mengakses model pada tahap instalasi awal, atau sediakan cache model secara internal.

Endpoint:

- `GET /health`
- `POST /face/register` dengan `{ "images": ["data:image/jpeg;base64,..."] }`
- `POST /face/verify` dengan `{ "image": "...", "embedding": [...] }`

Set `FACE_SERVICE_SHARED_SECRET` pada service dan environment PHP `FACE_SERVICE_SHARED_SECRET` dengan nilai yang sama. Jangan expose service ini langsung ke internet.
