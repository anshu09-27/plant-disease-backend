# Plant Disease Diagnosis — Flask Backend

Flask API that serves your trained MobileNetV2 TFLite model for the AI Plant
Disease Diagnosis app. Replaces the current demo-mode responses in your
Flutter `services/` module.

## 1. Local setup

```bash
cd plant_backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place your trained `.tflite` file at `model/plant_disease_model.tflite`
(same file you already have in `assets/model_ai/` in the Flutter repo —
just copy it here, or point `MODEL_PATH` at it via env var).

```bash
mkdir -p model
cp /path/to/your/assets/model_ai/your_model.tflite model/plant_disease_model.tflite
```

Run it:

```bash
python app.py
# Server on http://localhost:5000
```

## 2. IMPORTANT — verify class order

`disease_info.py` assumes the standard alphabetical PlantVillage 38-class
order (the order Keras assigns by default when training from folders). If
your training notebook printed a `class_indices` dict, compare it against
`CLASS_NAMES` in `disease_info.py` line by line — if the order doesn't
match, predictions will be mapped to the wrong disease names even though
the model itself is accurate. This is the #1 source of "wrong disease"
bugs in TFLite deployments.

## 3. Endpoints

| Method | Path       | Description                                   |
|--------|-----------|------------------------------------------------|
| GET    | `/health`  | Liveness + whether model loaded                |
| GET    | `/classes` | List of all 38 supported classes               |
| POST   | `/predict` | Upload image (`multipart/form-data`, field `image`) → prediction |

Example response from `/predict`:

```json
{
  "class_name": "Tomato___Late_blight",
  "display_name": "Tomato Late Blight",
  "healthy": false,
  "confidence": 0.9421,
  "treatment": "Act quickly with fungicides (chlorothalonil or copper-based)..."
}
```

Test locally with curl:

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/leaf.jpg"
```

## 4. Deploy to Render.com

1. Push this `plant_backend/` folder to a GitHub repo (can be a subfolder
   of your existing `AI-plant-disease-diagnosis` repo, or its own repo).
2. On Render: **New +** → **Web Service** → connect the repo.
3. Render should auto-detect `render.yaml`. If not, set manually:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
4. Make sure your `.tflite` model file is committed to the repo (or
   uploaded via a persistent disk / object storage — GitHub is fine for
   files under 100MB, use Git LFS if larger).
5. Once deployed, Render gives you a URL like
   `https://plant-disease-backend.onrender.com`.

Note: Render's free tier spins down after inactivity, so the first request
after idle time takes ~30-50s to cold-start.

## 5. Flutter integration (replacing demo mode)

In your `services/` module, point the API base URL at the deployed
backend and switch off demo mode:

```dart
// services/api_service.dart
class ApiService {
  static const String baseUrl = 'https://plant-disease-backend.onrender.com';

  static Future<Map<String, dynamic>> predictDisease(File imageFile) async {
    final uri = Uri.parse('$baseUrl/predict');
    final request = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath('image', imageFile.path));

    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      return jsonDecode(response.body) as Map<String, dynamic>;
    } else {
      throw Exception('Prediction failed: ${response.body}');
    }
  }
}
```

The response fields (`class_name`, `display_name`, `healthy`, `confidence`,
`treatment`) map directly onto whatever result model/widget you're already
using for the demo-mode UI — you should just need to swap the data source,
not the UI layer.

## 6. Notes on `tflite-runtime`

`requirements.txt` uses `tflite-runtime`, which is much smaller/faster to
install than full TensorFlow (good for Render's free tier build time). If
Render's Python version doesn't have a prebuilt `tflite-runtime` wheel
available, swap it out in `requirements.txt` for:

```
tensorflow-cpu==2.16.1
```

`app.py` already has a fallback that imports `tensorflow` if
`tflite_runtime` isn't installed, so no code changes are needed — just the
requirements file.
