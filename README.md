# Vision Ops: People & Object Detection (OpenCV + FastAPI)

Upload a photo or video, or point it at a webcam or RTSP camera. Every person and object is found, boxed and described, and the dashboard lists everything in the scene.

| What | How |
|---|---|
| **Objects and people in photos** (80 types: person, car, phone, bag, cup, laptop…) | YOLO11x, the most accurate YOLO11 model |
| **Thousands more object types in photos** (glasses, shoes, billboards, lamps, signs… 4,585 types) | YOLOE prompt-free, open-vocabulary |
| **Objects and people in live video** | YOLO11s (falls back to YOLO11n) |
| **Every region of the scene** (sky, building, wall, tree, plant, road, floor… 150 types) | SegFormer-B0 (ADE20K) semantic segmentation |
| **Is it a human?** Body skeleton with 17 joints | YOLO11n-pose |
| **What is the person doing?** | Rules on pose and objects: standing, sitting, lying down, crouching, walking, running, hand raised, arms crossed, pointing, on the phone, drinking, eating, reading, riding a bike, using a laptop… |
| **Perceived gender and estimated age** | YuNet face detector + InsightFace `genderage` |
| **Facial expression** | OpenCV Zoo MobileFaceNet (neutral, happy, sad, angry, surprised…) |
| **Clothing colour, carried and worn items** | Colour of the torso area; objects near hands, on the body or worn (backpack, handbag, tie…) |
| **Indoor or outdoor**, and what's in the background | Computed from the segmentation |
| **Crowd density** | Density heatmap from person positions |
| *Optional:* fire and smoke | Off by default (`ENABLE_HAZARDS=true` turns it on) |

Each person gets a stable **track ID**. Gender, age and expression are **voted over many frames**, so a single bad frame doesn't change the result. A plain-English summary is produced for each person, for example:

> *Man, ~33 yrs — upper body only, still, hands resting, facing camera, neutral expression. Wearing a blue top.*

Everything runs on **OpenCV DNN with ONNX models**, so PyTorch isn't needed at runtime.

## Quick start (Windows / PowerShell)

```powershell
cd ai-ml\object-detection
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python scripts\download_model.py          # base models (~55 MB): enough to run everything
uvicorn app.main:app --reload
```

**Optional: the high-accuracy models.** YOLO11s, YOLO11x and YOLOE aren't published as ONNX files, so they need a one-time export. The export uses PyTorch, so do it in a separate environment and the app itself stays PyTorch-free:

```powershell
py -3.12 -m venv .export-venv
.\.export-venv\Scripts\pip install ultralytics onnx onnxslim
.\.export-venv\Scripts\python scripts\export_models.py   # yolo11s, yolo11x, YOLOE (~400 MB)
```

Without them, photos and video use YOLO11n, and the health endpoint tells you what's missing.

Open **http://127.0.0.1:8000** and drop in an image or video. The live **People** and **Scene** panels update twice a second.
The API docs are at **/docs**. Any missing model just switches off its feature; `GET /api/v1/health` reports which features are on and why any are off.

## Project structure

```
app/
├── main.py                  # app factory; loads all models once at startup
├── core/config.py           # typed settings (env / .env)
├── api/                     # routes: health · detect · streams, plus dependency injection
├── schemas/                 # Pydantic response models (PersonOut, SceneOut, …)
├── ml/                      # pure computer vision: no web code
│   ├── detector.py          # YOLO objects: letterbox → forward → decode → NMS
│   ├── pose.py              # YOLO-pose: 17 keypoints per person
│   ├── face.py              # YuNet faces → gender/age + expression
│   ├── scene.py             # SegFormer: per-pixel scene labels, indoor/outdoor
│   ├── activity.py          # explainable rules: posture, gestures, interactions, clothing
│   ├── density.py · hazard.py · labels.py
├── services/
│   ├── models.py            # model registry (optional models degrade gracefully)
│   ├── analyzer.py          # the per-frame pipeline (parallel, async scene)
│   ├── tracker.py           # person IDs + evidence voting across frames
│   ├── results.py           # result types + natural-language descriptions
│   └── streams.py           # video sessions; real-time frame skipping
├── vision/hud.py            # overlay: skeletons, info cards, scene tint, panels
└── static/                  # dashboard (no build step)
scripts/download_model.py    # fetches every model, validating sizes
tests/                       # 32 tests (unit + API + full video flow)
```

## Pipeline

```
frame ─┬─ objects (YOLO, every 2nd frame) ─┐
       ├─ pose (YOLO-pose, every frame) ───┼─► persons ─► tracker ─► face votes (every 3rd frame)
       ├─ faces (YuNet + attributes) ──────┘                 └─► posture · motion · gestures · items · clothing
       ├─ scene (SegFormer, async, about 1/s) ─► background tint + labels + indoor/outdoor
       └─ fire/smoke · crowd density ─────────► HUD + JSON stats
```

The models run in parallel threads. Heavy models run less often in video:
- **Gender/age** stop running once each visible person has enough votes, and re-check occasionally after that.
- **Uploaded videos** drop frames to stay on the real timeline, so what you see is live, like a camera feed.

**Speed:** about 4 analysed frames per second with every feature on, measured on a laptop CPU (i7-1185G7). Each frame takes about 170 ms. For higher frame rates you can:
- set `ENABLE_SCENE=false`
- raise `OBJECT_EVERY_N`
- use a CUDA build of OpenCV (`USE_CUDA=true`)

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/health` | Enabled features, plus the reason any feature is off |
| POST | `/api/v1/detect` | Image → JSON: objects, `people[]`, `scene` (`?include_image=true` adds the annotated JPEG; `?mode=max_recall` finds more small or far objects) |
| POST | `/api/v1/detect/annotated` | Image → annotated JPEG |
| WS | `/api/v1/live` | **Browser webcam**: send JPEG frames, receive the annotated frame plus stats (one frame in flight, so no lag) |
| POST | `/api/v1/streams/upload` | Upload a video → `stream_id`, `video_url`, `stats_url` |
| POST | `/api/v1/streams/source` | Server-side camera (`{"source": "0"}`) or RTSP (when enabled). Many laptop cameras only work via the browser endpoint above |
| GET | `/api/v1/streams/{id}/video` | Live annotated MJPEG |
| GET | `/api/v1/streams/{id}/stats` | Live JSON, with the same `people[]` and `scene` fields |
| DELETE | `/api/v1/streams/{id}` | Stop the stream and delete the uploaded file |

Each entry in `people[]` has these fields:
- `track_id` and `description`
- `gender` and `gender_confidence`, `age`
- `expression`, `posture`, `motion`, `facing`
- `gestures`, `activities`, `holding`, `clothing_color`, `accessories`
- `box`, `face_box`
- `keypoints`, on the image endpoint only

## Measured accuracy

**No detector finds 99% of all people and objects in arbitrary photos.** The best published models score about 55–65 COCO mAP, because small, hidden, blurred and cut-off objects are genuinely hard. Rather than quote marketing numbers, `scripts/evaluate.py` measures this app's own detection code on the **COCO 2017 validation set** (human-labelled images the models never trained on).

**Model comparison** (500 images, 1,081 labelled people, threshold 0.35):

| Model | mAP50-95 | People found | Large people | Medium | Small | Precision |
|---|---|---|---|---|---|---|
| YOLO11n (old default) | 40.9 | 62.0% | 90.7% | 65.4% | 25.3% | 86.7% |
| YOLO11s (video default) | 48.1 | 68.9% | 91.7% | 72.6% | 38.7% | 86.1% |
| YOLO11m | 52.1 | 71.8% | 93.3% | 74.9% | 43.8% | 86.1% |
| YOLO11l | 54.4 | 71.9% | 93.8% | 73.7% | 44.6% | 86.6% |
| **YOLO11x (photo default)** | **56.7** | **74.9%** | **94.3%** | **79.3%** | **47.9%** | **87.0%** |

Large people are at least 96×96 px; small people are under 32×32 px.

**Photo modes** (YOLO11x, 150 images, 404 people):

| Mode | Precision | People found | Large | Medium | Small | CPU time per photo |
|---|---|---|---|---|---|---|
| `balanced` (default) | **85.8%** | 73.0% | 94.3% | 77.4% | 41.0% | about 3 s |
| `max_recall` (tiles + threshold 0.25) | 69.7% | **84.2%** | **95.6%** | **92.7%** | **60.7%** | about 15 s |

How to read this:
- **Clearly visible people are found about 94–96% of the time.** The rest are mostly heavily hidden, cut off at the frame edge, or seen only from behind in clutter.
- **Small, distant people are the hard case.** `max_recall` finds far more of them, at the price of more false alarms.
- Run `python scripts/evaluate.py --images … --labels … --model …` to measure on your own labelled photos. Your own camera images are the numbers that really matter.

## Limits (read before demoing)

- **Gender and age are appearance estimates.** They come from a face crop. They are never facts about a person's identity, and the UI labels them "est.".
  - On our test faces the gender model was 9 of 9 correct, including cases the older GoogLeNet model got wrong.
  - Age is typically within about ±8 years.
  - In video, a gender needs 3 or more agreeing votes with at least a 75% share; otherwise it shows **UNCERTAIN**.
  - With no visible face, for example a person seen from behind, gender shows "no clear face".
  - You can switch this feature off with `ENABLE_DEMOGRAPHICS=false`.
- **Actions** come from body geometry. If the legs are out of frame, the result says *"upper body only"* instead of guessing sitting or standing.
- **Objects:** only the 80 COCO types get boxes. Everything else (walls, plants, sky…) is covered by segmentation, which is updated about once per second in video.
- **Fire and smoke** are off by default. When switched on they use heuristics; for production, use a trained model via `HAZARD_MODEL_PATH`.

## Configuration

All settings are in `app/core/config.py`. You can override any of them in `.env`; see `.env.example`.

| Variable | Default | Notes |
|---|---|---|
| `CONF_THRESHOLD` | 0.35 | Lower it to catch more small or distant objects, at the cost of more false alarms |
| `IMAGE_MODEL` | yolo11x.onnx | Photo detector; falls back to `MODEL_PATH` if missing |
| `MAX_RECALL_TILES` / `MAX_RECALL_CONF` | 2 / 0.25 | Settings for `mode=max_recall` |
| `ENABLE_OPEN_VOCAB` | true | YOLOE with 4,585 object types (photos only; about 3 s on CPU) |
| `ENABLE_HAZARDS` | false | Optional fire and smoke heuristics |
| `ENABLE_POSE` / `ENABLE_FACES` / `ENABLE_SCENE` | true | Turn individual features on or off |
| `ENABLE_DEMOGRAPHICS` | true | Perceived gender and estimated age |
| `SCENE_OVERLAY` | true | Tint and label every scene region on the video |
| `SCENE_INTERVAL_S` | 1.0 | How often the scene is re-segmented in video |
| `OBJECT_EVERY_N` / `FACE_EVERY_N` | 2 / 3 | Video: how often to run those models |
| `ALLOW_URL_SOURCES` | false | Enable for RTSP cameras on trusted networks |

## Tests

```bash
pytest -q
```

The suite has 32 tests. They cover:
- **Detection:** YOLO decoding and NMS
- **Activities:** posture, gestures, phone and carried items, worn tie
- **Appearance:** clothing colours
- **Tracker:** stable IDs, no ID swaps between neighbours, gender voting
- **Descriptions:** the per-person summary text
- **Fire and density:** fire detection, density levels
- **API:** every endpoint, plus the full video upload → MJPEG → stats → delete flow

## Model credits

- YOLO11 (n/s/x), YOLO11n-pose and YOLOE: Ultralytics (AGPL-3.0)
- YuNet face detector and facial expression model: OpenCV Zoo
- `genderage`: InsightFace buffalo_l (non-commercial research licence)
- SegFormer-B0 ADE20K: NVIDIA, via Xenova/Hugging Face

**Check these licences before any commercial use.**
