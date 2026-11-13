# Luggage Tampering Detection

Local Phase 1 prototype. The native C++ application hosts the local dashboard, accepts a recorded video or RTSP/RTSPS URL, runs inference, and displays annotated recordings and review alerts. Inference uses RTMDet-Ins-S for person/luggage detections and luggage instance masks, RTMPose-M for COCO wrist keypoints, and a ByteTrack two-stage association tracker with Kalman motion prediction. The browser UI is HTML/CSS/JavaScript served by the C++ application; Python is not required.

An alert means that a person's wrist was close to a tracked luggage mask for the configured number of frames. It does not prove contact, intent, or theft. Review every alert manually.

## 1. Install dependencies

- Windows 10/11, Visual Studio C++ workload, and CMake 3.20 or newer.
- OpenCV development package with `dnn`, `videoio`, and FFmpeg/RTSP support.
- ONNX Runtime C++ CPU package; the project uses its C++ API to execute the model exports.
- Compatible RTMDet-Ins-S and RTMPose-M ONNX models are included in models/.

Download/extract the ONNX Runtime Windows x64 CPU archive into `deps/onnxruntime-win-x64-1.31.0/`, then configure CMake to find both dependencies. The build copies `onnxruntime.dll` beside the executable. Ensure the OpenCV runtime DLL directory is on `PATH` before launching.

```powershell
New-Item -ItemType Directory -Force build, deps | Out-Null
Invoke-WebRequest `
  -Uri "https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-win-x64-1.31.0.zip" `
  -OutFile "build/onnxruntime-win-x64-1.31.0.zip"
Expand-Archive build/onnxruntime-win-x64-1.31.0.zip -DestinationPath deps -Force
cmake -S . -B build -DOpenCV_DIR="C:/opencv/build/x64/vc16/lib" -DONNXRUNTIME_ROOT="$PWD/deps/onnxruntime-win-x64-1.31.0"
cmake --build build --config Release
```

Copy `config.example.json` to `config.json`. Put the ONNX files at the configured paths (default `models/rtmdet_ins_s.onnx` and `models/rtmpose_m.onnx`). Start the native C++ dashboard server:

```powershell
$env:PATH = "C:\opencv\opencv\build\x64\vc16\bin;$env:PATH"
.\build\Release\luggage_monitor.exe --server 8081
```

Open **http://127.0.0.1:8081**. The server listens on localhost. Pass another port after `--server` to change it. Select a video file or enter an `rtsp://` / `rtsps://` stream URL.

## 2. ONNX export requirements

RTMDet-Ins must be exported as a post-processed MMDeploy ONNX model with output names/order `dets`, `labels`, `masks` (the standard MMDeploy instance-segmentation export). The application decodes boxes/scores, COCO labels, and per-instance masks. Suitcases are rendered as tinted segmentation masks with their tracked `#ID`; suitcase rectangles are not drawn. Person rectangles and wrist markers are shown only on frames that trigger an alert. Set `detector_output_format` to `mmdeploy_dets_labels` in `config.json`; the dashboard sets this for each run.

RTMPose-M must expose SimCC x/y outputs for a 192×256 RGB person crop, conventionally named `simcc_x` and `simcc_y`, with 17 COCO keypoints. Set `pose_x_output_name` and `pose_y_output_name` in `config.json` if the export uses other names. The left and right wrist indices are 9 and 10. The input is expanded to the pose aspect ratio, resized, and standardized with the OpenMMLab RGB mean and standard deviation.

Follow [tools/export_models.md](tools/export_models.md) for concrete MMDeploy export commands and official COCO checkpoints.

OpenMMLab supports RTMDet-Ins ONNX deployment through MMDeploy's instance-segmentation config, with post-processed masks enabled. RTMPose's deployment config names the SimCC outputs `simcc_x` and `simcc_y`. See the [RTMDet deployment guide](https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/README.md) and [RTMPose deployment guide](https://github.com/open-mmlab/mmpose/blob/main/docs/en/user_guides/how_to_deploy.md) for export steps and supported versions.

## 3. Settings and outputs

Edit `config.json` for model paths, detector/pose output names, classes, thresholds, and input dimensions. Person detections use a 0.40 confidence threshold. Baggage classes are backpack 24, handbag 26, and suitcase 28, with class-specific thresholds and size-aware cutoffs:

| Class | Small box (<7% of detector image) | Medium box (7–20%) | Large box (≥20%) |
|---|---:|---:|---:|
| Suitcase | 0.30 | 0.30 | 0.35 |
| Backpack | 0.35 | 0.35 | 0.35 |
| Handbag | 0.40 | 0.40 | 0.40 |

The detector input is 640×640 and pose input is 192×256. Uploaded clips analyze every 6th source frame and confirm contact in 2 of 3 analyzed samples. RTSP keeps its existing interval from configuration (12 source frames by default). The inference interval is configurable from 1 to 2400 source frames for RTSP. Skipped source frames are advanced without retrieving full images. Long-axis detector tiling is off by default because it doubles detector inference on common widescreen clips; set `tile_long_axis` to `true` when small, distant bags need more detector pixels and slower processing is acceptable. The live preview is resized to at most 960 pixels wide and JPEG quality 82; inference, evidence images, and the finished annotated video retain full resolution. After analysis, the dashboard plays the uploaded source at its original frame rate and synchronizes transparent suitcase-mask and confirmed-contact person-box overlays from analyzed samples; alert evidence remains available for review. Near-identical baggage masks are suppressed, small mask fragments are removed, and detections with masks mostly overlapping a person are filtered. A tracked bag is treated as background only after it has stayed still for more than 3 seconds and its box area is larger than the average area of currently tracked suitcases. Alerts latch per 200-pixel image zone until contact clears for 10 analyzed frames. An alert is a visual estimate of contact, not proof of intent or theft. Pose and mask errors may cause missed or false alerts; review annotations and tune thresholds for the camera.

Each run creates or updates:

- `output/latest.jpg` — latest annotated frame, polled by the web page.
- `output/evidence/` — annotated still image per alert.
- `output/events.json` — line-delimited records displayed in the page.
- `output/events.csv` — review/export records.
- `output/pipeline.log` — inference process log.
- `input/` — uploaded recordings.

ByteTrack uses high-score matching first, then a lower-score association pass and a Kalman motion estimate. It keeps unmatched tracks briefly through occlusion; IDs can still switch in crowded or fully occluded scenes. Timestamps are stream elapsed time (frame number divided by reported FPS), rather than wall-clock time.

## Known limits

- Models must be supplied locally in the stated ONNX formats. Arbitrary ONNX exports do not share a universal tensor convention.
- Uploads are limited to 1 GB. RTSP needs an OpenCV build with FFmpeg and network support.
- Only annotated evidence frames are saved; event video clips are not generated yet.
- Validate model preprocessing, masks, tracking, lighting, occlusion, and alert thresholds on the target camera before using live footage.

