# GPU handoff notes

This bundle includes the C++ application, dashboard, example configuration, and the two ONNX model files used by the current project.

The current application is configured for the ONNX Runtime CPU execution provider. GPU execution is not enabled in this code yet. On the target device, select a compatible ONNX Runtime GPU provider for its hardware (for example DirectML on Windows or CUDA on NVIDIA), use matching headers and runtime libraries from the same ONNX Runtime version, then configure the detector and pose sessions to use that provider. Do not mix the existing CPU 1.31.0 headers with the older DirectML 1.24.4 runtime that may be present on the development machine.

Install a compatible OpenCV development package and ONNX Runtime GPU package on the target device, then follow the build steps in README.md. The model input sizes, preprocessing, and output tensor contracts are documented there. Recheck detection quality and speed on the target GPU after enabling its provider.
