# Model notices

- `rtmdet_ins_s.onnx`: RTMDet-Ins-S COCO ONNX export, converted from OpenMMLab MMDetection weights and distributed under Apache-2.0. Source export: [tori29umai/rtmdet-ins-onnx-person-masks](https://huggingface.co/tori29umai/rtmdet-ins-onnx-person-masks). Its model card documents static 640×640 input and `dets`, `labels`, and `masks` outputs.
- `rtmpose_m.onnx`: OpenMMLab RTMPose-M 17-keypoint SimCC ONNX SDK export, trained on the Body7 data mix with the COCO-17 keypoint layout. Source archive: [OpenMMLab ONNX SDK](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip). Its export metadata specifies 192×256 input and `simcc_x`/`simcc_y` outputs.

Review each upstream license and model card before redistribution or operational deployment.
