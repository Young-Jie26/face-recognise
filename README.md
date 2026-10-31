# MaixCAM 人脸识别能力测试

两个脚本，配合 MaixVision 使用。

| 文件 | 作用 | 需要摄像头 |
|---|---|---|
| `1_check_env.py` | 自检：型号/固件/模型/API/能否真正加载并推理，最后给结论 | 可选 |
| `2_face_demo.py` | 实时 demo：屏幕框人脸、终端打印 FPS 与结论 | 需要 |

## 怎么跑

1. MaixVision 连接设备（同一 WiFi，或 USB）。
2. `文件 → 打开`，选中 `1_check_env.py`，点右上角 **运行**。
3. 看 MaixVision 下方 **终端** 输出，重点看第 7 节「结论」。
4. 再打开 `2_face_demo.py` 运行，屏幕上应能看到红框实时跟脸。

> 也可以直接把代码复制粘贴到 MaixVision 的编辑器里运行，效果一样。

## 1_check_env.py 输出长什么样

```
7. 结论
  摄像头          : ✔ 可用
  人脸检测        : ✔ 可用
  人脸识别(认人)  : ✔ 可用
```

- **检测 ✔、识别 ✘**：能做"找到脸"，但缺特征模型，不能"认出是谁"。
- **两个都 ✘**：模型缺失或与芯片不匹配，第 3、4 节会打印具体原因。

## 模型从哪来

| 模型 | 用途 | 下载 | 放置位置 |
|---|---|---|---|
| `yolov8n_face.mud` | 人脸检测（推荐，效果好） | https://maixhub.com/model/zoo/407 | `/root/models/` |
| `retinaface.mud` | 人脸检测（快） | https://maixhub.com/model/zoo/378 | `/root/models/` |
| `face_detector.mud` | 人脸检测（最轻量 1MB） | https://maixhub.com/model/zoo/377 | `/root/models/` |
| `insghtface_webface_r50.mud` | 人脸特征（认人） | https://maixhub.com/model/zoo/462 | `/root/models/` |

- 下载页面要按 **硬件平台筛选**（maixcam / maixcam2），两类板子的模型不通用。
- 上传方式：MaixVision 左侧「设备文件」面板进入 `/root/models/`，把解压出来的 `.mud` 拖进去。
- 固件自带的应用里通常已经带了这几个模型，先跑自检看看，缺了再下。

## 常见报错对照

| 现象 | 原因 | 处理 |
|---|---|---|
| `no model file` / `open failed` | 路径不对或文件没传上去 | 确认在 `/root/models/`，文件名和脚本里一致 |
| `load model failed` + 型号不匹配 | MaixCAM 用了 MaixCAM2 的模型（或反之） | 回 MaixHub 按平台重新下 |
| 摄像头打不开 | 被别的 App / 预览窗口占用 | 关掉其它用摄像头的应用再跑 |
| `AttributeError: FaceRecognizer` | 固件偏低 | 升级 MaixPy（≥ 4.3.8 才有 YOLOv8-face / FaceRecognizer） |
| 帧率很低（< 5 FPS） | 模型太大或分辨率过高 | 换 `face_detector`，或换低分辨率模型 |
| 识别全是 `unknown` | 没人脸库 | 正常；要认人得先录入（见下） |

## 想更进一步：录入自己的人脸

`2_face_demo.py` 跑通后，在识别模式下可以用 `recognizer.add_face(face, "你的名字")`
把人脸加进库，再用 `recognizer.save_faces("/root/faces.bin")` 保存。
需要的话我可以把这个"按键录入 + 保存"的流程也写好给你。

## 说明

- 两个脚本都不保存图片、不上传任何数据，只做本地推理和终端打印。
- `1_check_env.py` 里对 API 做了兼容处理（`dual_buff` 参数、旧固件缺类），
  所以同一份代码在 MaixCAM 和 MaixCAM2 上都能跑，报错也能看出卡在哪一步。
