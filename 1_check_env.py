# -*- coding: utf-8 -*-
"""
MaixCAM / MaixCAM2 「人脸识别能力」自检脚本
========================================================
用法：在 MaixVision 里打开本文件 -> 点「运行」-> 看下方终端输出。

它会依次检查：
  1) 设备型号 / 固件版本 / 内存 / IP
  2) maix.nn 里人脸相关的 API 是否齐全
  3) /root/models 下有哪些人脸模型文件
  4) 检测模型能否真正加载（这一步最关键）
  5) 摄像头能否打开，并实跑一帧推理（给出耗时 / 上限帧率）
  6) 识别模型（特征提取+比对）能否加载并跑通
最后打印结论：这台设备到底能不能做人脸检测 / 人脸识别。

注意：整个过程不会保存任何图片，也不会上传任何数据。
"""

import os
import glob
import gc
import time

LINE = "=" * 62
result = {"camera": False, "detect": False, "recognize": False}


def head(title):
    print("\n" + LINE)
    print(title)
    print(LINE)


def kv(key, value):
    print("  {:<14}{}".format(key + ":", value))


def ok(msg):
    print("  [ OK ] " + msg)


def bad(msg):
    print("  [FAIL] " + msg)


def warn(msg):
    print("  [WARN] " + msg)


# ---------------------------------------------------------------
# 1. 系统信息
# ---------------------------------------------------------------
head("1. 系统信息")
try:
    from maix import sys as msys
except Exception as e:
    msys = None
    warn("maix.sys 导入失败: {}".format(e))

if msys:
    for label, fn in (("设备 ID", msys.device_id),
                      ("设备名称", msys.device_name),
                      ("MaixPy 版本", msys.maixpy_version),
                      ("系统版本", msys.os_version)):
        try:
            kv(label, fn())
        except Exception as e:
            kv(label, "读取失败: {}".format(e))
    try:
        m = msys.memory_info()
        kv("内存", "{} 总 / {} 已用".format(
            msys.bytes_to_human(m.get("total", 0)),
            msys.bytes_to_human(m.get("used", 0))))
    except Exception:
        pass
    try:
        kv("IP", ", ".join("{}={}".format(k, v) for k, v in msys.ip_address().items()))
    except Exception:
        pass

# ---------------------------------------------------------------
# 2. maix.nn 能力
# ---------------------------------------------------------------
head("2. maix.nn 人脸相关 API")
try:
    from maix import nn
except Exception as e:
    bad("maix.nn 导入失败: {}".format(e))
    raise SystemExit(1)

caps = {}
for name in ("FaceDetector", "Retinaface", "YOLOv8", "YOLOv5", "FaceRecognizer"):
    caps[name] = hasattr(nn, name)
    print("  {:<14}{}".format(name + ":", "可用" if caps[name] else "不可用"))

# ---------------------------------------------------------------
# 3. 模型文件
# ---------------------------------------------------------------
head("3. 模型文件 (/root/models)")
models = sorted(glob.glob("/root/models/**/*.mud", recursive=True))
if not models:
    warn("没有找到任何 .mud 文件，请先把模型放到 /root/models/ 下")
else:
    print("  共 {} 个模型文件:".format(len(models)))
    for p in models:
        try:
            size = os.path.getsize(p) / 1024.0 / 1024.0
            print("    - {:<44} {:6.1f} MB".format(os.path.basename(p), size))
        except Exception:
            print("    - {}".format(os.path.basename(p)))

detect_models = []
feature_models = []
for p in models:
    low = os.path.basename(p).lower()
    if any(k in low for k in ("feature", "insght", "insight", "webface", "arcface")):
        feature_models.append(p)
    elif any(k in low for k in ("retina", "face")):
        detect_models.append(p)

# 优先用 yolov8n_face（效果最好），其次 retinaface，最后 face_detector
detect_models.sort(key=lambda p: (0 if "yolov8" in os.path.basename(p).lower()
                                  else 1 if "retina" in os.path.basename(p).lower()
                                  else 2))

if detect_models:
    print("\n  识别到【人脸检测】模型:")
    for p in detect_models:
        print("    - {}".format(p))
else:
    warn("没有人脸检测模型（名字里含 face / retina 的 .mud）")

if feature_models:
    print("\n  识别到【人脸特征】模型:")
    for p in feature_models:
        print("    - {}".format(p))
else:
    warn("没有特征提取模型，只能做“检测”，做不了“认出是谁”")

# ---------------------------------------------------------------
# 4. 尝试加载检测模型
# ---------------------------------------------------------------
head("4. 加载人脸检测模型")


def guess_cls(path):
    low = os.path.basename(path).lower()
    if "retina" in low:
        return "Retinaface"
    if "face_detector" in low:
        return "FaceDetector"
    return "YOLOv8"


def load_model(cls_name, path, **extra):
    """兼容不同固件版本的构造方式：先带参数，不行再逐级去掉"""
    cls = getattr(nn, cls_name, None)
    if cls is None:
        print("  - {:<12} 本固件没有这个 API，跳过".format(cls_name))
        return None

    attempts = []
    if extra:
        attempts.append(lambda: cls(model=path, **extra))
    attempts.append(lambda: cls(model=path))
    if extra:
        attempts.append(lambda: cls(path, **extra))
    attempts.append(lambda: cls(path))

    last_err = None
    for make in attempts:
        t0 = time.time()
        try:
            m = make()
            ok("{} 加载成功 <- {}  ({:.0f} ms)".format(
                cls_name, os.path.basename(path), (time.time() - t0) * 1000))
            return m
        except TypeError as e:
            last_err = e      # 参数不兼容，换下一种写法
            continue
        except Exception as e:
            bad("{} 加载 {} 失败: {}".format(cls_name, os.path.basename(path), e))
            return None

    bad("{} 加载 {} 失败: {}".format(cls_name, os.path.basename(path), last_err))
    return None


detector = None
detect_path = None
detect_cls = None
for p in detect_models:
    cls_name = guess_cls(p)
    if not caps.get(cls_name):
        continue
    detector = load_model(cls_name, p, dual_buff=True)
    if detector is not None:
        detect_path, detect_cls = p, cls_name
        result["detect"] = True
        break

if detector is None:
    bad("没有任何人脸检测模型能加载成功")

# ---------------------------------------------------------------
# 5. 摄像头 + 单帧推理
# ---------------------------------------------------------------
head("5. 摄像头与推理实测")
img = None
if detector is not None:
    try:
        from maix import camera
        w, h, fmt = detector.input_width(), detector.input_height(), detector.input_format()
        kv("模型输入", "{} x {}  {}".format(w, h, fmt))
        cam = camera.Camera(w, h, fmt)
        img = cam.read()
        result["camera"] = True
        ok("摄像头打开成功，读到一帧图像")
        t0 = time.time()
        if detect_cls == "YOLOv8":
            objs = detector.detect(img, conf_th=0.5, iou_th=0.45, keypoint_th=0.5)
        else:
            objs = detector.detect(img, conf_th=0.5, iou_th=0.45)
        dt = (time.time() - t0) * 1000
        ok("推理成功: 检出 {} 张人脸，单帧 {:.0f} ms（理论上限 {:.1f} FPS）".format(
            len(objs), dt, 1000.0 / dt if dt > 1 else 0))
        if len(objs) > 0:
            o = objs[0]
            kv("首个人脸", "x={} y={} w={} h={} score={:.2f}".format(
                o.x, o.y, o.w, o.h, o.score))
        del cam
        gc.collect()
    except Exception as e:
        bad("摄像头/推理环节失败: {}: {}".format(type(e).__name__, e))
        print("       常见原因：摄像头被别的程序占用（先关掉其它预览/App），或模型和芯片不匹配")
else:
    # 没有检测模型时，单独测一下摄像头，避免把“模型缺失”误报成“摄像头坏”
    try:
        from maix import camera
        cam = camera.Camera(320, 240)
        img = cam.read()
        result["camera"] = True
        ok("摄像头可用（320x240 读帧成功），但缺少人脸检测模型")
        del cam
        gc.collect()
    except Exception as e:
        bad("摄像头打不开: {}: {}".format(type(e).__name__, e))
        print("       常见原因：摄像头被别的程序占用，或排线/驱动异常")

# ---------------------------------------------------------------
# 6. 人脸识别（特征比对）
# ---------------------------------------------------------------
head("6. 人脸识别（检测 + 特征比对）")
if not (detect_path and feature_models):
    warn("缺少检测模型或特征模型，跳过这一步")
else:
    if not caps.get("FaceRecognizer"):
        warn("本固件没有 nn.FaceRecognizer，只能做检测")
    else:
        # 先释放检测器，MaixCAM 内存有限，不同时驻留两个模型
        det_saved = detector
        detector = None
        del det_saved
        gc.collect()
        try:
            recognizer = nn.FaceRecognizer(detect_model=detect_path,
                                           feature_model=feature_models[0],
                                           dual_buff=True)
            ok("FaceRecognizer 加载成功（{} + {}）".format(
                os.path.basename(detect_path), os.path.basename(feature_models[0])))
            if os.path.exists("/root/faces.bin"):
                recognizer.load_faces("/root/faces.bin")
                ok("已加载人脸库 /root/faces.bin，当前库中人数: {}".format(
                    len(recognizer.labels) - 1))
            else:
                warn("没有人脸库 /root/faces.bin，识别结果都会是 unknown（这是正常的）")

            from maix import camera
            # 如果第 5 步没拿到图，这里重新开一次摄像头
            try:
                cam2 = camera.Camera(recognizer.input_width(),
                                     recognizer.input_height(),
                                     recognizer.input_format())
                frame = cam2.read()
                del cam2
                gc.collect()
            except Exception:
                frame = img
            if frame is None:
                raise RuntimeError("没有可用的图像帧")
            t0 = time.time()
            faces = recognizer.recognize(frame, 0.5, 0.45, 0.85)
            dt = (time.time() - t0) * 1000
            ok("识别流程跑通: 检出 {} 张脸，单帧 {:.0f} ms".format(len(faces), dt))
            for f in faces:
                label = recognizer.labels[f.class_id] if f.class_id < len(recognizer.labels) else "?"
                print("        - {}  score={:.2f}".format(label, f.score))
            result["recognize"] = True
            result["camera"] = True
        except Exception as e:
            bad("识别环节失败: {}: {}".format(type(e).__name__, e))

# ---------------------------------------------------------------
# 7. 结论
# ---------------------------------------------------------------
head("7. 结论")
print("  摄像头          : {}".format("✔ 可用" if result["camera"] else "✘ 不可用"))
print("  人脸检测        : {}".format("✔ 可用" if result["detect"] else "✘ 不可用"))
print("  人脸识别(认人)  : {}".format("✔ 可用" if result["recognize"] else "✘ 不可用"))
print("")

if result["detect"] and result["recognize"]:
    print("  ==> 这台设备完全可以做人脸识别，直接运行 2_face_demo.py 看实时效果。")
elif result["detect"]:
    print("  ==> 能做【人脸检测】，但【认人】还差一个特征模型。")
    print("      去 MaixHub 下载 insightface 特征模型（.mud），放到 /root/models/ 即可。")
    print("      也可以直接运行 2_face_demo.py，它会自动只用检测模式。")
else:
    print("  ==> 目前还做不了人脸检测，请检查：")
    print("      1. 模型文件是否在 /root/models/ 下（可用 MaixVision 的设备文件面板上传）")
    print("      2. 模型是否和你这块板子匹配（MaixCAM 与 MaixCAM2 的模型不通用）")
    print("      3. 固件版本是否 >= 4.3.8（老固件没有 YOLOv8-face）")
    print("      下载地址：https://maixhub.com/model/zoo/407 （YOLOv8-face）")

print("\n自检结束。")
