# -*- coding: utf-8 -*-
"""
MaixCAM / MaixCAM2 实时人脸检测 / 识别 Demo
========================================================
在 MaixVision 里打开本文件 -> 点「运行」，屏幕上会实时框出人脸，
终端会打印帧率、检出数量以及最终判定结论。

特点：
  * 自动挑模型：yolov8n_face > retinaface > face_detector
  * 如果有特征模型 + nn.FaceRecognizer，自动升级成“认人”模式
  * 没有屏幕也能跑（只是不显示，终端照样输出统计）
  * 单帧耗时 / 平均 FPS / 最大同时人脸数 都会打印出来

想改参数直接改下面的 CONFIG。
"""

from maix import app, image, nn
import os
import glob
import gc
import time

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DETECT_CONF = 0.5      # 检测阈值，越大越严格
IOU_TH = 0.45          # 重叠框过滤阈值
FEATURE_TH = 0.85      # 认人相似度阈值（越大越严格）
SHOW_KEYPOINTS = True  # 是否画五官关键点
VERDICT_AFTER = 5.0    # 跑满多少秒后打印一次“能力结论”
AUTO_EXIT = 0.0        # >0 时跑够这么多秒自动退出（0 = 一直跑，手动停止）

LINE = "=" * 58


def log(msg):
    print(msg)


# ------------------------------------------------------------------
# 1. 找模型
# ------------------------------------------------------------------
def scan(keywords, exclude=()):
    """在 /root/models 下按关键字找 .mud，返回按优先级排好的列表"""
    found = []
    for p in sorted(glob.glob("/root/models/**/*.mud", recursive=True)):
        low = os.path.basename(p).lower()
        if any(k in low for k in exclude):
            continue
        if any(k in low for k in keywords):
            found.append(p)
    return found


log(LINE)
log(" MaixCAM 人脸检测 / 识别 Demo")
log(LINE)

det_candidates = scan(["yolov8n_face", "yolov8s_face", "yolov8_face",
                       "retinaface", "face_detector"],
                      exclude=["feature", "insght", "insight", "webface", "arcface"])
feat_candidates = scan(["feature", "insght", "insight", "webface", "arcface"])

if not det_candidates:
    log("[ERROR] /root/models 下没有找到人脸检测模型。")
    log("        请先运行 1_check_env.py 看自检结果，或到 maixhub.com/model/zoo/407 下载。")
    raise SystemExit(1)

detect_path = det_candidates[0]
log("检测模型 : {}".format(detect_path))
log("特征模型 : {}".format(feat_candidates[0] if feat_candidates else "无（只做检测）"))

# ------------------------------------------------------------------
# 2. 建模型（失败自动降级）
# ------------------------------------------------------------------
def build(cls, **kwargs):
    try:
        return cls(**kwargs)
    except TypeError:
        kwargs.pop("dual_buff", None)
        try:
            return cls(**kwargs)
        except TypeError:
            return cls(kwargs.get("model"))


mode = "detect"
model = None
detector_cls = None

if feat_candidates and hasattr(nn, "FaceRecognizer"):
    try:
        model = build(nn.FaceRecognizer,
                      detect_model=detect_path,
                      feature_model=feat_candidates[0],
                      dual_buff=True)
        mode = "recognize"
        log("模式     : 人脸识别（检测 + 特征比对）")
        if os.path.exists("/root/faces.bin"):
            model.load_faces("/root/faces.bin")
            log("人脸库   : /root/faces.bin，已录入 {} 人".format(len(model.labels) - 1))
        else:
            log("人脸库   : 无（所有人脸都会显示 unknown，属正常现象）")
    except Exception as e:
        log("[WARN] 识别模式初始化失败，退回检测模式: {}".format(e))
        model = None

if model is None:
    low = os.path.basename(detect_path).lower()
    if "retina" in low:
        detector_cls = "Retinaface"
    elif "face_detector" in low:
        detector_cls = "FaceDetector"
    else:
        detector_cls = "YOLOv8"
    cls = getattr(nn, detector_cls, None)
    if cls is None:
        log("[ERROR] 本固件没有 nn.{}，请升级 MaixPy 固件（>= 4.3.8）".format(detector_cls))
        raise SystemExit(1)
    model = build(cls, model=detect_path, dual_buff=True)
    log("模式     : 人脸检测（{}）".format(detector_cls))

log(LINE)

# ------------------------------------------------------------------
# 3. 摄像头 / 屏幕
# ------------------------------------------------------------------
from maix import camera

cam = camera.Camera(model.input_width(), model.input_height(), model.input_format())

disp = None
try:
    from maix import display
    disp = display.Display()
except Exception as e:
    log("[WARN] 没有可用屏幕，仅输出到终端: {}".format(e))

# ------------------------------------------------------------------
# 4. 主循环
# ------------------------------------------------------------------
frame_cnt = 0
face_cnt_max = 0
t_start = time.time()
fps = 0.0
t_last_stat = t_start
verdict_done = False
sum_dt = 0.0

try:
    while not app.need_exit():
        img = cam.read()

        t0 = time.time()
        if mode == "recognize":
            objs = model.recognize(img, DETECT_CONF, IOU_TH, FEATURE_TH)
        elif detector_cls == "YOLOv8":
            objs = model.detect(img, conf_th=DETECT_CONF, iou_th=IOU_TH, keypoint_th=0.5)
        else:
            objs = model.detect(img, conf_th=DETECT_CONF, iou_th=IOU_TH)
        dt = time.time() - t0

        frame_cnt += 1
        sum_dt += dt
        n = len(objs)
        if n > face_cnt_max:
            face_cnt_max = n

        # ---- 画框 ----
        for o in objs:
            img.draw_rect(o.x, o.y, o.w, o.h, color=image.COLOR_RED)

            pts = getattr(o, "points", None)
            if SHOW_KEYPOINTS and pts:
                try:
                    if mode == "detect" and detector_cls != "Retinaface":
                        model.draw_pose(img, pts, 2, image.COLOR_GREEN)
                    else:
                        img.draw_keypoints(pts, image.COLOR_GREEN, size=4)
                except Exception:
                    pass

            if mode == "recognize":
                cid = o.class_id
                label = model.labels[cid] if cid < len(model.labels) else "?"
                color = image.COLOR_GREEN if cid != 0 else image.COLOR_YELLOW
                txt = "{} {:.2f}".format(label, o.score)
            else:
                color = image.COLOR_RED
                txt = "face {:.2f}".format(o.score)
            img.draw_string(o.x, o.y - 18 if o.y > 18 else o.y + 2, txt, color=color)

        # ---- 状态栏 ----
        fps = frame_cnt / max(1e-6, time.time() - t_start)
        try:
            img.draw_string(4, 4, "FPS %.1f  faces %d" % (fps, n), color=image.COLOR_GREEN)
        except Exception:
            pass

        if disp is not None:
            disp.show(img)

        # ---- 每秒一次终端统计 ----
        now = time.time()
        if now - t_last_stat >= 1.0:
            log("[stat] t=%4.0fs  FPS=%5.1f  单帧=%5.1fms  当前人脸=%d  最大=%d"
                % (now - t_start, fps, dt * 1000, n, face_cnt_max))
            t_last_stat = now

        # ---- 能力结论只打一次 ----
        if not verdict_done and (now - t_start) >= VERDICT_AFTER:
            verdict_done = True
            avg_ms = sum_dt / max(1, frame_cnt) * 1000
            log(LINE)
            log(" 能力判定（已连续处理 {} 帧）".format(frame_cnt))
            log("   平均单帧耗时 : {:.1f} ms  →  约 {:.1f} FPS".format(avg_ms, 1000 / max(1e-6, avg_ms)))
            log("   同时最多人脸 : {} 张".format(face_cnt_max))
            log("   结论         : {}".format(
                "✔ 这台设备可以正常运行人脸识别" if mode == "recognize"
                else "✔ 这台设备可以正常运行人脸检测（认人需再加特征模型）"))
            log(LINE)

        if AUTO_EXIT > 0 and (now - t_start) >= AUTO_EXIT:
            break

except KeyboardInterrupt:
    log("\n[INFO] 用户中断")
except Exception as e:
    log("\n[ERROR] 运行出错: {}: {}".format(type(e).__name__, e))
    import traceback
    traceback.print_exc()

finally:
    # ---- 结束汇总 ----
    total = max(1e-6, time.time() - t_start)
    log(LINE)
    log(" 运行结束汇总")
    log("   总帧数   : {}".format(frame_cnt))
    log("   总时长   : {:.1f} s".format(total))
    log("   平均帧率 : {:.1f} FPS".format(frame_cnt / total))
    log("   平均耗时 : {:.1f} ms/帧".format(sum_dt / max(1, frame_cnt) * 1000))
    log("   最多人脸 : {} 张".format(face_cnt_max))
    log("   模式     : {}".format("人脸识别" if mode == "recognize" else "人脸检测"))
    log("   模型     : {}".format(os.path.basename(detect_path)))
    log(LINE)
    del model
    gc.collect()
