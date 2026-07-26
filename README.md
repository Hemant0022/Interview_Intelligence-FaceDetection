# 🎯 Interview Intelligence System

## 📌 Project Overview

The Interview Intelligence System is an AI-powered computer vision application designed to monitor a candidate during an online interview. The objective is **not emotion recognition**, but to determine whether the candidate is attentive to the interview or distracted by unauthorized behavior.

The system continuously analyzes the webcam feed and provides structured outputs regarding:

- Candidate attention
- Screen focus
- Head direction
- Face detection
- Multiple person detection
- Phone detection
- Face/eye occlusion
- Candidate behavior

The output is generated as a structured JSON and will be visualized in a Streamlit dashboard.

---

# 🎯 Project Goal

Instead of detecting emotions such as:

- Happy
- Sad
- Angry
- Surprise

the system focuses on interview-related parameters like:

- Is the candidate looking at the screen?
- Is the candidate distracted?
- Is another person present?
- Is a phone visible?
- Is the candidate's face blocked?
- Are the eyes visible?
- Is the candidate continuously looking away?

---

# Models and Approaches Explored

## Phase 1 — Emotion Detection

Initially the project focused on emotion recognition.

### Models Evaluated

### 1. DeepFace

Purpose

- Facial emotion recognition
- Face verification
- Age estimation
- Gender estimation

Pros

- Easy to use
- Supports multiple pretrained models

Cons

- Heavy
- Slow on CPU
- Not suitable for real-time interview monitoring

Decision

❌ Rejected

Reason

Interview intelligence does not require emotion detection.

---

### 2. PyFeat

Purpose

- Facial Action Units
- Emotion Detection
- Head Pose
- Face Landmarks

Pros

- Research oriented
- Rich facial analysis

Cons

- Heavy
- CPU intensive

Decision

❌ Rejected

Reason

Too heavy for real-time monitoring.

---

### 3. HSEmotionONNX

Purpose

Fast emotion recognition using ONNX.

Pros

- Fast inference
- Lightweight

Cons

- Only predicts emotions

Decision

❌ Removed

Reason

Emotion prediction is not useful for interview evaluation.

---

### 4. MobileNetV3 Emotion Models

Reference

https://huggingface.co/docs/timm/en/models/mobilenet-v3

Purpose

Lightweight emotion recognition.

Pros

- Faster than DeepFace

Cons

Still focuses on emotion classification.

Decision

❌ Not used

---

# Phase 2 — Face Detection

---

## YuNet

Model

OpenCV YuNet Face Detector

Purpose

Detect candidate face.

Advantages

- Extremely fast
- Lightweight
- CPU friendly
- High detection accuracy

Current Status

✅ Implemented

Outputs

- Face Bounding Box
- Face Presence
- Face Count

---

# Phase 3 — Face Mesh

---

## MediaPipe FaceMesh

Purpose

Extract facial landmarks.

Outputs

- 478 facial landmarks
- Eye landmarks
- Nose
- Mouth
- Jawline

Advantages

- Fast
- Real-time
- Highly accurate

Current Status

✅ Implemented

Used For

- Head Pose
- Eye Region
- Future Gaze Estimation
- Face Occlusion

---

# Phase 4 — Head Pose Estimation

Technique

OpenCV solvePnP()

Inputs

MediaPipe Facial Landmarks

Outputs

Yaw

Pitch

Roll

Current JSON

```json
{
  "yaw": 0,
  "pitch": 0,
  "roll": 0,
  "head_direction": "Center"
}
```

Current Status

✅ Basic Implementation

Known Issues

- Requires calibration
- False predictions
- Threshold tuning required

Future Improvement

- Canonical 3D Face Model
- MediaPipe Face Geometry

---

# Phase 5 — Object Detection

Model

YOLOv8n

Purpose

Detect

- Phone
- Person
- Other objects

Current Status

✅ Implemented

Current Output

```json
{
  "phone_detected": false,
  "person_count": 1
}
```

Future Work

- Detect only relevant objects
- Ignore unnecessary classes

---

# Phase 6 — Face / Eye Occlusion

Current Idea

Use

YOLO Object Detection

+

MediaPipe FaceMesh

to determine

- Object on Face
- Object on Eyes

Status

🚧 Planned

Future Output

```json
{
  "object_on_face": false,
  "object_on_eyes": false
}
```

---

# Phase 7 — Gaze Estimation

Current Status

❌ Not Implemented

Planned

Eye gaze estimation using FaceMesh eye landmarks.

Output

```json
{
  "gaze_direction": "Center"
}
```

Future Decision

Screen Attention

↓

Head Pose

+

Eye Gaze

↓

Focused / Distracted

---

# Phase 8 — Attention Engine

Current Status

Partial

Current Logic

Head Direction

↓

Focused

Future Logic

Head Pose

+

Eye Gaze

+

Eye Visibility

+

Face Presence

↓

Attention Score

---

# Phase 9 — Behavior Detection

Current Status

Partial

Current Parameters

```json
{
  "looking_left": false,
  "looking_right": false,
  "looking_down": false,
  "multiple_faces": false,
  "face_missing": false
}
```

Future Improvement

Continuous tracking

Example

Looking Left

↓

3 Seconds

↓

Looking Left = True

instead of

Single Frame

↓

True

---

# Phase 10 — Streamlit Dashboard

Planned

Live Webcam

Attention Panel

Behavior Panel

Object Panel

Quality Panel

JSON Viewer

Future Graphs

Timeline

Attention %

Distraction %

Phone Detection

Logs

---

# Current JSON Structure

```json
{
  "timestamp": "",

  "face": {
    "detected": false,
    "mesh_detected": false,
    "landmarks": 0,
    "visibility": "",
    "size": ""
  },

  "quality": {
    "lighting": "",
    "brightness": 0,
    "blur": ""
  },

  "attention": {
    "status": "",
    "score": 0,
    "head_direction": "",
    "gaze_direction": "",
    "yaw": 0,
    "pitch": 0,
    "roll": 0,
    "looking_at_screen": false
  },

  "behavior": {
    "looking_left": false,
    "looking_right": false,
    "looking_down": false,
    "multiple_faces": false,
    "face_missing": false
  },

  "objects": {
    "phone_detected": false,
    "person_count": 0,
    "object_on_face": false,
    "object_on_eyes": false
  },

  "system": {
    "fps": 0
  }
}
```

---

# Current Project Status

| Module | Status |
|----------|--------|
| YuNet Face Detection | ✅ Completed |
| MediaPipe FaceMesh | ✅ Completed |
| Head Pose | ✅ Basic |
| JSON Output | ✅ Completed |
| YOLO Object Detection | ✅ Basic |
| Phone Detection | ✅ Basic |
| Multiple Face Detection | ✅ Basic |
| Face Occlusion | 🚧 Planned |
| Eye Occlusion | 🚧 Planned |
| Eye Gaze | 🚧 Planned |
| Attention Engine | 🚧 Planned |
| Behavior Engine | 🚧 Planned |
| Streamlit Dashboard | 🚧 In Progress |

---

# Planned Roadmap

## Phase 1

✅ Face Detection

✅ Face Mesh

✅ Head Pose

✅ JSON

---

## Phase 2

Eye Gaze

Face Occlusion

Eye Occlusion

Phone Detection Improvement

---

## Phase 3

Attention Engine

Behavior Engine

Continuous Tracking

Decision Engine

---

## Phase 4

Professional Streamlit Dashboard

Timeline Analytics

Session Report

Interview Score

PDF Report Generation

---

# Tech Stack

Python

OpenCV

MediaPipe

YOLOv8

NumPy

ONNX Runtime

Streamlit

JSON

---

# Final Objective

Develop a lightweight, real-time Interview Intelligence System capable of monitoring candidate attention, distraction, unauthorized object usage, and interview behavior using computer vision and AI, with structured outputs and a professional monitoring dashboard suitable for interview analytics.