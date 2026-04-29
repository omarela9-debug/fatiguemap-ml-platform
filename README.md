# FatigueMap ML Platform

Real-time multimodal fatigue detection system using Meta Project Aria smart glasses, built with a production-style ML pipeline.

## Overview

FatigueMap is an AI/ML system designed to detect driver fatigue using live sensor data from Project Aria. The system combines eye-tracking, IMU motion signals, and fatigue scoring logic to estimate drowsiness in real time.

## Key Features

- Real-time blink rate monitoring
- PERCLOS / frame closure ratio fatigue metric
- IMU-based head nod and microsleep detection
- Flask API for live data streaming
- React dashboard for real-time visualization
- Modular ML pipeline structure for future model deployment

## System Architecture

```text
Project Aria Sensors
        ↓
Python Data Stream
        ↓
Feature Extraction
        ↓
Fatigue Scoring Model
        ↓
Flask API
        ↓
React Dashboard
