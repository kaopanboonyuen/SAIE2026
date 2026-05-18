# 🚀 SAIE 2026 — AI in the Real World
## Trade-offs Behind Fast & Scalable Object Detection

🎓 **Super AI Engineer Thailand 2026 (SAIE 2026)**  
🧠 Real-World Computer Vision Workshop Series  
⚡ From Research → Optimization → Deployment  

---

## 👨‍💻 Author

**Teerapong Panboonyuen, Ph.D. (P'Kao)**  
📍 Chulalongkorn University, Thailand  
📫 `teerapong [dot] pa [at] chula [dot] ac [dot] th`

🌐 Repository  
https://github.com/kaopanboonyuen/SAIE2026

---

## 👨‍🏫 Lecture Materials

📄 Lecture Slides  
[![View Lecture Slides](https://img.shields.io/badge/View-Lecture_Slides-red?style=for-the-badge)](https://github.com/kaopanboonyuen/SAIE2026/blob/main/slides/SAIE2026_ObjectDetection_PanboonyuenLecture.pdf)

📓 Google Colab Notebook (Student Version)  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaopanboonyuen/SAIE2026/blob/main/notebooks/SAIE2026_ObjectDetection_Workshop_toStudent.ipynb)

✅ Google Colab Notebook (With Solutions)  
[![Open Solution Notebook](https://img.shields.io/badge/Open-Solution_Notebook-success?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/github/kaopanboonyuen/SAIE2026/blob/main/notebooks/SAIE2026_ObjectDetection_Workshop_withSolution.ipynb)

---

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)
![YOLO](https://img.shields.io/badge/YOLO-ObjectDetection-green.svg)
![GPU](https://img.shields.io/badge/GPU-Accelerated-orange.svg)
![License](https://img.shields.io/badge/License-Educational-lightgrey.svg)

---

## 🔧 Useful Resources for Jetson Optimization (Real-Time Object Detection)

This section collects practical resources for deploying and optimizing AI models on **NVIDIA Jetson devices**, especially for **real-time object detection, multi-camera systems, and edge AI applications**.

| Resource | What it is for | Key Tricks / Why it matters |
|----------|----------------|-----------------------------|
| 🚀 Jetson AI Lab<br>https://www.jetson-ai-lab.com/ | Practical Jetson tutorials and deployment guides | End-to-end Jetson workflows, YOLO deployment, TensorRT optimization, real edge AI examples |
| ⚡ Ultralytics Jetson Guide<br>https://docs.ultralytics.com/guides/nvidia-jetson | YOLO deployment on Jetson (modern pipeline) | Export YOLO → TensorRT, FP16/INT8 acceleration, FPS benchmarking, easy deployment |
| 🎥 DeepStream + Jetson Guide<br>https://docs.ultralytics.com/guides/deepstream-nvidia-jetson | Multi-stream inference with YOLO | Real-time multi-camera pipelines, scalable inference, production-level AI systems |
| 🧠 NVIDIA Blog<br>https://developer.nvidia.com/blog/ | Engineering insights from NVIDIA | Search: “TensorRT optimization”, “Jetson object detection”, “DeepStream pipeline” → real performance tuning tricks |
| 📡 DeepStream SDK Docs<br>https://docs.nvidia.com/metropolis/deepstream/dev-guide/ | Full production AI video analytics framework | Multi-camera AI, streaming pipelines, GPU acceleration, real-time inference architecture |
| ⚡ TensorRT Documentation<br>https://docs.nvidia.com/deeplearning/tensorrt/ | Core inference optimization engine | Graph fusion, layer optimization, FP32 → FP16 → INT8 conversion, latency reduction techniques |
| 🧪 DeepStream Python Apps (GitHub)<br>https://github.com/NVIDIA-AI-IOT/deepstream_python_apps | Python-based DeepStream examples | Multi-stream video processing, RTSP camera pipelines, scalable inference demos |
| ⚙️ TensorRT GitHub<br>https://github.com/NVIDIA/TensorRT | Low-level inference optimization toolkit | Custom plugins, deployment examples, advanced optimization and inference acceleration |

---

# 🌏 Overview

Welcome to the official workshop repository for:

> **AI in the Real World: Trade-offs Behind Fast & Scalable Object Detection**

This lecture and hands-on workshop are designed for the next generation of AI engineers participating in **SAIE 2026 — Super AI Engineer Thailand**.

Unlike typical tutorials focused only on accuracy, this workshop explores the *real engineering trade-offs* behind deploying object detection systems in production environments:

- ⚡ Latency vs Accuracy
- 💾 Model Size vs Performance
- 🔋 Edge Deployment Constraints
- 📹 Multi-Camera Scalability
- 🧠 Knowledge Distillation
- 🪶 Structured Pruning
- 🔢 Quantization (INT8 / FP16)
- 🚀 Real-time Deployment Optimization

Students will experience how modern AI systems are engineered under practical limitations — exactly like in industry-scale AI products.

---

# 📂 Repository Structure

```bash
SAIE2026/
│
├── slides/
│   └── SAIE2026_ObjectDetection_PanboonyuenLecture.pdf
│
├── notebooks/
│   └── SAIE2026_ObjectDetection_Workshop_toStudent.ipynb
│
├── datasets/
│
└── README.md
```

---

# 📘 Lecture Materials

## 🎤 Main Lecture Slides

[![View Lecture Slides](https://img.shields.io/badge/View-Lecture_Slides-red?style=for-the-badge)](https://github.com/kaopanboonyuen/SAIE2026/blob/main/slides/SAIE2026_ObjectDetection_PanboonyuenLecture.pdf)

Topics include:

- Evolution of Object Detection
- YOLO Family Overview
- Real-Time AI Systems
- AI Optimization Strategies
- Edge AI Deployment
- Industrial Computer Vision Pipelines
- Scalability Challenges
- Research → Production Transition

---

# 🧪 Workshop Curriculum

## ⭐ SAIE 2026 — Super AI Engineer Thailand

### 📚 Workshop Resources

📓 Google Colab Notebook (Student Version)  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaopanboonyuen/SAIE2026/blob/main/notebooks/SAIE2026_ObjectDetection_Workshop_toStudent.ipynb)

✅ Google Colab Notebook (With Solutions)  
[![Open Solution Notebook](https://img.shields.io/badge/Open-Solution_Notebook-success?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/github/kaopanboonyuen/SAIE2026/blob/main/notebooks/SAIE2026_ObjectDetection_Workshop_withSolution.ipynb)

---

### 🛠️ SETUP — Environment, Dataset & Pretrained Model
- Python Environment
- GPU Setup
- Dataset Preparation
- Loading Pretrained Models
- Experiment Structure

---

## ⚡ Lab 1 — Baseline Profiling: Latency, FLOPs & mAP

Understand the hidden computational costs behind object detection systems.

### Topics
- Benchmarking
- FLOPs Analysis
- FPS Measurement
- mAP Evaluation
- Throughput vs Accuracy

### Homework
- Compare different YOLO variants
- Analyze inference bottlenecks

---

## ✂️ Lab 2 — Structured Pruning: Removing What Doesn't Matter

Reduce model complexity while preserving performance.

### Topics
- Channel Pruning
- Structured Sparsity
- Compression Trade-offs
- Lightweight Inference

### Homework
- Build your own compressed detector
- Evaluate accuracy degradation

---

## 🔢 Lab 3 — Post-Training Quantization: INT8 & FP16

Deploy faster models with lower precision arithmetic.

### Topics
- FP32 vs FP16 vs INT8
- TensorRT Concepts
- Edge AI Optimization
- Hardware-aware Inference

### Homework
- Benchmark quantized models
- Compare latency gains

---

## 🧠 Lab 4 — Knowledge Distillation: Teacher → Student

Train compact models using large expert networks.

### Topics
- Teacher-Student Learning
- Soft Labels
- Distillation Loss
- Efficient Deployment

### Homework
- Distill a lightweight detector
- Compare student performance

---

## 🏗️ Lab 5 — Multi-Scale Heads & Architecture Design

Design detection heads for scalable AI systems.

### Topics
- Feature Pyramid Networks
- Multi-Scale Detection
- Backbone Design
- Real-Time Architecture Trade-offs

### Homework
- Modify detection heads
- Analyze scale sensitivity

---

## 📹 Lab 6 — Multi-Camera Scalability & Deployment

Move from single-model demos to production-scale AI systems.

### Topics
- Multi-Camera Pipelines
- Parallel Inference
- GPU Scheduling
- Deployment Architecture
- Stream Processing

### Final Hackathon Challenge
Build an optimized scalable detection system under real-world constraints.

---

## 🎯 SAIE 2026 — Workshop Summary

By the end of this workshop, students will understand:

✅ How modern object detection systems work  
✅ How to optimize AI models for deployment  
✅ How to scale AI systems in production  
✅ How to balance speed, accuracy, and cost  
✅ How research concepts translate into industrial AI products  

---

# 🚀 Getting Started

## 1️⃣ Clone Repository

```bash
git clone https://github.com/kaopanboonyuen/SAIE2026.git
cd SAIE2026
```

---

## 2️⃣ Open Workshop Notebook

### 🔗 Google Colab

https://colab.research.google.com/github/kaopanboonyuen/SAIE2026/blob/main/notebooks/SAIE2026_ObjectDetection_Workshop_toStudent.ipynb

### 🔗 Local Notebook

```bash
notebooks/SAIE2026_ObjectDetection_Workshop_toStudent.ipynb
```

---

## 3️⃣ Run Experiments

Students are encouraged to:

- Modify architectures
- Profile latency
- Optimize inference
- Experiment with pruning & quantization
- Deploy lightweight models

---

# 🧠 Educational Philosophy

This workshop is intentionally designed beyond academic toy examples.

Students will encounter:
- Real deployment constraints
- Engineering trade-offs
- Production-oriented optimization
- Industrial AI thinking

Because:

> “A model that is accurate but unusable in production is not enough.”

---

# 👨‍🏫 Lecturer

## 🎓 Teerapong Panboonyuen, Ph.D. (P'Kao)

📫 Contact:

```text
teerapong [dot] pa [at] chula [dot] ac [dot] th
```

---

# 💡 Recommended Background

Students will benefit from:
- Basic Python knowledge
- Introductory Deep Learning concepts
- Fundamental Computer Vision understanding

No prior optimization experience required.

---

# ⚠️ Disclaimer

This repository and all workshop materials are provided **strictly for educational and research purposes only**.

The content is intended to:
- Teach AI engineering concepts
- Demonstrate optimization techniques
- Support academic learning

Users are responsible for ensuring ethical and lawful use of AI technologies.

The lecturer and contributors are not responsible for:
- Misuse of models or code
- Unauthorized deployments
- Harmful or unethical applications

---

# 🙏 Acknowledgements

Special thanks to:

- SAIE 2026 Organizing Team
- Open-source AI community
- PyTorch & Ultralytics contributors
- Students and researchers advancing efficient AI systems

---

# 📚 Citation

If you find this lecture, workshop, repository, or educational material useful in your research, teaching, or projects, please consider citing:

```bibtex
@misc{panboonyuen2026saie,
  title        = {AI in the Real World: Trade-offs Behind Fast and Scalable Object Detection},
  author       = {Teerapong Panboonyuen},
  year         = {2026},
  howpublished = {\url{https://github.com/kaopanboonyuen/SAIE2026}},
  note         = {SAIE 2026 -- Super AI Engineer Thailand Workshop on Efficient Object Detection}
}
```

---

<div align="center">

🚀 **Build AI That Works In The Real World**

Fast • Efficient • Scalable • Deployable

⭐ If this workshop helps you, consider starring the repository!

</div>