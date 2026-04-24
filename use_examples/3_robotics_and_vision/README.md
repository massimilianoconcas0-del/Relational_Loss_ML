# 🤖 Robotics & Vision: Interaction with the Physical World

Welcome to the Robotics and Computer Vision section of the Relational Calculus framework. 

In these domains, AI systems must interact with a variable and often unpredictable physical environment. Traditional models fail because they are "scale-locked": they learn to operate within the specific constraints of their training environment (e.g., a specific drone mass or a specific indoor lighting intensity). When these constraints change in the real world, the model's logic breaks.

This directory provides two high-impact demonstrations of how **Relational Invariance** solves the Sim2Real gap and the illumination drift problem.

## 🗂️ The Experiments

These scripts demonstrate zero-shot adaptation to hardware changes and extreme environmental shifts.

### 1. `robotics_zero_shots.py`
**The Problem:** Sim2Real transfer and payload variance. How do you train a flight controller on a small drone and expect it to fly a much larger one?
**The Absolute Trap:** Standard controllers predict the raw **Thrust in Newtons**. Because Force is tied to Mass ($F=ma$), a controller trained on a 1kg drone will predict 10 Newtons to hover. If you put those weights on a 50kg drone, it won't even lift off the ground.

**The Relational Fix:** The network is forced to predict the **Thrust-to-Weight Ratio [0,1]**. 
* **What you will see:** An **8,900x lower MSE**. The Relational Model, despite being trained *only* on the 1kg drone, flawlessly calculates the power needed for the 50kg drone by understanding the *relationship* between gravity and thrust, rather than memorizing a fixed number of Newtons.

### 2. `vision_hdr_inverse_rendering.py`
**The Problem:** Neural Radiance Fields (NeRFs) and visual models suffer from "baked-in" lighting. If you train a model indoors and move it outdoors, the absolute RGB values jump by 500x.
**The Absolute Trap:** Standard vision models predict **RGB Pixel Intensity (0-255)**. Under HDR (High Dynamic Range) sunlight, the model produces "white-out" artifacts or absolute garbage because the input scale is outside its known memory.

**The Relational Fix:** We decouple the light source from the material, forcing the network to predict the **Intrinsic Albedo (Reflectance Ratio [0,1])**.
* **What you will see:** A **153x improvement** in rendering accuracy. The model ignores the blinding sun and focuses on the material's relational property. It renders perfect textures regardless of whether it's under a candle or the midday sun.

## 🚀 How to Run

These examples use synthetic environments to simulate physical sensors and camera outputs.

Ensure you have the required libraries:
```bash
pip install numpy scikit-learn matplotlib
```

Run the robotics or vision demo:
```bash
python robotics_zero_shots.py
```
*(Each script generates a comparison plot showing the 'Scale-Locked' failure of absolute models vs. the 'Scale-Invariant' success of the relational approach).*
