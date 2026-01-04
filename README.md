# FishCounter

Manual fish counting in production is tedious, slow, and inaccurate. Existing solutions are often:
- Expensive - require specialized equipment
- Slow - cannot handle high fish density or struggle with rapidly changing scenes

Our project solves these problems! We are developing an affordable and fast algorithm that works on a standard computer (Core i5) and processes 500+ frames per second.

## What do we do?
Task: Teach a computer to count fish in surveillance camera video in real-time.
Conditions:
- Standard cameras (not super expensive)
- Core i5 processor
- Processing speed: 500+ FPS
- Real-world conditions: changing light, many fish in the frame, static camera

## How it works?
## Detection
### Step 1: Separate fish from the background (MOG2)
MOG2 models each pixel as a mixture of several Gaussian distributions that are continuously updated. The algorithm compares new pixel values with existing distributions, classifying them as background or foreground. This allows adaptation to lighting changes and separates moving fish from a static background in real time.
- Each pixel in the video is analyzed uniquely
- The algorithm adapts to lighting changes automatically
- Result: Clear highlighting of moving fish
mog2

<img width="850" height="640" alt="mog2" src="https://github.com/user-attachments/assets/59b51e96-48cf-4061-800c-c6d01c023fe7" />

### Step 2: Improve the image (CLAHE)
- Intelligent contrast enhancement even in murky water
- Works with individual sections of the frame
- Result: Fish become more visible

### Step 3: Clean and merge (Morphological Operations)
- Remove "noise" and small artifacts
- Fill holes in fish contours
- Result: Neat and solid objects for counting

### Work Example
<img width="1601" height="636" alt="fish_results" src="https://github.com/user-attachments/assets/e86b0ab9-d1f3-4ba7-ad19-b0834695a70f" />


# Final Results
<img width="1037" height="406" alt="image" src="https://github.com/user-attachments/assets/04b5c824-39da-465b-9a01-cd28c7c31dc1" />


