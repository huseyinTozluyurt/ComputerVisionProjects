# OpenCV Center Line and Center Box Drawing

This project contains two simple OpenCV programs for practicing image loading, resizing, center coordinate calculation, and drawing guide lines or boxes on an image.

<img width="1815" height="893" alt="image" src="https://github.com/user-attachments/assets/84580b36-e028-4739-b458-b435f9880380" />

## Files

### 1. Draw Center without Parameters

This version loads an image, resizes it, finds the center of the image, and draws lines manually using fixed values.

Main ideas practiced:

* Loading an image with `cv2.imread()`
* Resizing an image with `cv2.resize()`
* Getting image dimensions using `img.shape`
* Finding center coordinates
* Drawing lines with `cv2.line()`
* Displaying an image with `cv2.imshow()`

### 2. Draw Center with Margin and Paading

This version improves the code by using parameters for the center box.

It uses:

```python
center_margin = 50
center_padding = 0.25
```

`center_margin` controls the horizontal distance from the center.

`center_padding` controls the vertical empty space from the top and bottom of the image.

This makes the center box easier to adjust.

## Requirements

Install OpenCV:

```bash
pip install opencv-python
```

## How to Run

Make sure your image exists in the correct folder:

```text
DATA/elevator.jpg
```

Then run the Python file:

```bash
python3 main.py
```

## What the Program Does

The program:

1. Loads an image
2. Resizes the image
3. Finds the center x and y coordinates
4. Draws a center guide box or center line
5. Displays the result

## Example Concept

The edited version creates a center region of interest like this:

```python
left_x = center_x - center_margin
right_x = center_x + center_margin

top_y = int(height * center_padding)
bottom_y = int(height * (1 - center_padding))
```

Then it draws the box:

```python
cv2.rectangle(img, (left_x, top_y), (right_x, bottom_y), (0, 0, 255), 3)
```

## Purpose

This project is useful for learning basic Computer Vision concepts with Python and OpenCV. It can also be a starting point for robot camera alignment, lane correction, or detecting whether an object is inside the center region of the camera frame.
