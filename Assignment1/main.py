import csv #type: ignore
import numpy as np  # type: ignore
import cv2 as cv  # type: ignore
import os #type: ignore
import sys

# Custom Grayscale Conversion
def convert_to_grayscale(image):
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

# Custom Gaussian Blur using 5x5 Kernel
def gaussian_blur(image):
    kernel = np.array([
        [1,  4,  7,  4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1,  4,  7,  4, 1]
    ]) / 273.0

    padded_img = np.pad(image, ((2, 2), (2, 2)), mode='reflect')
    blurred = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            blurred[i, j] = np.sum(padded_img[i:i+5, j:j+5] * kernel)

    return blurred

# Custom Sobel Edge Detection as a substitute for Canny
def sobel_edge_detection(image, low_threshold=50, high_threshold=200):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0,  0,  0],
                        [1,  2,  1]])

    padded_img = np.pad(image, ((1, 1), (1, 1)), mode='reflect')
    grad_x = np.zeros_like(image, dtype=np.float32)
    grad_y = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            grad_x[i, j] = np.sum(padded_img[i:i+3, j:j+3] * sobel_x)
            grad_y[i, j] = np.sum(padded_img[i:i+3, j:j+3] * sobel_y)

    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = (grad_mag / grad_mag.max() * 255).astype(np.uint8)

    # Apply thresholding to mimic Canny behavior
    edges = np.zeros_like(grad_mag)
    edges[grad_mag >= high_threshold] = 255 # Strong edges
    edges[(grad_mag >= low_threshold) & (grad_mag < high_threshold)] = 128  # Weak edges

    return edges

# Modified Load and Preprocess Function
def load_and_preprocess(image_path):
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    if image is None:
        print(f"Not found: {image_path}")
        return None, None
    blurred = gaussian_blur(convert_to_grayscale(image))
    
    edges = sobel_edge_detection(blurred, 50, 200)
    return image, edges

# Custom Hough Line Segment Detector (Replacement for cv.HoughLinesP)
def custom_hough_lines_p(edges, rho, theta, threshold, min_line_length, max_line_gap):
    height, width = edges.shape
    diag_len = int(np.sqrt(height**2 + width**2))
    rhos = np.arange(-diag_len, diag_len + 1, rho)
    thetas = np.arange(0, np.pi, theta)

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    edge_pixels = np.argwhere(edges == 255)

    # Voting
    for y, x in edge_pixels:
        for t_idx, theta_val in enumerate(thetas):
            r = int(x * np.cos(theta_val) + y * np.sin(theta_val))
            r_idx = np.argmin(np.abs(rhos - r))
            accumulator[r_idx, t_idx] += 1

    # Find indices where accumulator exceeds threshold
    lines = []
    for r_idx, t_idx in zip(*np.where(accumulator >= threshold)):
        r = rhos[r_idx]
        t = thetas[t_idx]

        # Extract lines from votes
        a = np.cos(t)
        b = np.sin(t)

        x0 = a * r
        y0 = b * r

        # Search along the line for points (line segments)
        line_pixels = []
        for i in range(-diag_len, diag_len):
            x = int(x0 + i * (-b))
            y = int(y0 + i * a)

            if 0 <= x < width and 0 <= y < height and edges[y, x] == 255:
                if line_pixels and (abs(x - line_pixels[-1][0]) > max_line_gap or abs(y - line_pixels[-1][1]) > max_line_gap):
                    if len(line_pixels) >= min_line_length:
                        lines.append([line_pixels[0][0], line_pixels[0][1], line_pixels[-1][0], line_pixels[-1][1]])
                    line_pixels = []

                line_pixels.append((x, y))

        if len(line_pixels) >= min_line_length:
            lines.append([line_pixels[0][0], line_pixels[0][1], line_pixels[-1][0], line_pixels[-1][1]])

    return np.array(lines)

# Detect and Draw Lanes using Hough Transform
def detect_and_draw_lanes(image, edges):
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=250, minLineLength=100, maxLineGap=50)
    # lines = custom_hough_lines_p(edges, 1, np.pi / 180, threshold=170, min_line_length=100, max_line_gap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return image

# Process Images in Directory
def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, file_name)
        image, edges = load_and_preprocess(image_path)
        if image is None:
            continue

        result = detect_and_draw_lanes(image, edges)
        output_path = os.path.join(output_dir, file_name)
        cv.imwrite(output_path, result)
        print(f'{file_name} processed...')

# Function to compute line intersections
def compute_intersections(lines):
    intersections = []
    if lines is None or len(lines) < 2:
        return intersections

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            x3, y3, x4, y4 = lines[j][0]

            # Solve for intersection
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                continue  # Parallel lines
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

            intersections.append((px, py))

    return intersections

# Function to compute centroid
def compute_centroid(points):
    if len(points) == 0:
        return (0, 0)
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    return (sum(x_coords) / len(points), sum(y_coords) / len(points))

# Function to compute sum of distances from centroid
def sum_of_distances(points, centroid):
    return sum(np.sqrt((p[0] - centroid[0]) ** 2 + (p[1] - centroid[1]) ** 2) for p in points)

# Function to analyze lines in an image
def analyze_lines(image_path):
    image, edges = load_and_preprocess(image_path)
    if image is None:
        return None

    # Detect lines using Hough Transform
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=200, minLineLength=100, maxLineGap=50)
    if lines is None:
        return None
    
    # Compute intersections and centroid
    intersections = compute_intersections(lines)
    centroid = compute_centroid(intersections)
    distance_sum = sum_of_distances(intersections, centroid)
    return os.path.basename(image_path), distance_sum

def fit_analysis_task(input_dir, output_csv):
    results = []

    for file_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, file_name)
        result = analyze_lines(image_path)
        if result:
            results.append(result)

    # Save results to CSV
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image_Name", "Line_Fit_Score"])
        writer.writerows(results)

    print(f"Results saved to {output_csv}")

# Main Execution
def main():
    if len(sys.argv) != 4:
        print("Usage: python3 main.py 1 <input_img_dir> <output_img_dir>")
        sys.exit(1)
        
    if sys.argv[1] == '1':
        input_dir = sys.argv[2]
        output_dir = sys.argv[3]
        process_images(input_dir, output_dir)
    
    if sys.argv[1] == '2':
        input_dir = sys.argv[2]
        output_csv = sys.argv[3]
        fit_analysis_task(input_dir, output_csv)

if __name__ == '__main__':
    main()
