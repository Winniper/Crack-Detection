import cv2
import numpy as np

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_16bit = cv2.convertScaleAbs(gray, alpha=(65535/255))

    denoised = cv2.fastNlMeansDenoising(gray_16bit, h=20, templateWindowSize=7, searchWindowSize=21)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    return enhanced

def calculate_dimensions(contours, pixel_size=1.12e-6, L=0.13, f=0.003):
    ζ = L / f  # Scale factor
    total_length = 0
    widths = []
    
    for cnt in contours:
        # Calculate length
        total_length += cv2.arcLength(cnt, False) * pixel_size * ζ
        
        # Calculate width
        rect = cv2.minAreaRect(cnt)
        width_px = min(rect[1])
        widths.append(width_px * pixel_size * (ζ - 1))
    
    return {
        'total_length_mm': total_length * 1000,
        'max_width_mm': np.max(widths) * 1000,
        'mean_width_mm': np.mean(widths) * 1000
    }

def estimate_depth(original_img, mask, L):
    surface_intensity = cv2.mean(original_img, mask=255-mask)[0]
    crack_intensity = cv2.mean(original_img, mask=mask)[0]
    ξ = crack_intensity / surface_intensity
    return L * ((1 / (ξ ** 0.25)) - 1) * 1000  # Convert to mm

def analyze_crack(image_path):
    # Load and process image
    img = cv2.imread(image_path)
    processed = preprocess_image(img)
    
    # Find contours
    thresh = cv2.adaptiveThreshold(processed, 65535, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 15, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask
    mask = np.zeros_like(processed)
    cv2.drawContours(mask, contours, -1, 255, -1)
    
    # Calculate dimensions
    dimensions = calculate_dimensions(contours)
    depth_mm = estimate_depth(processed, mask, 0.13)  # L=0.13m
    
    # Print results
    print(f"Crack Length: {dimensions['total_length_mm']:.2f} mm")
    print(f"Maximum Width: {dimensions['max_width_mm']:.2f} mm")
    print(f"Average Width: {dimensions['mean_width_mm']:.2f} mm")
    print(f"Estimated Depth: {depth_mm:.2f} mm")
    
    return dimensions, depth_mm

if __name__ == "__main__":
    image_path = r"E:\Capstone\Crack-Detection\CV\sample.jpg"
    analyze_crack(image_path)
