
import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(title, image):
    """Utility function to display an image properly."""
    plt.figure(figsize=(8, 8))
    if len(image.shape) == 2: # Grayscale
        plt.imshow(image, cmap='gray')
    else: # Color
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    plt.title(title)
    plt.axis('off')
    plt.show() # Moved outside the 'else' so it always runs

def interactive_edge_detection(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Image '{image_path}' not found!")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    display_image("Original Grayscale Image", gray_image)

    while True:
        print("\n--- Select an option ---")
        print("1. Sobel Edge Detection")
        print("2. Canny Edge Detection")
        print("3. Laplacian Edge Detection")
        print("4. Gaussian Smoothing")
        print("5. Median Filtering")
        print("6. Exit")

        choice = input("Enter your choice (1-6): ")

        if choice == "1":
            # Sobel calculates the gradient (intensity change) in x and y directions
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            # Convert back to uint8
            abs_sobel_x = cv2.convertScaleAbs(sobel_x)
            abs_sobel_y = cv2.convertScaleAbs(sobel_y)
            combined_sobel = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
            display_image("Sobel Edge Detection", combined_sobel)

        elif choice == "2":
            try:
                lower = int(input("Enter Lower threshold (e.g., 100): "))
                upper = int(input("Enter Upper threshold (e.g., 200): "))
                edges = cv2.Canny(gray_image, lower, upper)
                display_image("Canny Edge Detection", edges)
            except ValueError:
                print("Please enter valid numbers.")

        elif choice == "3":
            # Laplacian is a 2nd order derivative, very sensitive to noise
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            display_image("Laplacian Edge Detection", cv2.convertScaleAbs(laplacian))

        elif choice == "4":
            try:
                k = int(input("Enter kernel size (must be odd, e.g., 5): "))
                if k % 2 == 0: k += 1 # Auto-fix even numbers
                blurred = cv2.GaussianBlur(image, (k, k), 0)
                display_image("Gaussian Smoothed", blurred)
            except ValueError:
                print("Invalid input.")

        elif choice == "5":
            try:
                k = int(input("Enter kernel size (must be odd, e.g., 5): "))
                if k % 2 == 0: k += 1
                median_filtered = cv2.medianBlur(image, k)
                display_image("Median Filtered", median_filtered)
            except ValueError:
                print("Invalid input.")

        elif choice == "6":
            print("Exiting...")
            break
        else:
            print("Invalid choice.")