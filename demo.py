import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Button, Label

def process_image(image_path):
    original = cv2.imread(image_path)
    if original is None:
        print("Error: Unable to read the image.")
        return

    original = cv2.resize(original, (256, 256))
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    equalized = cv2.equalizeHist(gray)

    titles = [
        'Original (RGB)',
        'Grayscale',
        'Gaussian Blur',
        'Canny Edge Detection',
        'Thresholding',
        'Histogram Equalized'
    ]

    descriptions = [
        'Used as reference for visual inspection.',
        'Converts image to 1 channel for simplicity.',
        'Smooths noise, helps reveal disease spots.',
        'Highlights leaf edges & disease outlines.',
        'Separates healthy vs infected leaf areas.',
        'Enhances faint disease visibility.'
    ]

    images = [rgb, gray, blurred, edges, thresh, equalized]

    # Set figure size large enough to accommodate all content
    plt.figure("DIP Methods - Plant Disease Detection", figsize=(18, 10))

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        cmap = 'gray' if len(images[i].shape) == 2 else None
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i], fontsize=12)
        plt.xlabel(descriptions[i], fontsize=10)
        plt.xticks([]), plt.yticks([])

    # Add more space between plots
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.tight_layout()
    plt.show()

def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG")]
    )
    
    if file_path:
        print(f"[INFO] Selected file: {file_path}")
        process_image(file_path)
    else:
        print("[INFO] No file selected.")

# GUI setup
root = Tk()
root.title("DIP Plant Disease Visualizer")
root.geometry("320x180")

label = Label(root, text="Upload Plant Leaf Image", font=("Arial", 12))
label.pack(pady=10)

upload_btn = Button(root, text="Upload Image", command=upload_image, width=25)
upload_btn.pack(pady=20)

root.mainloop()
