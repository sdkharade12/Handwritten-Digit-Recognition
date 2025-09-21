import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os

# Load the model
import tensorflow as tf
model = tf.keras.models.load_model('backend/model.h5')

def preprocess_digit_image(image_path: str):
    """Enhanced preprocessing with debugging info"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Not found: {image_path}")

    # Load and convert to grayscale
    img = Image.open(image_path).convert('L')
    print(f"Original image size: {img.size}")
    print(f"Original image mean brightness: {np.mean(np.array(img)):.2f}")

    # Auto-invert if background appears white
    np_img = np.array(img)
    if np.mean(np_img) > 127:
        img = ImageOps.invert(img)
        print("Image inverted (background was white)")

    # Improve contrast
    img = ImageOps.autocontrast(img)
    print(f"After autocontrast mean brightness: {np.mean(np.array(img)):.2f}")

    # Fit image into a 28x28 canvas, keeping aspect ratio
    img.thumbnail((28, 28), Image.LANCZOS)
    print(f"After thumbnail size: {img.size}")
    
    canvas = Image.new('L', (28, 28), color=0)
    paste_x = (28 - img.width) // 2
    paste_y = (28 - img.height) // 2
    canvas.paste(img, (paste_x, paste_y))
    print(f"Pasted at position: ({paste_x}, {paste_y})")

    # Normalize and reshape to model input
    arr = np.array(canvas).astype('float32') / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return canvas, arr

def analyze_prediction(image_path):
    """Analyze why the model makes a specific prediction"""
    processed_pil, input_tensor = preprocess_digit_image(image_path)
    
    # Get prediction probabilities
    probs = model.predict(input_tensor, verbose=0)[0]
    predicted_digit = int(np.argmax(probs))
    
    print(f"\nPrediction Analysis:")
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {probs[predicted_digit]:.4f}")
    
    print(f"\nAll probabilities:")
    for i, prob in enumerate(probs):
        print(f"Digit {i}: {prob:.4f} ({prob*100:.1f}%)")
    
    # Show the processed image
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(processed_pil, cmap='gray')
    plt.title(f'Processed Image\nPredicted: {predicted_digit}')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.bar(range(10), probs)
    plt.title('Prediction Probabilities')
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.xticks(range(10))
    
    plt.subplot(1, 3, 3)
    # Show the raw pixel values
    plt.imshow(input_tensor[0, :, :, 0], cmap='gray')
    plt.title('Raw Input to Model')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return processed_pil, input_tensor, probs

if __name__ == "__main__":
    # Test with the problematic image
    image_path = "4.png"  # The image that's being misclassified
    analyze_prediction(image_path)
