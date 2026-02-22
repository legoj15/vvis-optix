import sys
import os

try:
    from PIL import Image
except ImportError:
    print("Error: PIL/Pillow not installed. Please run 'pip install pillow'")
    sys.exit(1)

def convert_tga_to_png(file_path):
    if not os.path.exists(file_path):
        return
    
    try:
        # Open the TGA file
        with Image.open(file_path) as img:
            # Construct png path
            base, ext = os.path.splitext(file_path)
            png_path = f"{base}.png"
            
            # Save as PNG
            img.save(png_path)
            print(f"Converted: {file_path} -> {png_path}")

            # For diff images with alpha, extract alpha channel visualizations
            if "diff" in os.path.basename(file_path).lower() and img.mode == "RGBA":
                alpha = img.split()[3]

                # Save raw alpha channel
                alpha_path = f"{base}-alpha.png"
                alpha.save(alpha_path)
                print(f"  Alpha:    {file_path} -> {alpha_path}")

                # Save contrast-enhanced alpha (20x amplification)
                enhanced = alpha.point(lambda p: min(p * 20, 255))
                contrast_path = f"{base}-alphacontrast.png"
                enhanced.save(contrast_path)
                print(f"  Contrast: {file_path} -> {contrast_path}")
            
    except Exception as e:
        print(f"Failed to convert {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tga2png.py <file1.tga> [file2.tga ...]")
        sys.exit(0)
        
    for arg in sys.argv[1:]:
        convert_tga_to_png(arg)
