# Add this at the top of your data.py file, right after importing PIL Image:

from PIL import Image
# Disable decompression bomb warning for large images
Image.MAX_IMAGE_PIXELS = None  # or set to a large number like 1000000000

# Rest of your imports and code...
