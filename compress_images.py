import os
from PIL import Image

input_dir = 'docs/assets'
output_dir = 'docs/assets_compressed'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith('.png') or file.lower().endswith('.jpg'):
            input_path = os.path.join(root, file)
            # Create subdirectories in output folder
            rel_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            
            out_dir = os.path.dirname(output_path)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                
            try:
                img = Image.open(input_path)
                # Convert to RGB if it has alpha channel to save as JPEG or just optimize PNG
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Resize if width > 1000
                max_width = 800
                if img.width > max_width:
                    ratio = max_width / float(img.width)
                    new_height = int(float(img.height) * float(ratio))
                    img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
                
                # Save optimized
                img.save(output_path, optimize=True, quality=85)
                print(f"Compressed: {rel_path}")
            except Exception as e:
                print(f"Error compressing {file}: {e}")
