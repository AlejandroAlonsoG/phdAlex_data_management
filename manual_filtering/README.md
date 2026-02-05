# Manual Image Filtering Application

A desktop application for visually filtering and organizing images from directories. Browse subdirectories, view images in a paginated mosaic grid, and batch delete unwanted images to the recycle bin.

## Features

- **Directory Browser**: Select a root directory and see all first-level subdirectories
- **Image Count**: See how many images are in each subdirectory (including nested folders)
- **Mosaic View**: View all images in a paginated grid layout (200 images per page)
- **Multi-Select**: Click images to select/deselect them for batch operations
- **Safe Delete**: Selected images are moved to the Recycle Bin (can be restored)
- **Wide Format Support**: Supports 15+ image formats including RAW files

## Supported Image Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| JPEG | `.jpg`, `.jpeg` | Full support |
| PNG | `.png` | Full support |
| GIF | `.gif` | Full support |
| BMP | `.bmp` | Full support |
| WebP | `.webp` | Full support |
| TIFF | `.tif`, `.tiff` | Full support |
| Photoshop | `.psd` | Thumbnail preview |
| Olympus RAW | `.orf` | Thumbnail/placeholder |
| Nikon RAW | `.nef` | Thumbnail/placeholder |
| Sony RAW | `.arw` | Thumbnail/placeholder |
| Illustrator | `.ai` | Placeholder |
| EPS | `.eps` | Placeholder |

## Installation

1. Make sure you have Python 3.8+ installed

2. Install dependencies:
   ```bash
   cd manual_filtering
   pip install -r requirements.txt
   ```

3. (Optional) For better RAW file support, install rawpy:
   ```bash
   pip install rawpy
   ```

## Usage

1. Run the application:
   ```bash
   python app.py
   ```

2. Click **"Select Root Directory"** to choose a folder containing subdirectories with images

3. The app will show all first-level subdirectories with image counts

4. Click on a subdirectory to open the mosaic view

5. **In the mosaic view:**
   - Click on images to select/deselect them
   - Use **"Select All (Page)"** to select all images on the current page
   - Use **"Deselect All"** to clear selection
   - Navigate pages with **Previous/Next** buttons
   - Click **"Delete Selected"** to move selected images to Recycle Bin

6. Use **"← Back to Folders"** to return to the directory list

## Keyboard Shortcuts

- **Mouse Click**: Toggle image selection
- **Ctrl + Click**: Toggle image selection (same behavior)
- **Mouse Wheel**: Scroll through images

## Configuration

Default settings (can be modified in the code):
- **Images per page**: 200
- **Thumbnail size**: 120x120 pixels

## Tips

- RAW and PSD files show placeholder thumbnails if full preview isn't available
- Deleted files go to the Recycle Bin and can be restored
- The app caches thumbnails in memory for faster navigation within a session
- Large directories may take a moment to scan initially

## Troubleshooting

### "No module named 'PIL'"
Install Pillow: `pip install Pillow`

### "No module named 'send2trash'"
Install send2trash: `pip install send2trash`

### RAW files show placeholders
Install rawpy for better RAW support: `pip install rawpy`
(Requires libraw on some systems)

### App is slow with many images
- This is normal for first load as thumbnails are generated
- Try reducing images_per_page in the code if needed
- Subsequent page loads will be faster due to caching
