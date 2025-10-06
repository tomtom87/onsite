#
# RTSP Camera Checker - Checks an array of IPs to get the information on their feeds
# Repository: https://github.com/tomtom87/onsite
# License: GNU General Public License
#
import cv2
import numpy as np

# Configuration
USERNAME = "rtsp_username"
PASSWORD = "rtsp_password"

# Array of cameras with IPs and labels
CAMERAS = [
    {"ip": "192.168.1.22", "label": "Forklift Camera 1"},
    {"ip": "192.168.0.33", "label": "Master"},
    {"ip": "192.168.1.27", "label": "Melech HaOlam"},
    {"ip": "192.168.1.55", "label": "Hilarious"},
    {"ip": "192.168.1.111", "label": "Ein Od Milvado"},
    {"ip": "192.168.20.11", "label": "Ketoret"},
    {"ip": "192.168.20.1", "label": "Light"},
    {"ip": "192.168.33.1", "label": "Completion"},
    {"ip": "192.168.33.2", "label": "Quest"}
]

def get_resolution_label(width, height):
    """Determine resolution label (HD, 2K, 4K, 8K) based on dimensions."""
    resolution = width * height
    if resolution >= 7680 * 4320:
        return "8K"
    elif resolution >= 3840 * 2160:
        return "4K"
    elif resolution >= 2560 * 1440:
        return "2K"
    elif resolution >= 1280 * 720:
        return "HD"
    return ""

def connect_to_rtsp(camera):
    """Connect to RTSP stream and retrieve video properties."""
    ip = camera["ip"]
    label = camera["label"]
    rtsp_url = f"rtsp://{USERNAME}:{PASSWORD}@{ip}:554/live"
    
    #print(f"Attempting to connect to RTSP stream: {rtsp_url}")
    
    # Initialize video capture
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print(f"Failed to connect to {rtsp_url}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution_label = get_resolution_label(width, height)
    
    # Output properties
    print(f"{ip} ({label}) - FPS={fps:.1f}, {width}x{height} {resolution_label} ")
    
    # Release the capture
    cap.release()

def main():
    """Process all cameras in the array."""
    for camera in CAMERAS:
        connect_to_rtsp(camera)
        print()  # Add blank line between cameras

if __name__ == "__main__":
    main()