#
# RTSP Camera Capture - Grabs a frame from an RTSP stream and saves to the directory the script is run in, as ip_camera_image-<ip-address>.jpg
#
# Usage: python capture.py --ip <ip_address> [options]
#
# Example:
#     python capture.py --ip 192.168.1.100 --port 554 --username admin --password pass33
#
# Options:
#     --ip           IP address of the RTSP IP camera [required]
#     --port         RTSP port (default: 554)
#     --username     Username for camera authentication (optional, default: '')
#     --password     Password for camera authentication (optional, default: '')
#     --stream-path  Stream path for RTSP (default: live)
#
# Repository: https://github.com/tomtom87/onsite
# License: GNU General Public License
#
import cv2
import argparse

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="Capture an image from an RTSP IP camera")
parser.add_argument('--ip', required=True, help="IP address of the RTSP IP camera")
parser.add_argument('--port', default='554', help="RTSP port (default: 554)")
parser.add_argument('--username', default='', help="Username for camera authentication (optional)")
parser.add_argument('--password', default='', help="Password for camera authentication (optional)")
parser.add_argument('--stream-path', default='live', help="Stream path for RTSP (default: live)")
args = parser.parse_args()

# Construct the RTSP URL
if args.username and args.password:
    rtsp_url = f"rtsp://{args.username}:{args.password}@{args.ip}:{args.port}/{args.stream_path}"
else:
    rtsp_url = f"rtsp://{args.ip}:{args.port}/{args.stream_path}"

# Initialize the video capture with the RTSP URL
cap = cv2.VideoCapture(rtsp_url)

# Check if the camera stream opened successfully
if not cap.isOpened():
    print(f"Error: Could not open RTSP stream at {rtsp_url}")
    exit()

# Read a frame from the camera
ret, frame = cap.read()

# If frame is read correctly, save it
if ret:
    # Save the captured image with IP address in the filename
    output_filename = f'ip_camera_image-{args.ip}.jpg'
    cv2.imwrite(output_filename, frame)
    print(f"Image captured and saved as '{output_filename}'")
else:
    print("Error: Could not capture image")

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()