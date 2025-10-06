#
# RTSP Hi-viz Jacket Detector with Grok computer vision implementation for sanity check, outputs to Telegram chat and log file
#
# Usage: python detect_hiviz.py --rtsp <rtsp_url> --username <username> --password <password> [options]
#
# Example:
#     python detect_hiviz.py --rtsp rtsp://192.168.1.33:554/stream --username admin --password pass33 --verbose
#
# Options:
#     --rtsp         RTSP stream URL (e.g., rtsp://<ip>:<port>/<path>) [required]
#     --username     RTSP stream username [required]
#     --password     RTSP stream password [required]
#     -v, --verbose  Enable verbose YOLO output
#
# Repository: https://github.com/tomtom87/onsite
# License: GNU General Public License
#

import cv2
import numpy as np
from ultralytics import YOLO
import os
import logging
import argparse
import datetime
import time
from twilio.rest import Client
import subprocess
import json
from telegram import Bot
import asyncio
import requests
import tempfile
from grok import check_hiviz_and_truck 

# Parse command-line arguments
parser = argparse.ArgumentParser(description="OnSite - Hi-viz Jacket detection on RTSP stream")
parser.add_argument('--rtsp', required=True, help="RTSP stream URL (e.g., rtsp://<ip>:<port>/<path>)")
parser.add_argument('--username', required=True, help="RTSP stream username")
parser.add_argument('--password', required=True, help="RTSP stream password")
parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose YOLO output")
args = parser.parse_args()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def send_images(image_urls, message_texts, bot_token, chat_id):
    """
    Send images with custom captions to a Telegram chat by downloading images from URLs.
    
    Parameters:
    - image_urls: List of image URLs to send
    - message_texts: List of corresponding captions for each image
    - bot_token: Your Telegram bot token
    - chat_id: The chat ID to send images to
    """
    
    bot = Bot(token=bot_token)
    
    # Ensure the lists have the same length to avoid mismatches
    if len(image_urls) != len(message_texts):
        logger.error("Error: The number of image URLs and message texts must match.")
        return
    
    # Iterate over image URLs and message texts together
    for image_url, text in zip(image_urls, message_texts):
        try:
            # Download image
            response = requests.get(image_url, timeout=10)
            if response.status_code == 200 and 'image' in response.headers.get('content-type', '').lower():
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path = temp_file.name
                    temp_file.write(response.content)
                logger.info(f"Downloaded image to {temp_path}")

                # Send image
                try:
                    await bot.send_photo(chat_id=chat_id, photo=open(temp_path, 'rb'), caption=text)
                    logger.info(f"Sent image to chat_id {chat_id}: {temp_path} with caption: {text}")
                finally:
                    os.remove(temp_path)
                    logger.info(f"Deleted {temp_path}")
            else:
                logger.error(f"Failed to download image from {image_url}: status {response.status_code}, content-type {response.headers.get('content-type', '')}")
        except Exception as e:
            logger.error(f"Error downloading/sending image from {image_url}: {e}", exc_info=True)
        await asyncio.sleep(1)  # Avoid rate limits

# Function to initialize or reinitialize RTSP stream
def initialize_stream(rtsp_url, max_retries=3, retry_delay=5):
    for attempt in range(max_retries):
        logger.info(f"Attempting to connect to RTSP stream (Attempt {attempt + 1}/{max_retries})")
        cap = cv2.VideoCapture(rtsp_url)
        # Optimize for low latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize frame buffering
        if cap.isOpened():
            logger.info("Successfully connected to RTSP stream")
            return cap
        logger.warning(f"Failed to connect to RTSP stream. Retrying in {retry_delay} seconds...")
        cap.release()
        time.sleep(retry_delay)
    logger.error("Could not connect to RTSP stream after maximum retries")
    return None

def is_point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using cv2.pointPolygonTest."""
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# Define hi-viz color range in HSV for #b1b881, #dee2d3, #9b9c90
lower_hiviz = np.array([35, 20, 80])  # Lower bound for hi-viz colors
upper_hiviz = np.array([60, 255, 255])  # Upper bound for hi-viz colors

# Function to check for hi-viz colors in the middle of a bounding box
def has_hiviz_color(image, bbox, debug=False, frame_idx=0):
    try:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, bbox)
        
        # Calculate the middle portion (50% of width and height)
        width = x2 - x1
        height = y2 - y1
        roi_width = int(width * 0.5)  # 50% of the width
        roi_height = int(height * 0.5)  # 50% of the height
        
        # Calculate the coordinates of the middle ROI
        roi_x1 = x1 + int(width * 0.25)  # Start 25% from the left
        roi_y1 = y1 + int(height * 0.25)  # Start 25% from the top
        roi_x2 = roi_x1 + roi_width
        roi_y2 = roi_y1 + roi_height
        
        # Ensure ROI is within image bounds
        roi_x1 = max(0, roi_x1)
        roi_y1 = max(0, roi_y1)
        roi_x2 = min(image.shape[1], roi_x2)
        roi_y2 = min(image.shape[0], roi_y2)
        
        # Crop the middle ROI
        roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Check if ROI is valid (non-empty)
        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            logger.warning(f"Invalid ROI at frame {frame_idx}: bbox=({x1},{y1},{x2},{y2})")
            return False
        
        # Convert ROI to HSV color space
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Create a mask for hi-viz colors
        mask = cv2.inRange(hsv_roi, lower_hiviz, upper_hiviz)
        
        # Calculate the percentage of hi-viz pixels
        hiviz_pixels = cv2.countNonZero(mask)
        total_pixels = roi.shape[0] * roi.shape[1]
        hiviz_ratio = hiviz_pixels / total_pixels if total_pixels > 0 else 0
        
        # Debugging: Print ratio and save ROI/mask
        if debug:
            logger.info(f"Frame {frame_idx}: Hi-viz ratio = {hiviz_ratio:.4f}, bbox=({x1},{y1},{x2},{y2})")
            cv2.imwrite(f"debug_roi_{frame_idx}_{x1}_{y1}.png", roi)
            cv2.imwrite(f"debug_mask_{frame_idx}_{x1}_{y1}.png", mask)
            cv2.rectangle(image, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 1)
        
        # Return True if significant hi-viz color is detected
        return hiviz_ratio > 0.05  # Threshold: 5% of pixels are hi-viz
    except Exception as e:
        logger.error(f"Error in has_hiviz_color at frame {frame_idx}: {e}")
        return False

async def process_video(): 

    XAI_API_KEY = "xai-333333"  # Replace with your actual API key   
    BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_GOES_HERE"
    CHAT_ID = "-123456789" #Use @myidbot /getgroupdid to retrieve

    # Twilio account credentials (for whatsapp)
    account_sid = 'YOUR_TWILIO_ACCOUNT_ID'  # Replace with your Account SID
    auth_token = 'YOUR_TWILIO_AUTH_TOKEN'    # Replace with your Auth Token
    client = Client(account_sid, auth_token)

    # Configure ultralytics logging based on verbose flag
    logging.getLogger("ultralytics").setLevel(logging.INFO if args.verbose else logging.WARNING)

    # Construct RTSP URL with credentials
    rtsp_url = f"rtsp://{args.username}:{args.password}@{args.rtsp.lstrip('rtsp://')}"

    # Load the YOLO11 model
    model = YOLO("yolo11n.pt")

    # Initialize variable to store the last trigger time
    last_trigger_time = 0

    # Initialize RTSP stream
    cap = initialize_stream(rtsp_url)
    if cap is None:
        logger.error("Exiting due to failure to connect to RTSP stream")
        exit(1)

    # Create directory for saving images
    image_save_dir = "loading_hiviz_images"
    os.makedirs(image_save_dir, exist_ok=True)

    # Get video properties (with fallback for invalid values)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default to 30 if FPS is not available
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280  # Default to 1280
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720  # Default to 720
    logger.info(f"Stream properties: FPS={fps}, Width={frame_width}, Height={frame_height}")

    # Configure multiple camera streams POI (Point Of Interest) Polygons here
    if args.rtsp.endswith('.33'):
        polygon = np.array([
            [1983, 600],   # Top-left
            [2482, 725],  # Top-right
            [1733, 1423],   # Bottom-right
            [967, 1072]    # Bottom-left
        ], dtype=np.int32)
    elif args.rtsp.endswith('.27'):
        polygon = np.array([
            [526, 564],   # Top-left
            [1353, 560],  # Top-right
            [1648, 734],  # Bottom-right
            [633, 785]    # Bottom-left
        ], dtype=np.int32)

    # Initialize frame counter and tracking dictionary
    frame_idx = 0
    track_history = {}  # Dictionary to store {track_id: {'hiviz': bool, 'frames_left': int}}

    # Main loop
    max_frame_drops = 10  # Maximum consecutive frame read failures before reconnect
    frame_drop_count = 0
    reconnect_delay = 5  # Seconds to wait before reconnecting


    try:
        while True:
            # Read a frame from the video
            success, frame = cap.read()

            if success and frame is not None:
                frame_drop_count = 0  # Reset drop counter on successful read
                
                # Increment frame index
                frame_idx += 1
                
                # Skip every other frame to reduce processing load
                if frame_idx % 2 == 0:
                    continue

                try:
                    # Run YOLO11 tracking on the frame, persisting tracks between frames
                    results = model.track(frame, persist=True, verbose=args.verbose)

                    # Count people and hi-viz jackets inside the polygon
                    person_count = 0
                    hiviz_count = 0
                    current_ids = set()  # Track IDs in the current frame

                    if results[0].boxes is not None:
                        for box in results[0].boxes:
                            # Check if the detection is a person (class ID 0)
                            if int(box.cls) == 0:  # 0 is the class ID for 'person' in YOLO
                                # Get bounding box coordinates
                                Boxes = box.xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max]
                                # Get track ID (if available)
                                track_id = int(box.id) if box.id is not None else None

                                # Calculate the center of the bounding box
                                center_x = (Boxes[0] + Boxes[2]) / 2
                                center_y = (Boxes[1] + Boxes[3]) / 2
                                center_point = (center_x, center_y)

                                # Check if the center is inside the polygon
                                if is_point_in_polygon(center_point, polygon):
                                    person_count += 1
                                    if track_id is not None:
                                        current_ids.add(track_id)

                                    # Initialize or update hi-viz status for this track ID
                                    if track_id is not None:
                                        if track_id not in track_history:
                                            # New track: Check for hi-viz jacket
                                            is_hiviz = has_hiviz_color(frame, Boxes, debug=False, frame_idx=frame_idx)
                                            track_history[track_id] = {'hiviz': is_hiviz, 'frames_left': 30 if is_hiviz else 0}
                                        else:
                                            # Existing track: Check if hi-viz is detected in this frame
                                            if has_hiviz_color(frame, Boxes, debug=False, frame_idx=frame_idx):
                                                # Update to hi-viz and reset persistence
                                                track_history[track_id]['hiviz'] = True
                                                track_history[track_id]['frames_left'] = 30
                                            else:
                                                # Decrement persistence counter
                                                track_history[track_id]['frames_left'] = max(0, track_history[track_id]['frames_left'] - 1)
                                                # Reset hi-viz status if counter reaches 0
                                                if track_history[track_id]['frames_left'] == 0:
                                                    track_history[track_id]['hiviz'] = False

                                        # Use persisted hi-viz status for visualization
                                        is_hiviz = track_history[track_id]['hiviz']
                                    else:
                                        # No track ID: Fall back to frame-by-frame detection
                                        is_hiviz = has_hiviz_color(frame, Boxes, debug=False, frame_idx=frame_idx)

                                    # Visualize and save based on hi-viz status
                                    if is_hiviz:
                                        hiviz_count += 1
                                        # Highlight person with hi-viz jacket (green rectangle)
                                        cv2.rectangle(frame, 
                                                    (int(Boxes[0]), int(Boxes[1])), 
                                                    (int(Boxes[2]), int(Boxes[3])), 
                                                    (0, 255, 0), 2)  # Green for hi-viz
                                        cv2.putText(frame, 'Hi-Viz Jacket', 
                                                    (int(Boxes[0]), int(Boxes[1]) - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                                    (0, 255, 0), 2)
                                    else:
                                        current_time = time.time()
                                        if current_time - last_trigger_time >= 120:
                                            last_trigger_time = current_time
                                            # Highlight person without hi-viz jacket (red rectangle and label)
                                            cv2.rectangle(frame, 
                                                        (int(Boxes[0]), int(Boxes[1])), 
                                                        (int(Boxes[2]), int(Boxes[3])), 
                                                        (0, 0, 255), 2)  # Red for non-hi-viz
                                            cv2.putText(frame, 'No Hi-Viz', 
                                                        (int(Boxes[0]), int(Boxes[1]) - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                                        (0, 0, 255), 2)
                                            
                                            # Check with Grok
                                            valid_frame = 0
                                            try:
                                                result = await check_hiviz_and_truck(frame, XAI_API_KEY)
                                                if result in ["true", "false"]:
                                                    logger.error(f"Grok said Result: {result}")
                                                    # Add your logic here based on result - currently checks for a truck being in frame also
                                                    if result == "true":
                                                        valid_frame = 1
                                                        logger.error("Both conditions met: no hi-viz jacket in red box and truck in frame")
                                                    else:
                                                        logger.error("Conditions not met")
                                                else:
                                                    # Handle API warnings/errors returned as strings
                                                    logger.error(f"API Error/Warning: {result}")
                                            except Exception as inner_e:
                                                logger.error(f"Grok Error: {str(inner_e)}")
                                                # Handle specific errors from the function call if needed
                                                raise  # Re-raise or handle as needed

                                            if valid_frame == 1:
                                                # Save the entire frame
                                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                                save_path = os.path.join(image_save_dir, f"no_hiviz_{timestamp}.jpg")
                                                cv2.imwrite(save_path, frame)

                                                # Upload image using cURL
                                                curl_command = (
                                                    f'curl -X POST http://localhost:3000/upload '
                                                    f'-H "Authorization: Bearer 1233333333.'
                                                    f'12313123312312312312.'
                                                    f'12312312331-33333" '
                                                    f'-F "image=@{save_path}"'
                                                )

                                                # Run the cURL command and capture the output
                                                result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)

                                                # Parse the JSON response
                                                response = json.loads(result.stdout)

                                                file_id = response['fileId']
                                                base_url = 'http://localhost:3000/media/'
                                                full_media_url = f'{base_url}{file_id}'  

                                                # Send whatsapp message
                                                """message = client.messages.create(
                                                    body=f"No Hiviz Detected",
                                                    from_='whatsapp:+123456789',  # Replace with your Twilio WhatsApp-enabled number
                                                    media_url=[full_media_url],
                                                    to='whatsapp:+123456789'     # Replace with the recipient's WhatsApp number
                                                )"""
                                                image = [full_media_url]
                                                caption = [f'No Hi-viz Jacket Detected (IP: {args.rtsp})']
                                                await send_images(image, caption, BOT_TOKEN, CHAT_ID)
                                                
                                                logger.info(f"Saved a valid no hi-viz frame: {save_path}")

                    # Clean up track_history: Remove IDs not in the current frame
                    track_history = {tid: info for tid, info in track_history.items() if tid in current_ids}

                    # Draw the polygon for visualization
                    cv2.polylines(frame, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)

                    # Display the person and hi-viz jacket counts on the frame
                    cv2.putText(frame, f"People in ROI: {person_count}", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(frame, f"Hi-Viz Jackets: {hiviz_count}", (50, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    # Display the frame (optional, comment out if running headless)
                    #cv2.imshow("HiViz Loading Area", frame)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User terminated the script")
                        break

                except Exception as e:
                    logger.error(f"Error processing frame {frame_idx}: {e}")
                    continue

            else:
                # Handle frame read failure
                frame_drop_count += 1
                logger.warning(f"Failed to read frame {frame_idx}. Drop count: {frame_drop_count}")
                if frame_drop_count >= max_frame_drops:
                    logger.error(f"Too many consecutive frame drops ({frame_drop_count}). Attempting to reconnect...")
                    cap.release()
                    cap = initialize_stream(rtsp_url)
                    if cap is None:
                        logger.error("Reconnection failed. Exiting...")
                        break
                    frame_drop_count = 0
                time.sleep(0.1)  # Brief pause to avoid tight loop

    except KeyboardInterrupt:
        logger.info("Script interrupted by user")

    finally:
        # Release resources
        logger.info("Releasing resources")
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(process_video())