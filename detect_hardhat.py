#
# RTSP Hard Hat Detector with Shelly IoT support for perphieral detection (e.g a Crane) and Polygon Interest areas
# - Outputs to Telegram chat and Log files
# - Run within pm2 for daemonization 
#
# Usage: python detect_hardhat.py --rtsp <rtsp_url> --username <username> --password <password> [options]
#
# Example:
#     python detect_hardhat.py --rtsp rtsp://192.168.1.33:554/stream --username admin --password pass33 --verbose --retry-delay 10.0 --max-retries 5
#
# Options:
#     --rtsp         RTSP stream URL (e.g., rtsp://<ip>:<port>/<path>) [required]
#     --username     RTSP stream username [required]
#     --password     RTSP stream password [required]
#     -v, --verbose  Enable verbose YOLO output
#     --retry-delay  Initial delay between reconnection attempts (seconds, default: 5.0)
#     --max-retries  Maximum reconnection attempts before exiting (default: 10)
#
# Repository: https://github.com/tomtom87/onsite
# License: GNU General Public License
#
import json
import cv2
import numpy as np
from ultralytics import YOLO
import datetime
import time
import os
import requests
import argparse
import logging
import asyncio
import tempfile
import subprocess
from telegram import Bot

os.environ["NNPACK"] = "0"  # Disable NNPACK to avoid the warning
from collections import defaultdict

# Parse command-line arguments
parser = argparse.ArgumentParser(description="OnSite - Helmet detection on RTSP stream with CUDA and reconnection")
parser.add_argument('--rtsp', required=True, help="RTSP stream URL (e.g., rtsp://<ip>:<port>/<path>)")
parser.add_argument('--username', required=True, help="RTSP stream username")
parser.add_argument('--password', required=True, help="RTSP stream password")
parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose YOLO output")
parser.add_argument('--retry-delay', type=float, default=5.0, help="Initial delay between reconnection attempts (seconds)")
parser.add_argument('--max-retries', type=int, default=10, help="Maximum reconnection attempts before exiting")
args = parser.parse_args()

# Configure ultralytics logging based on verbose flag
logging.getLogger("ultralytics").setLevel(logging.INFO if args.verbose else logging.WARNING)
logger = logging.getLogger(__name__)

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Construct RTSP URL with credentials
rtsp_url = f"rtsp://{args.username}:{args.password}@{args.rtsp.lstrip('rtsp://')}"

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
    
    if len(image_urls) != len(message_texts):
        logger.error("Error: The number of image URLs and message texts must match.")
        return
    
    for image_url, text in zip(image_urls, message_texts):
        try:
            response = requests.get(image_url, timeout=10)
            if response.status_code == 200 and 'image' in response.headers.get('content-type', '').lower():
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path = temp_file.name
                    temp_file.write(response.content)
                logger.info(f"Downloaded image to {temp_path}")

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

def connect_to_stream(url, max_retries, initial_delay):
    """Attempt to connect to the RTSP stream with retries and exponential backoff."""
    cap = None
    retry_count = 0
    delay = initial_delay

    while retry_count < max_retries:
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            logging.info(f"Successfully connected to RTSP stream: {url}")
            return cap, True
        else:
            logging.warning(f"Failed to connect to RTSP stream (attempt {retry_count + 1}/{max_retries}). Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            retry_count += 1
            delay *= 2  # Exponential backoff

    logging.error(f"Could not connect to RTSP stream after {max_retries} attempts. Exiting.")
    return None, False

def check_shelly_status():
    """Check if Shelly2 device is responding via HTTP POST request."""
    try:
        payload = {"id": 1, "method": "Shelly.GetStatus"}
        response = requests.post("http://shelly_ip/rpc", json=payload, timeout=5)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def assign_id(cx, cy, tracked_objects, next_id):
    """Assign a new or existing ID to a detection based on proximity."""
    min_dist = float('inf')
    matched_id = None
    
    for obj_id, (center, _, _, _) in tracked_objects.items():
        dist = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
        if dist < min_dist and dist < 50:
            min_dist = dist
            matched_id = obj_id
    
    if matched_id is None:
        matched_id = next_id
        next_id += 1
    
    return matched_id, next_id

async def process_frame(cap, frame, model, helmet_polygon, tracked_objects, next_id, shelly_on, last_shelly_check, log_file, image_save_dir, frame_width, frame_height, NO_HELMET_THRESHOLD, LOG_COOLDOWN, SHELLY_CHECK_INTERVAL):
    """Process a single frame asynchronously."""
    current_time = time.time()
    BOT_TOKEN = "TELEGRAM_BOT_TOKEN_GOES_HERE"
    CHAT_ID = "-123456789" #Use @myidbot /getgroupdid to retrieve


    # Check Shelly2 status periodically
    if current_time - last_shelly_check >= SHELLY_CHECK_INTERVAL:
        shelly_on = check_shelly_status()
        last_shelly_check = current_time
        if shelly_on:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            #with open(log_file, "a") as log:
            #    log.write(f"{timestamp} - Shelly2 device is ON\n")

    # Display Shelly status on frame
    cv2.putText(frame, f"Shelly2: {'ON' if shelly_on else 'OFF'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if shelly_on else (0, 0, 255), 2)

    # Draw the helmet polygon
    cv2.polylines(frame, [helmet_polygon], isClosed=True, color=(255, 0, 0), thickness=2)

    # Perform detection only if Shelly2 is on
    current_detections = {}
    if shelly_on:
        results = model(frame)
        for detection in results[0].boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            label = int(detection.cls)

            # Check if person in helmet ROI
            if cv2.pointPolygonTest(helmet_polygon, (cx, cy), False) >= 0 and label == 0:
                # Assign or match ID
                obj_id, next_id = assign_id(cx, cy, tracked_objects, next_id)
                
                # Head region processing
                box_height = y2 - y1
                head_y2 = y1 + int(0.25 * box_height)
                head_roi = frame[y1:head_y2, x1:x2]
                gray_roi = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray_roi, 220, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                helmet_detected = False
                for contour in contours:
                    if cv2.contourArea(contour) > 50:
                        helmet_detected = True
                        cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)
                        cv2.putText(frame, f"ID: {obj_id} Helmet", (x1, y1 - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Update tracking info
                no_helmet_start, last_logged = tracked_objects.get(obj_id, (None, None, None, None))[2:4]
                if not helmet_detected:
                    cv2.putText(frame, f"ID: {obj_id} No Helmet!", (x1, y1 - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if no_helmet_start is None:
                        no_helmet_start = current_time
                else:
                    no_helmet_start = None
                
                current_detections[obj_id] = ((cx, cy), current_time, no_helmet_start, last_logged)
                
                # Draw bounding boxes
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, head_y2), (255, 0, 0), 2)

                # Log and save image if no helmet for 3+ seconds and 1 minute has passed since last log
                if not helmet_detected and no_helmet_start and (current_time - no_helmet_start) >= NO_HELMET_THRESHOLD:
                    if last_logged is None or (current_time - last_logged) >= LOG_COOLDOWN:
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        with open(log_file, "a") as log:
                            log.write(f"{timestamp} - ID: {obj_id} - No Helmet Detected for 3+ seconds\n")
                        
                        # Save cropped image
                        cropped_img = frame[max(0, y1-50):min(frame_height, y2+50), 
                                         max(0, x1-50):min(frame_width, x2+50)]
                        img_path = os.path.join(image_save_dir, f"no_helmet_ID{obj_id}_{timestamp}.jpg")
                        cv2.imwrite(img_path, cropped_img)
                        
                        # Update last logged time
                        current_detections[obj_id] = ((cx, cy), current_time, no_helmet_start, current_time)
                        
                        # Upload image using cURL
                        curl_command = (
                            f'curl -X POST http://localhost:3000/upload '
                            f'-H "Authorization: Bearer 1231312231241324.'
                            f'12312334123412431231.'
                            f'12131312312-333" '
                            f'-F "image=@{img_path}"'
                        )

                        # Run the cURL command and capture the output
                        result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)

                        # Parse the JSON response
                        response = json.loads(result.stdout)

                        file_id = response['fileId']
                        base_url = 'http://localhost:3000/media/'
                        full_media_url = f'{base_url}{file_id}'  
                        
                        image = [full_media_url]
                        caption = [f'No Helmet Detected (IP: {args.rtsp})']
                        await send_images(image, caption, BOT_TOKEN, CHAT_ID)
                        
                        logger.info(f"Saved a valid no helmet frame: {img_path}")        

        tracked_objects.update(current_detections)

    # Display frame
    #cv2.imshow("Detection", frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    return False, tracked_objects, next_id, shelly_on, last_shelly_check

    return True, tracked_objects, next_id, shelly_on, last_shelly_check

async def main():
    # Initialize RTSP stream connection
    cap, success = await asyncio.get_event_loop().run_in_executor(None, lambda: connect_to_stream(rtsp_url, args.max_retries, args.retry_delay))
    if not success:
        logger.error("Failed to initialize RTSP stream. Exiting.")
        return

    try:
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        # Define the polygonal region of interest (ROI) for helmet detection
        helmet_polygon = np.array([
            [861, 461],   # Top-left
            [1156, 465],  # Top-right
            [1300, 615],  # Bottom-right
            [905, 648]    # Bottom-left
        ], dtype=np.int32)

        # Initialize tracking variables
        next_id = 1
        tracked_objects = {}
        NO_HELMET_THRESHOLD = 3.0  # 3 seconds
        LOG_COOLDOWN = 60.0  # 1 minute
        SHELLY_CHECK_INTERVAL = 5.0  # Check Shelly every 10 seconds
        last_shelly_check = 0.0
        shelly_on = False

        # Open the log file
        log_file = "helmet_detection.log"
        image_save_dir = "no_helmet_images"
        os.makedirs(image_save_dir, exist_ok=True)

        # Main loop
        while True:
            # Read frame in executor to avoid blocking
            ret, frame = await asyncio.get_event_loop().run_in_executor(None, lambda: cap.read())
            if not ret:
                logger.warning("Failed to read frame. Attempting to reconnect...")
                cap.release()
                cap, success = await asyncio.get_event_loop().run_in_executor(None, lambda: connect_to_stream(rtsp_url, args.max_retries, args.retry_delay))
                if not success:
                    logger.error("Reconnection failed. Exiting.")
                    break
                continue

            # Process frame
            continue_loop, tracked_objects, next_id, shelly_on, last_shelly_check = await process_frame(
                cap, frame, model, helmet_polygon, tracked_objects, next_id, shelly_on, 
                last_shelly_check, log_file, image_save_dir, frame_width, frame_height,
                NO_HELMET_THRESHOLD, LOG_COOLDOWN, SHELLY_CHECK_INTERVAL
            )

            if not continue_loop:
                break

            # Control frame rate
            await asyncio.sleep(1 / fps)

    finally:
        # Release resources
        await asyncio.get_event_loop().run_in_executor(None, cap.release)
        await asyncio.get_event_loop().run_in_executor(None, cv2.destroyAllWindows)
        logger.info("Script terminated.")

if __name__ == "__main__":
    asyncio.run(main())