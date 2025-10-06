#
# Check if Shelly2 IoT device is reporting
#
# Repository: https://github.com/tomtom87/onsite
# License: GNU General Public License
#

import requests
import datetime
import time

# Constants
SHELLY_CHECK_INTERVAL = 5  # Check every 60 seconds

def check_shelly_status():
    """Check if Shelly2 device is responding via HTTP POST request."""
    try:
        payload = {"id": 1, "method": "Shelly.GetStatus"}
        response = requests.post("http://shelly_ip/rpc", json=payload, timeout=5)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False

def main():
    last_shelly_check = 0  # Initialize last check time
    while True:
        current_time = time.time()
        if current_time - last_shelly_check >= SHELLY_CHECK_INTERVAL:
            shelly_on = check_shelly_status()
            last_shelly_check = current_time
            if shelly_on:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                print(f"{timestamp} - Shelly2 device is ON")
        time.sleep(1)  # Prevent CPU overuse

if __name__ == "__main__":
    main()