#
# Helper functions for use with Grok go here, you will need your own xAI key.
#
# Repository: https://github.com/tomtom87/onsite
# License: GNU General Public License
#

import base64
import cv2
import numpy as np
from openai import AsyncOpenAI
import re

async def check_hiviz_and_truck(frame: np.ndarray, api_key: str) -> str:
    """
    Asynchronously check if a person not in a hi-viz jacket is in the red box and a truck is in the frame.
    
    Args:
        frame (np.ndarray): OpenCV image frame with red box drawn.
        api_key (str): xAI API key.
    
    Returns:
        str: 'true' if both conditions are met, 'false' otherwise, or error message if API call fails.
    """
    try:
        # Initialize the xAI async client
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )

        # Encode the OpenCV frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        image_data = base64.b64encode(buffer).decode('utf-8')

        # Prompt
        prompt = (
            "Based on the image provided:\n\n"
            "1. Check if there is a person inside the red box who is not wearing a high-visibility jacket.\n"
            "2. Check if there is a truck visible in the entire frame of the image.\n\n"
            "Output the result in the following format:\n"
            "Based on the image provided:\n\n"
            "1. There is [a/no] person inside the red box who is not wearing a high-visibility jacket.\n"
            "2. There is [a/no] truck visible in the entire frame of the image.\n\n"
            "Since both conditions are [met/not met], the output is:\n"
            "Therefore, the output is `true` or `false`"
        )

        # Send the request to the xAI Grok API
        vision_response = await client.chat.completions.create(
            model="grok-2-vision-latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            temperature=0.01
        )

        # Extract and validate the response
        result = vision_response.choices[0].message.content.strip()
        # Split the response into lines and get the last line
        last_line = result.split('\n')[-1].strip()
        # Extract the last word from the last line, removing backticks and any surrounding text
        last_word = last_line.split('`')[-1].strip().lower() if '`' in last_line else last_line.split()[-1].lower()
        
        # Regular expression to match "Therefore, the output is `<value>`."
        pattern = r"Therefore, the output is `(true|false)`\."

        # Search for the pattern in the input string
        match = re.search(pattern, result)

        # Extract the result if found
        if match:
            last_word = match.group(1)  # Captures 'true' or 'false'
        else:
            last_word = "false"

        if last_word not in ["true", "false"]:
            return f"Warning: Last word of API response is not 'true' or 'false'. Response: {result}"
        return last_word

    except Exception as e:
        return f"Error: {str(e)}"