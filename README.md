# OnSite - Video Analytics

Detect Hard Hat and Hi-Viz jackets over RTSP streams with Python and deliver alerts via Telegram, Whatsapp and to log files. Includes a Node.js backend using Mongo and GridFS for self-hosting the images used in alerts. Supports IoT checks via Shelly2 for checking if a circuit is live during detection and sanity of results via AI with Grok by xAi. 

## Features
- **detect_ py scripts**: Detects an event within a polygon, defined within the file. Called with command line arguments. Supports secure RTSP streams.
- **backend/**: RESTful API for saving images that have been detected from the python scripts.

- Easy setup with Python virtual environments and Node.js dependencies. 
- Manage multiple streams with PM2 
- Built-in Support for Nvidia CUDA Cores, approx 20~30 concurrent streams with a RTX 5090.

## Prerequisites
Ensure you have the following installed:
- Python 3.x ([Download](https://www.python.org/downloads/))
- Node.js (version 14 or higher recommended) ([Download](https://nodejs.org/))
- npm (included with Node.js)
- Git ([Download](https://git-scm.com/))

## Installation

1. **Clone the repo**:
   ```bash
   git clone https://github.com/tomtom87/onsite
   cd onsite
   ```
2. **Setup your virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Setup your backend**:
   ```bash
   cd backend
   npm install
   npm install -g pm2
   pm2 start main.js --name onsite-backend
   ```

## Usage

**Run a detector via PM2**:
```bash
pm2 start detect_hardhat.py --interpreter=python3 -- --rtsp rtsp://192.168.1.33:554/stream --username admin --password pass33 --verbose --retry-delay 10.0 --max-retries 5
```
**View Logs**:
```bash
pm2 logs
```
**View CUDA Usage**:
```bash
nvidia-smi
```
## Tools

**check_cameras.py**: Checks the status of multiple camera feeds, first setup the file and then run it via `python check_cameras.py` inside your venv

**capture.py**: Save a single frame from your RTSP stream for a quick check - `python capture.py --ip 192.168.1.100 --port 554 --username admin --password pass33`

**check_shelly.py**: Checks the connectivity and status of a Shelly2 device's API - `python check_shelly.py`

## Contributing

Contributions are welcome! 
To contribute:
1. Fork the repository.
2. Create a new branch (git checkout -b feature/your-feature).
3. Make your changes and commit (git commit -m "Add your feature").
4. Push to the branch (git push origin feature/your-feature).
5. Open a pull request.

## Thanks 
Thank you for using OnSite (GNUGPL General Public License.)


Thanks to Shai Snir, Ofer Taib, Rafat, Avidan Tal, Tomer Vaknin + Everyone at VGold for the help during development and Oded Daniel for the encouragement! 