/* 
 * Node.js Backend for Onsite Safety Detection
 *
 * Usage: pm2 start main.js --name onsite-backend
 *
 * Example:
 *     pm2 start main.js --name onsite-backend
 *
 * Description:
 *     This script serves as the main entry point for the Node.js backend of the onsite safety detection system.
 *     It must be run using PM2, a process manager for Node.js, to ensure reliable operation, automatic restarts, 
 *     and monitoring of the application.
 *
 * Instructions:
 *     - Install PM2 globally using: npm install -g pm2
 *     - Start the application with: pm2 start main.js --name onsite-backend
 *     - Monitor the application with: pm2 logs onsite-backend
 *     - Stop the application with: pm2 stop onsite-backend
 *     - Restart the application with: pm2 restart onsite-backend
 *
 * Repository: https://github.com/tomtom87/onsite
 * License: GNU General Public License
 */

require("dotenv").config();
const express = require("express");
const cors = require("cors");
const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const multer = require("multer");
const { GridFSBucket, ObjectId } = require("mongodb");

const app = express();

// CORS configuration
const corsOptions = {
  origin: true, // Reflects request origin for credentials
  methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"],
  credentials: true,
};

// Apply CORS middleware
app.use(cors(corsOptions));

// Configuration
const MONGO_URI = process.env.MONGO_URI;
const JWT_SECRET = process.env.JWT_SECRET;
const PORT = process.env.PORT || 8081; // Default to 8081

// Validate environment variables
if (!JWT_SECRET || !MONGO_URI) {
  console.error("Error: JWT_SECRET or MONGO_URI is not defined");
  process.exit(1);
}

let db, gfs;

// MongoDB Connection (single connection for Mongoose and GridFS)
mongoose.connect(MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
}).then(() => {
  console.log("Mongoose connected to MongoDB");
  db = mongoose.connection.db;
  gfs = new GridFSBucket(db, { bucketName: "uploads" });
  console.log("GridFS initialized");
}).catch((err) => {
  console.error("MongoDB connection error:", err);
  process.exit(1);
});

// User Schema
const userSchema = new mongoose.Schema({
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
});
const User = mongoose.model("User", userSchema);

// Middleware
app.use(express.json());

// JWT Authentication Middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers["authorization"];
  const token = authHeader && authHeader.split(" ")[1];

  if (!token) {
    return res.status(401).json({ error: "Access token required" });
  }

  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) {
      return res.status(403).json({ error: "Invalid or expired token" });
    }
    req.user = user;
    next();
  });
};

// Multer setup
const storage = multer.memoryStorage();
const upload = multer({
  storage,
  limits: { fileSize: 5 * 1024 * 1024 },
});

// Sign-Up Endpoint
app.post("/api/signup", async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) {
      return res.status(400).json({ error: "Email and password are required" });
    }

    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ error: "Email already exists" });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const user = new User({ email, password: hashedPassword });
    await user.save();

    res.status(201).json({ message: "User created successfully" });
  } catch (error) {
    console.error("Signup error:", error);
    res.status(500).json({ error: "Failed to create user" });
  }
});

// Login Endpoint
app.post("/api/login", async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) {
      return res.status(400).json({ error: "Email and password are required" });
    }

    const user = await User.findOne({ email });
    if (!user) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    const token = jwt.sign({ id: user._id, email: user.email }, JWT_SECRET, {
      expiresIn: "1h",
    });
    res.json({ token });
  } catch (error) {
    console.error("Login error:", error);
    res.status(500).json({ error: "Failed to log in" });
  }
});

// Get Events (Files) Chronologically
app.get("/api/events", authenticateToken, async (req, res) => {
  try {
    // Get page and limit from query parameters, with defaults
    const page = parseInt(req.query.page) || 1; // Default to page 1
    const limit = parseInt(req.query.limit) || 30; // Default to 30 items per page
    const skip = (page - 1) * limit; // Calculate number of documents to skip

    // Fetch total count for pagination metadata
    const totalFiles = await db.collection("uploads.files")
      .countDocuments({ "metadata.uploadedBy": 1 });

    // Fetch paginated files
    const files = await db.collection("uploads.files")
      .find({ "metadata.uploadedBy": 1 }) // Filter by authenticated user
      .sort({ uploadDate: -1 }) // Sort by uploadDate, newest first
      .skip(skip) // Skip documents for pagination
      .limit(limit) // Limit to 30 documents per page
      .toArray();

    if (!files || files.length === 0) {
      return res.status(404).json({ error: "No events found" });
    }

    // Map files to a simplified response format
    const events = files.map(file => ({
      id: file._id.toString(),
      filename: file.filename,
      uploadDate: file.uploadDate,
      size: file.length,
      metadata: file.metadata,
    }));

    // Calculate pagination metadata
    const totalPages = Math.ceil(totalFiles / limit);
    const pagination = {
      currentPage: page,
      totalPages: totalPages,
      totalItems: totalFiles,
      itemsPerPage: limit,
    };

    res.json({ events, pagination });
  } catch (error) {
    console.error("Error fetching events:", error);
    res.status(500).json({ error: "Failed to fetch events" });
  }
});

// Upload Endpoint
app.post("/upload", authenticateToken, upload.single("image"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded" });
  }

  const uploadStream = gfs.openUploadStream(req.file.originalname, {
    metadata: { uploadedBy: req.user.id },
  });

  uploadStream.write(req.file.buffer);
  uploadStream.end();

  uploadStream.on("finish", () => {
    res.status(201).json({
      message: "File uploaded successfully",
      fileId: uploadStream.id,
    });
  });

  uploadStream.on("error", (err) => {
    console.error("Upload error:", err);
    res.status(500).json({ error: "Failed to upload file" });
  });
});

// False positive Endpoint
app.post("/mark-false-positive", authenticateToken, upload.single("image"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded" });
  }

  const ipAddress = req.body.ipAddress;
  if (!ipAddress) {
    return res.status(400).json({ error: "IP address is required" });
  }

  const uploadStream = gfs.openUploadStream(req.file.originalname, {
    metadata: {
      uploadedBy: req.user.id,
      ipAddress: ipAddress,
      falsePositive: true
    },
  });

  uploadStream.write(req.file.buffer);
  uploadStream.end();

  uploadStream.on("finish", () => {
    res.status(201).json({
      message: "False positive image marked and uploaded successfully",
      fileId: uploadStream.id,
    });
  });

  uploadStream.on("error", (err) => {
    console.error("Upload error:", err);
    res.status(500).json({ error: "Failed to upload file" });
  });
});

// Media Endpoint
app.get("/media/:id", async (req, res) => {
  try {
    const fileId = new ObjectId(req.params.id);
    const file = await db.collection("uploads.files").findOne({ _id: fileId });
    if (!file) {
      return res.status(404).json({ error: "Media not found" });
    }

    const contentType = file.contentType || "image/jpeg";
    const isImage = contentType.startsWith("image/");
    const isVideo = contentType.startsWith("video/");

    if (!isImage && !isVideo) {
      return res.status(400).json({ error: "File is neither an image nor a video" });
    }

    res.set("Content-Type", contentType);

    if (isVideo) {
      res.set({
        "Accept-Ranges": "bytes",
        "Content-Length": file.length,
      });

      const range = req.headers.range;
      if (range) {
        const parts = range.replace(/bytes=/, "").split("-");
        const start = parseInt(parts[0], 10);
        const end = parts[1] ? parseInt(parts[1], 10) : file.length - 1;
        const chunkSize = end - start + 1;

        res.set({
          "Content-Range": `bytes ${start}-${end}/${file.length}`,
          "Content-Length": chunkSize,
        });
        res.status(206);

        const downloadStream = gfs.openDownloadStream(fileId, { start, end: end + 1 });
        downloadStream.pipe(res);

        downloadStream.on("error", (err) => {
          console.error("Download stream error:", err);
          res.status(500).json({ error: "Failed to retrieve media" });
        });
      } else {
        const downloadStream = gfs.openDownloadStream(fileId);
        downloadStream.pipe(res);

        downloadStream.on("error", (err) => {
          console.error("Download stream error:", err);
          res.status(500).json({ error: "Failed to retrieve media" });
        });
      }
    } else {
      res.set("Content-Type", contentType);
      const downloadStream = gfs.openDownloadStream(fileId);
      downloadStream.pipe(res);

      downloadStream.on("error", (err) => {
        console.error("Download stream error:", err);
        res.status(500).json({ error: "Failed to retrieve image" });
      });
    }
  } catch (err) {
    console.error("Media retrieval error:", err);
    if (err.name === "BSONError") {
      return res.status(400).json({ error: "Invalid file ID" });
    }
    res.status(500).json({ error: "Server error" });
  }
});

// Error Handling Middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: "Error 500, oops! Something went wrong! Check the logs and Mongo connection..." });
});

// Start Server
app.listen(PORT, '0.0.0.0', () => console.log(`Server running on port ${PORT}`));