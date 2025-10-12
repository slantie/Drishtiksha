// Backend/src/storage/local.provider.js

import { promises as fs, existsSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import ffmpeg from "fluent-ffmpeg";
import sharp from "sharp";
import mime from "mime-types";
import { config } from "../config/env.js";
import logger from "../utils/logger.js";
import { ApiError } from "../utils/ApiError.js";

// --- FFMpeg Path Configuration ---
// If ffmpeg is not in your system's PATH, you need to specify its location.
// Uncomment the line below and set the correct path if needed.
// import ffmpegPath from '@ffmpeg-installer/ffmpeg';
// ffmpeg.setFfmpegPath(ffmpegPath.path);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PROJECT_ROOT = path.resolve(__dirname, "..", "..");
const STORAGE_ROOT = path.join(PROJECT_ROOT, config.LOCAL_STORAGE_PATH);

const ensureDirectoryExists = async (dirPath) => {
  try {
    if (!existsSync(dirPath)) {
      await fs.mkdir(dirPath, { recursive: true });
    }
  } catch (error) {
    throw new ApiError(
      500,
      `Could not create storage directory: ${error.message}`
    );
  }
};

// ====================================================================
//  MEDIA CONVERSION HELPERS
// ====================================================================

/**
 * Extracts comprehensive metadata from media files using ffprobe.
 * @param {string} filePath - The path to the media file.
 * @returns {Promise<Object>} Metadata object with format, video, audio, and streams info.
 */
const extractMediaMetadata = (filePath) => {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(filePath, (error, metadata) => {
      if (error) {
        logger.warn(`Failed to extract metadata from ${filePath}: ${error.message}`);
        resolve(null); // Return null instead of rejecting - metadata is optional
        return;
      }

      try {
        const processed = {
          format: {},
          video: null,
          audio: null,
          streams: [],
        };

        // Format information
        if (metadata.format) {
          processed.format = {
            duration: parseFloat(metadata.format.duration) || 0,
            size: parseInt(metadata.format.size) || 0,
            bitRate: parseInt(metadata.format.bit_rate) || 0,
            formatName: metadata.format.format_name,
          };
        }

        // Stream information
        if (metadata.streams && Array.isArray(metadata.streams)) {
          metadata.streams.forEach((stream) => {
            if (stream.codec_type === "video") {
              processed.video = {
                codecName: stream.codec_name,
                width: stream.width,
                height: stream.height,
                frameRate: stream.r_frame_rate,
                bitRate: parseInt(stream.bit_rate) || 0,
              };
            } else if (stream.codec_type === "audio") {
              processed.audio = {
                codecName: stream.codec_name,
                sampleRate: parseInt(stream.sample_rate) || 0,
                channels: stream.channels,
                bitRate: parseInt(stream.bit_rate) || 0,
              };
            }

            processed.streams.push({
              codecType: stream.codec_type,
              codecName: stream.codec_name,
            });
          });
        }

        resolve(processed);
      } catch (processError) {
        logger.warn(`Failed to process metadata: ${processError.message}`);
        resolve(null);
      }
    });
  });
};

/**
 * Converts any compatible video file to MP4 using ffmpeg.
 * @param {string} inputPath - The path to the source video file.
 * @param {string} outputPath - The path to save the converted MP4 file.
 * @returns {Promise<string>} A promise that resolves with the output path.
 */
const convertToMp4 = (inputPath, outputPath) => {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .outputOptions([
        "-c:v libx264", // H.264 codec for wide compatibility
        "-preset slow", // Good balance of quality and speed
        "-crf 23", // Constant Rate Factor for quality (lower is better)
        "-c:a aac", // AAC audio codec
        "-b:a 128k", // Audio bitrate
        "-movflags +faststart", // Optimizes for web streaming
      ])
      .toFormat("mp4")
      .on("end", () => {
        logger.info(`Video converted successfully to: ${outputPath}`);
        resolve(outputPath);
      })
      .on("error", (err) => {
        logger.error(`Error converting video: ${err.message}`);
        reject(new ApiError(500, `Video conversion failed: ${err.message}`));
      })
      .save(outputPath);
  });
};

/**
 * Converts any compatible audio file to MP3 using ffmpeg.
 * @param {string} inputPath - The path to the source audio file.
 * @param {string} outputPath - The path to save the converted MP3 file.
 * @returns {Promise<string>} A promise that resolves with the output path.
 */
const convertToMp3 = (inputPath, outputPath) => {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .audioCodec("libmp3lame")
      .audioBitrate("192")
      .toFormat("mp3")
      .on("end", () => {
        logger.info(`Audio converted successfully to: ${outputPath}`);
        resolve(outputPath);
      })
      .on("error", (err) => {
        logger.error(`Error converting audio: ${err.message}`);
        reject(new ApiError(500, `Audio conversion failed: ${err.message}`));
      })
      .save(outputPath);
  });
};

/**
 * Converts any compatible image file to PNG using sharp.
 * @param {string} inputPath - The path to the source image file.
 * @param {string} outputPath - The path to save the converted PNG file.
 * @returns {Promise<string>} A promise that resolves with the output path.
 */
const convertToPng = async (inputPath, outputPath) => {
  try {
    await sharp(inputPath).toFormat("png").toFile(outputPath);
    logger.info(`Image converted successfully to: ${outputPath}`);
    return outputPath;
  } catch (error) {
    logger.error(`Error converting image: ${error.message}`);
    throw new ApiError(500, `Image conversion failed: ${error.message}`);
  }
};

// ====================================================================
//  REFACTORED LOCAL PROVIDER
// ====================================================================

const localProvider = {
  /**
   * Detects file type, converts to a standard format, and saves it.
   * @param {string} localFilePath - Path to the temporarily uploaded file.
   * @param {string} originalFilename - The original name of the file to detect its type.
   * @param {string} subfolder - The subfolder to store the file in.
   * @returns {Promise<{url: string, publicId: string}>}
   */
  async uploadFile(localFilePath, originalFilename, subfolder = "uploads") {
    const permanentStorageDir = path.join(STORAGE_ROOT, subfolder);
    await ensureDirectoryExists(permanentStorageDir);

    const mimeType =
      mime.lookup(originalFilename) || "application/octet-stream";
    let fileType = mimeType.split("/")[0];

    // special case: sometimes .mp4 is labeled as application/mp4
    if (mimeType === "application/mp4") {
      fileType = "video";
    }

    const timestamp = Date.now();
    const baseFilename = `${timestamp}-${path.parse(originalFilename).name}`;

    let conversionFn;
    let targetExtension;

    // Determine the target format and conversion function based on MIME type
    switch (fileType) {
      case "video":
        targetExtension = ".mp4";
        conversionFn = convertToMp4;
        break;
      case "audio":
        targetExtension = ".mp3";
        conversionFn = convertToMp3;
        break;
      case "image":
        targetExtension = ".png";
        conversionFn = convertToPng;
        break;
      default:
        // If the file type is not supported for conversion, we can choose to reject it.
        await fs.unlink(localFilePath); // Clean up the original temp file
        throw new ApiError(415, `Unsupported file type: ${mimeType}`);
    }

    const uniqueFilename = `${baseFilename}${targetExtension}`;
    const destinationPath = path.join(permanentStorageDir, uniqueFilename);

    try {
      // Perform the conversion and save directly to the destination
      await conversionFn(localFilePath, destinationPath);
    } catch (error) {
      // If conversion fails, re-throw the specific ApiError from the helper
      throw error;
    } finally {
      // IMPORTANT: Clean up the original temporary file after processing
      if (existsSync(localFilePath)) {
        await fs.unlink(localFilePath);
      }
    }

    // ====================================================================
    //  CRITICAL FIX: Get stats for the newly converted file
    // ====================================================================
    const stats = await fs.stat(destinationPath);
    const newSize = stats.size;
    const newMimeType =
      mime.lookup(destinationPath) || "application/octet-stream";

    // Extract metadata for videos and audio files
    let metadata = null;
    if (fileType === "video" || fileType === "audio") {
      metadata = await extractMediaMetadata(destinationPath);
    }

    const publicId = path.join(subfolder, uniqueFilename).replace(/\\/g, "/");
    const urlPathSegment = config.LOCAL_STORAGE_PATH.split("/")
      .slice(1)
      .join("/");
    const publicUrl = new URL(
      `${urlPathSegment}/${publicId}`,
      config.ASSETS_BASE_URL
    ).href;

    return {
      url: publicUrl,
      publicId: publicId,
      mimetype: newMimeType, // This was the missing piece of data
      size: newSize, // This was the other missing piece
      metadata: metadata, // Rich metadata from ffprobe
    };
  },

  async deleteFile(publicId) {
    if (!publicId) return;
    const fullPath = path.join(STORAGE_ROOT, publicId);
    try {
      if (existsSync(fullPath)) {
        await fs.unlink(fullPath);
        logger.info(`Successfully deleted local file: ${fullPath}`);
      }
    } catch (error) {
      logger.error(`Failed to delete local file ${fullPath}: ${error.message}`);
    }
  },
};

export default localProvider;
