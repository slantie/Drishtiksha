// src/services/mediaConverter.service.js

import ffmpeg from "fluent-ffmpeg";
import ffmpegPath from "ffmpeg-static";
import ffprobePath from "ffprobe-static";
import path from "path";
import fs from "fs/promises";
import { ApiError } from "../utils/ApiError.js";
import logger from "../utils/logger.js";

// Configure FFmpeg paths
if (ffmpegPath) ffmpeg.setFfmpegPath(ffmpegPath);
if (ffprobePath?.path) ffmpeg.setFfprobePath(ffprobePath.path);

/**
 * Comprehensive Media Converter Service
 * Standardizes video files to MP4 and audio files to MP3
 * Preserves metadata and ensures consistent playback across browsers
 */
class MediaConverterService {
  constructor() {
    this.TEMP_CONVERSION_DIR = path.resolve("temp/conversions");
    this.initializeTempDirectory();

    // Supported input formats
    this.VIDEO_FORMATS = new Set([
      "video/mp4",
      "video/quicktime",
      "video/x-msvideo",
      "video/webm",
      "video/mpeg",
      "video/3gpp",
      "video/x-ms-wmv",
      "video/x-flv",
    ]);

    this.AUDIO_FORMATS = new Set([
      "audio/mpeg",
      "audio/wav",
      "audio/ogg",
      "audio/aac",
      "audio/flac",
      "audio/x-ms-wma",
      "audio/x-m4a",
      "audio/mp4",
    ]);

    // Target formats
    this.TARGET_VIDEO_FORMAT = "mp4";
    this.TARGET_AUDIO_FORMAT = "mp3";

    // Conversion settings
    this.VIDEO_SETTINGS = {
      codec: "libx264",
      audioCodec: "aac",
      preset: "medium",
      crf: 23, // Good quality with reasonable file size
      maxBitrate: "2M",
      audioBitrate: "128k",
    };

    this.AUDIO_SETTINGS = {
      codec: "libmp3lame",
      bitrate: "192k",
      sampleRate: 44100,
      channels: 2,
    };
  }

  async initializeTempDirectory() {
    try {
      await fs.mkdir(this.TEMP_CONVERSION_DIR, { recursive: true });
      logger.info(
        `[MediaConverter] Temporary conversion directory initialized: ${this.TEMP_CONVERSION_DIR}`
      );
    } catch (error) {
      logger.error(
        "[MediaConverter] Failed to create temp conversion directory:",
        error
      );
      throw new ApiError(500, "Failed to initialize media conversion service");
    }
  }

  /**
   * Main conversion entry point
   * @param {string} inputPath - Path to the input file
   * @param {string} mimetype - MIME type of the input file
   * @param {string} originalName - Original filename
   * @returns {Promise<Object>} Conversion result with converted file path and metadata
   */
  async convertMedia(inputPath, mimetype, originalName) {
    try {
      logger.info(
        `[MediaConverter] Starting conversion for ${originalName} (${mimetype})`
      );

      // Check if conversion is needed
      if (!this.needsConversion(mimetype)) {
        logger.info(`[MediaConverter] No conversion needed for ${mimetype}`);
        return {
          needsConversion: false,
          outputPath: inputPath,
          originalPath: inputPath,
          mimetype: mimetype,
          metadata: await this.extractMetadata(inputPath),
        };
      }

      // Extract metadata before conversion
      const originalMetadata = await this.extractMetadata(inputPath);

      // Determine target format and perform conversion
      const isVideo = this.VIDEO_FORMATS.has(mimetype);
      const isAudio = this.AUDIO_FORMATS.has(mimetype);

      let conversionResult;
      if (isVideo) {
        conversionResult = await this.convertVideo(
          inputPath,
          originalName,
          originalMetadata
        );
      } else if (isAudio) {
        conversionResult = await this.convertAudio(
          inputPath,
          originalName,
          originalMetadata
        );
      } else {
        throw new ApiError(415, `Unsupported media type: ${mimetype}`);
      }

      logger.info(`[MediaConverter] Successfully converted ${originalName}`);
      return {
        needsConversion: true,
        ...conversionResult,
        originalPath: inputPath,
        originalMetadata,
      };
    } catch (error) {
      logger.error(
        `[MediaConverter] Conversion failed for ${originalName}:`,
        error
      );
      throw new ApiError(500, `Media conversion failed: ${error.message}`);
    }
  }

  /**
   * Convert video files to MP4
   */
  async convertVideo(inputPath, originalName, metadata) {
    const outputFileName = this.generateOutputFileName(
      originalName,
      this.TARGET_VIDEO_FORMAT
    );
    const outputPath = path.join(this.TEMP_CONVERSION_DIR, outputFileName);

    return new Promise((resolve, reject) => {
      const startTime = Date.now();
      let conversion = ffmpeg(inputPath);

      // Video codec settings
      conversion
        .videoCodec(this.VIDEO_SETTINGS.codec)
        .audioCodec(this.VIDEO_SETTINGS.audioCodec)
        .preset(this.VIDEO_SETTINGS.preset)
        .addOption("-crf", this.VIDEO_SETTINGS.crf)
        .videoBitrate(this.VIDEO_SETTINGS.maxBitrate)
        .audioBitrate(this.VIDEO_SETTINGS.audioBitrate);

      // Preserve metadata
      conversion.addOption("-map_metadata", "0");

      // Ensure compatibility
      conversion
        .addOption("-movflags", "+faststart") // Web optimization
        .addOption("-pix_fmt", "yuv420p") // Broad compatibility
        .format("mp4");

      // Handle different input resolutions intelligently
      if (metadata.video?.width && metadata.video?.height) {
        // Scale down if too large, maintain aspect ratio
        if (metadata.video.width > 1920 || metadata.video.height > 1080) {
          conversion.size("1920x1080");
        }
      }

      conversion
        .on("start", (commandLine) => {
          logger.debug(`[MediaConverter] FFmpeg command: ${commandLine}`);
        })
        .on("progress", (progress) => {
          if (progress.percent) {
            logger.debug(
              `[MediaConverter] Video conversion progress: ${progress.percent.toFixed(
                1
              )}%`
            );
          }
        })
        .on("end", async () => {
          const conversionTime = Date.now() - startTime;
          logger.info(
            `[MediaConverter] Video conversion completed in ${conversionTime}ms`
          );

          try {
            const convertedMetadata = await this.extractMetadata(outputPath);
            const stats = await fs.stat(outputPath);

            resolve({
              outputPath,
              mimetype: "video/mp4",
              size: stats.size,
              metadata: convertedMetadata,
              conversionTime,
            });
          } catch (error) {
            reject(
              new ApiError(
                500,
                `Failed to verify converted video: ${error.message}`
              )
            );
          }
        })
        .on("error", (error) => {
          logger.error(`[MediaConverter] Video conversion error:`, error);
          reject(
            new ApiError(500, `Video conversion failed: ${error.message}`)
          );
        })
        .save(outputPath);
    });
  }

  /**
   * Convert audio files to MP3
   */
  async convertAudio(inputPath, originalName, metadata) {
    const outputFileName = this.generateOutputFileName(
      originalName,
      this.TARGET_AUDIO_FORMAT
    );
    const outputPath = path.join(this.TEMP_CONVERSION_DIR, outputFileName);

    return new Promise((resolve, reject) => {
      const startTime = Date.now();

      ffmpeg(inputPath)
        .audioCodec(this.AUDIO_SETTINGS.codec)
        .audioBitrate(this.AUDIO_SETTINGS.bitrate)
        .audioChannels(this.AUDIO_SETTINGS.channels)
        .audioFrequency(this.AUDIO_SETTINGS.sampleRate)
        .addOption("-map_metadata", "0") // Preserve metadata
        .format("mp3")
        .on("start", (commandLine) => {
          logger.debug(`[MediaConverter] FFmpeg command: ${commandLine}`);
        })
        .on("progress", (progress) => {
          if (progress.percent) {
            logger.debug(
              `[MediaConverter] Audio conversion progress: ${progress.percent.toFixed(
                1
              )}%`
            );
          }
        })
        .on("end", async () => {
          const conversionTime = Date.now() - startTime;
          logger.info(
            `[MediaConverter] Audio conversion completed in ${conversionTime}ms`
          );

          try {
            const convertedMetadata = await this.extractMetadata(outputPath);
            const stats = await fs.stat(outputPath);

            resolve({
              outputPath,
              mimetype: "audio/mpeg",
              size: stats.size,
              metadata: convertedMetadata,
              conversionTime,
            });
          } catch (error) {
            reject(
              new ApiError(
                500,
                `Failed to verify converted audio: ${error.message}`
              )
            );
          }
        })
        .on("error", (error) => {
          logger.error(`[MediaConverter] Audio conversion error:`, error);
          reject(
            new ApiError(500, `Audio conversion failed: ${error.message}`)
          );
        })
        .save(outputPath);
    });
  }

  /**
   * Extract comprehensive metadata from media files
   */
  async extractMetadata(filePath) {
    return new Promise((resolve, reject) => {
      ffmpeg.ffprobe(filePath, (error, metadata) => {
        if (error) {
          logger.error(
            `[MediaConverter] Metadata extraction failed for ${filePath}:`,
            error
          );
          reject(
            new ApiError(500, `Failed to extract metadata: ${error.message}`)
          );
          return;
        }

        try {
          const processedMetadata = this.processMetadata(metadata);
          resolve(processedMetadata);
        } catch (processError) {
          logger.error(
            `[MediaConverter] Metadata processing failed:`,
            processError
          );
          reject(
            new ApiError(
              500,
              `Failed to process metadata: ${processError.message}`
            )
          );
        }
      });
    });
  }

  /**
   * Process raw FFprobe metadata into structured format
   */
  processMetadata(rawMetadata) {
    const processed = {
      format: {},
      video: null,
      audio: null,
      streams: [],
    };

    // Format information
    if (rawMetadata.format) {
      processed.format = {
        filename: rawMetadata.format.filename,
        duration: parseFloat(rawMetadata.format.duration) || 0,
        size: parseInt(rawMetadata.format.size) || 0,
        bitRate: parseInt(rawMetadata.format.bit_rate) || 0,
        formatName: rawMetadata.format.format_name,
        formatLongName: rawMetadata.format.format_long_name,
        tags: rawMetadata.format.tags || {},
      };
    }

    // Stream information
    if (rawMetadata.streams && Array.isArray(rawMetadata.streams)) {
      rawMetadata.streams.forEach((stream) => {
        const streamInfo = {
          index: stream.index,
          codecType: stream.codec_type,
          codecName: stream.codec_name,
          codecLongName: stream.codec_long_name,
          duration: parseFloat(stream.duration) || 0,
          tags: stream.tags || {},
        };

        if (stream.codec_type === "video") {
          processed.video = {
            ...streamInfo,
            width: stream.width,
            height: stream.height,
            aspectRatio: stream.display_aspect_ratio,
            frameRate: this.parseFrameRate(stream.r_frame_rate),
            bitRate: parseInt(stream.bit_rate) || 0,
            pixelFormat: stream.pix_fmt,
          };
        } else if (stream.codec_type === "audio") {
          processed.audio = {
            ...streamInfo,
            sampleRate: parseInt(stream.sample_rate) || 0,
            channels: stream.channels,
            channelLayout: stream.channel_layout,
            bitRate: parseInt(stream.bit_rate) || 0,
            sampleFormat: stream.sample_fmt,
          };
        }

        processed.streams.push(streamInfo);
      });
    }

    return processed;
  }

  /**
   * Parse frame rate from FFprobe string format
   */
  parseFrameRate(frameRateString) {
    if (!frameRateString) return 0;

    try {
      const parts = frameRateString.split("/");
      if (parts.length === 2) {
        const numerator = parseInt(parts[0]);
        const denominator = parseInt(parts[1]);
        return denominator !== 0 ? numerator / denominator : 0;
      }
      return parseFloat(frameRateString);
    } catch {
      return 0;
    }
  }

  /**
   * Check if media needs conversion
   */
  needsConversion(mimetype) {
    // Videos: convert if not MP4
    if (this.VIDEO_FORMATS.has(mimetype)) {
      return mimetype !== "video/mp4";
    }

    // Audio: convert if not MP3
    if (this.AUDIO_FORMATS.has(mimetype)) {
      return mimetype !== "audio/mpeg";
    }

    return false;
  }

  /**
   * Generate output filename with proper extension
   */
  generateOutputFileName(originalName, targetExtension) {
    const baseName = path.parse(originalName).name;
    const timestamp = Date.now();
    return `converted-${baseName}-${timestamp}.${targetExtension}`;
  }

  /**
   * Clean up temporary conversion files
   */
  async cleanupTempFiles(filePaths) {
    const cleanupPromises = filePaths.map(async (filePath) => {
      try {
        await fs.unlink(filePath);
        logger.debug(`[MediaConverter] Cleaned up temp file: ${filePath}`);
      } catch (error) {
        logger.warn(
          `[MediaConverter] Failed to cleanup temp file ${filePath}:`,
          error
        );
      }
    });

    await Promise.allSettled(cleanupPromises);
  }

  /**
   * Get supported formats information
   */
  getSupportedFormats() {
    return {
      video: {
        input: Array.from(this.VIDEO_FORMATS),
        output: [`video/${this.TARGET_VIDEO_FORMAT}`],
      },
      audio: {
        input: Array.from(this.AUDIO_FORMATS),
        output: [`audio/${this.TARGET_AUDIO_FORMAT}`],
      },
    };
  }

  /**
   * Validate FFmpeg installation
   */
  async validateFFmpegInstallation() {
    return new Promise((resolve) => {
      ffmpeg().version((error, version) => {
        if (error) {
          logger.error("[MediaConverter] FFmpeg validation failed:", error);
          resolve({ isValid: false, error: error.message });
        } else {
          logger.info(`[MediaConverter] FFmpeg validated: ${version}`);
          resolve({ isValid: true, version });
        }
      });
    });
  }
}

export const mediaConverterService = new MediaConverterService();
