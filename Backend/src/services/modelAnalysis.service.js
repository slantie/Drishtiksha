// src/services/modelAnalysis.service.js

import axios from 'axios';
import FormData from 'form-data';
import fs from 'fs';
import path from 'path';
import { ApiError } from '../utils/ApiError.js';
import logger from '../utils/logger.js';
import { config } from '../config/env.js';
import { mediaRepository } from '../repositories/media.repository.js';
import { redisCache } from '../config/index.js';

// Cache configuration constants
const SERVER_STATS_CACHE_KEY = 'ml_server_stats';
const SERVER_STATS_CACHE_TTL = 60; // seconds

class ModelAnalysisService {
    constructor() {
        this.serverUrl = config.SERVER_URL;
        this.apiKey = config.SERVER_API_KEY;
        this.requestTimeout = 1200000; // 20 minutes
        this.healthTimeout = 20000; // 20 seconds

        if (!this.isAvailable()) {
            logger.warn('SERVER_URL or SERVER_API_KEY is not configured. Model analysis service is disabled.');
        } else {
            logger.info(`Model analysis service initialized for URL: ${this.serverUrl}`);
        }
    }

    isAvailable() {
        return !!(this.serverUrl && this.apiKey);
    }

    async getServerStatistics() {
        if (!this.isAvailable()) {
            throw new ApiError(503, 'Model service is not configured.');
        }

        try {
            // 1. Try Redis cache first
            const cached = await redisCache.get(SERVER_STATS_CACHE_KEY);
            if (cached) {
                logger.debug('[ModelAnalysisService] Returning cached ML server stats');
                return JSON.parse(cached);
            }

            // 2. Cache miss - fetch from ML server
            logger.debug('[ModelAnalysisService] Cache miss - fetching ML server stats');
            const startTime = Date.now();
            const response = await axios.get(`${this.serverUrl}/stats`, {
                timeout: this.healthTimeout,
                headers: { 'X-API-Key': this.apiKey },
            });
            const responseTimeMs = Date.now() - startTime;
            
            const statsPayload = { 
                ...response.data, 
                responseTimeMs,
                cachedAt: new Date().toISOString()
            };

            // 3. Store in Redis with TTL
            await redisCache.set(
                SERVER_STATS_CACHE_KEY,
                JSON.stringify(statsPayload),
                'EX',
                SERVER_STATS_CACHE_TTL
            );
            
            logger.info(`[ModelAnalysisService] ML server stats fetched and cached (${responseTimeMs}ms, TTL: ${SERVER_STATS_CACHE_TTL}s)`);

            // 4. Store in database for historical tracking (background)
            mediaRepository.storeServerHealth(statsPayload).catch(err => {
                logger.error(`[ModelAnalysisService] Failed to store server health in background: ${err.message}`);
            });
            
            return statsPayload;
        } catch (error) {
            // On error, don't cache and still try to store error state
            const errorPayload = { 
                status: 'UNHEALTHY', 
                errorMessage: error.message, 
                responseTimeMs: this.healthTimeout 
            };
            
            mediaRepository.storeServerHealth(errorPayload).catch(err => {
                logger.error(`[ModelAnalysisService] Failed to store FAILED server health in background: ${err.message}`);
            });
            
            this._handleApiError(error, 'STATS');
        }
    }

    async runAnalysis(mediaPath, modelName, mediaId, userId) {
        if (!this.isAvailable()) throw new ApiError(503, 'Model analysis service is not configured.');
        if (!fs.existsSync(mediaPath)) throw new ApiError(404, `Media file not found at path: ${mediaPath}`);

        const logId = mediaId || path.basename(mediaPath);
        logger.info(`[ModelAnalysisService] Starting analysis for media ${logId} with model ${modelName}`);

        try {
            const formData = new FormData();
            formData.append('media', fs.createReadStream(mediaPath));
            formData.append('model', modelName);
            if (mediaId) formData.append('mediaId', mediaId);
            if (userId) formData.append('userId', userId);

            const response = await axios.post(`${this.serverUrl}/analyze`, formData, {
                headers: { ...formData.getHeaders(), 'X-API-Key': this.apiKey },
                timeout: this.requestTimeout,
                maxContentLength: Infinity,
                maxBodyLength: Infinity,
            });

            if (!response.data?.success) {
                throw new ApiError(500, 'Analysis failed with an invalid response from the model server.');
            }
            logger.info(`[ModelAnalysisService] Analysis completed for media ${logId}`);
            return response.data.data;
        } catch (error) {
            this._handleApiError(error, 'ANALYSIS', logId);
        }
    }

    async downloadVisualization(filename) {
        if (!this.isAvailable()) throw new ApiError(503, 'Model service is not configured.');
        try {
            const response = await axios.get(`${this.serverUrl}/analyze/visualization/${filename}`, {
                headers: { 'X-API-Key': this.apiKey },
                responseType: 'stream',
            });
            return response.data;
        } catch (error) {
            this._handleApiError(error, 'VISUALIZATION_DOWNLOAD', filename);
        }
    }

    /**
     * Manually invalidate the ML server stats cache.
     * Useful for testing or when you know the server state has changed.
     */
    async invalidateStatsCache() {
        try {
            const result = await redisCache.del(SERVER_STATS_CACHE_KEY);
            if (result === 1) {
                logger.info('[ModelAnalysisService] ML server stats cache invalidated successfully');
            } else {
                logger.debug('[ModelAnalysisService] No cache to invalidate (already empty)');
            }
            return result;
        } catch (error) {
            logger.error('[ModelAnalysisService] Failed to invalidate stats cache:', error);
            throw error;
        }
    }
    
    _handleApiError(error, operation, logId = '') {
        logger.error(`[ModelAnalysisService] API Error during ${operation} for ${logId}: ${error.message}`);
        if (error.response) {
            const { status, data } = error.response;
            
            // Extract meaningful error message (ensure it's a string)
            let message;
            if (typeof data?.detail === 'string') {
                message = data.detail;
            } else if (typeof data?.detail?.message === 'string') {
                message = data.detail.message;
            } else if (typeof data?.message === 'string') {
                message = data.message;
            } else if (typeof data?.detail === 'object') {
                message = data.detail.error || data.detail.message || 'Analysis failed on the model server.';
            } else {
                message = 'Analysis failed on the model server.';
            }
            
            logger.error(`[ModelAnalysisService] Server responded with status ${status}: ${JSON.stringify(data)}`);
            
            // Create error with string message and full data as details
            const apiError = new ApiError(status, message, data);
            // Store the full error response for debugging
            apiError.serverResponse = data;
            throw apiError;
        } else if (error.code === 'ECONNREFUSED') {
            throw new ApiError(503, 'Model analysis service is unavailable. Connection refused.');
        } else if (error.code === 'ETIMEDOUT' || error.code === 'ECONNABORTED') {
            throw new ApiError(504, 'Request to model analysis service timed out.');
        } else {
            throw new ApiError(500, `An unknown error occurred while communicating with the model service: ${error.message}`);
        }
    }
}

export const modelAnalysisService = new ModelAnalysisService();