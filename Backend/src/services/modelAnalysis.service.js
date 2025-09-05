// src/services/modelAnalysis.service.js

import axios from 'axios';
import FormData from 'form-data';
import fs from 'fs';
import path from 'path';
import { ApiError } from '../utils/ApiError.js';
import logger from '../utils/logger.js';
import { config } from '../config/env.js';
import { mediaRepository } from '../repositories/media.repository.js';

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
        let statsPayload;
        try {
            const startTime = Date.now();
            const response = await axios.get(`${this.serverUrl}/stats`, {
                timeout: this.healthTimeout,
                headers: { 'X-API-Key': this.apiKey },
            });
            const responseTimeMs = Date.now() - startTime;
            statsPayload = { ...response.data, responseTimeMs };

            mediaRepository.storeServerHealth(statsPayload).catch(err => {
                logger.error(`[ModelAnalysisService] Failed to store server health in background: ${err.message}`);
            });
            
            return statsPayload;
        } catch (error) {
            const errorPayload = { status: 'UNHEALTHY', errorMessage: error.message, responseTimeMs: this.healthTimeout };
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
    
    _handleApiError(error, operation, logId = '') {
        logger.error(`[ModelAnalysisService] API Error during ${operation} for ${logId}: ${error.message}`);
        if (error.response) {
            const { status, data } = error.response;
            const message = data?.detail || data?.message || 'Analysis failed on the model server.';
            logger.error(`[ModelAnalysisService] Server responded with status ${status}: ${JSON.stringify(data)}`);
            throw new ApiError(status, message, data?.details || data);
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