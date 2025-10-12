// src/services/pdf.service.js

import { mdToPdf } from 'md-to-pdf';
import path from 'path';
import { fileURLToPath } from 'url';
import { mediaService } from './media.service.js';
import logger from '../utils/logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * PDF Generation Service using md-to-pdf
 * Handles the complete flow: fetch data → generate markdown → create PDF
 */
class PDFService {
    constructor() {
        // Path to the CSS stylesheet
        this.stylesheetPath = path.resolve(__dirname, '../../pdf-assets/report.css');
        
        // md-to-pdf configuration
        this.pdfConfig = {
            stylesheet: this.stylesheetPath,
            body_class: 'markdown-body',
            pdf_options: {
                format: 'A4',
                margin: {
                    top: '20mm',
                    right: '15mm',
                    bottom: '20mm',
                    left: '15mm',
                },
                printBackground: true,
                preferCSSPageSize: true,
            },
            marked_options: {
                headerIds: true,
                mangle: false,
            },
        };
    }

    /**
     * Generate complete PDF report for a specific analysis run
     * @param {string} analysisRunId - Analysis Run UUID
     * @param {string} userId - User UUID
     * @returns {Promise<{pdfBuffer: Buffer, filename: string}>}
     */
    async generateRunReportPDF(analysisRunId, userId) {
        try {
            logger.info(`[PDF Service] Starting report generation for analysis run ${analysisRunId}`);
            
            // 1. Fetch analysis run data with media and analyses
            const analysisRunData = await this.getAnalysisRunData(analysisRunId, userId);
            
            if (!analysisRunData) {
                throw new Error('Analysis run not found or access denied');
            }
            
            logger.debug(`[PDF Service] Fetched analysis run data for media ${analysisRunData.id}`);
            
            // 2. Prepare data (enrich with calculations)
            const preparedData = this.prepareReportData(analysisRunData);
            logger.debug(`[PDF Service] Prepared report data with ${preparedData.completedAnalyses.length} completed analyses`);
            
            // 3. Generate markdown from template
            const markdown = this.generateMarkdownReport(preparedData);
            logger.debug(`[PDF Service] Generated markdown report (${markdown.length} characters)`);
            
            // 4. Convert markdown to PDF
            const pdfBuffer = await this.generatePDFFromMarkdown(markdown);
            logger.debug(`[PDF Service] Converted markdown to PDF (${pdfBuffer.length} bytes)`);
            
            // 5. Generate filename
            const filename = this.generateFilename(analysisRunData);
            
            logger.info(`[PDF Service] Report generated successfully: ${filename}`);
            
            return { pdfBuffer, filename };
        } catch (error) {
            logger.error('[PDF Service] Error generating report:', error);
            logger.error('[PDF Service] Stack trace:', error.stack);
            throw new Error(`Failed to generate report: ${error.message}`);
        }
    }

    /**
     * Get analysis run data with media info
     * @param {string} analysisRunId - Analysis Run UUID
     * @param {string} userId - User UUID
     * @returns {Promise<Object>} Analysis run with media and analyses
     */
    async getAnalysisRunData(analysisRunId, userId) {
        try {
            const { PrismaClient } = await import('@prisma/client');
            const prisma = new PrismaClient();
            
            const analysisRun = await prisma.analysisRun.findFirst({
                where: {
                    id: analysisRunId,
                    media: {
                        userId: userId,
                    },
                },
                include: {
                    media: true,
                    analyses: true, // No model relation - modelName is a string field
                },
            });
            
            await prisma.$disconnect();
            
            if (!analysisRun) {
                return null;
            }
            
            // Restructure to match expected format
            return {
                ...analysisRun.media,
                analysisRuns: [analysisRun],
            };
        } catch (error) {
            logger.error('[PDF Service] Error fetching analysis run:', error);
            throw error;
        }
    }

    /**
     * Generate PDF from markdown content
     * @param {string} markdown - Markdown content
     * @returns {Promise<Buffer>} PDF buffer
     */
    async generatePDFFromMarkdown(markdown) {
        try {
            if (!markdown || typeof markdown !== 'string') {
                throw new Error('Invalid markdown content provided');
            }

            // Generate PDF using md-to-pdf
            const pdf = await mdToPdf(
                { content: markdown },
                this.pdfConfig
            );

            if (!pdf || !pdf.content) {
                throw new Error('md-to-pdf did not return valid content');
            }

            // md-to-pdf may return a Uint8Array - convert to Node Buffer to ensure
            // Express sends binary data correctly (res.send with a TypedArray can be
            // interpreted as an object and JSON-ified, corrupting the PDF).
            return Buffer.from(pdf.content);
        } catch (error) {
            logger.error('[PDF Service] Error generating PDF:', error);
            throw new Error(`Failed to generate PDF: ${error.message}`);
        }
    }

    /**
     * Prepare and enrich media data for PDF generation
     * @param {Object} media - Media object from database
     * @returns {Object} Enriched data
     */
    prepareReportData(media) {
        const latestAnalysisRun = media.analysisRuns?.[0];
        
        // Normalize analysis objects and extract data from resultPayload
        const normalizeAnalysis = (analysis) => {
            const payload = analysis.resultPayload || {};
            
            return {
                ...analysis,
                confidence: analysis.confidence ?? payload.confidence ?? 0,
                prediction: analysis.prediction ?? payload.prediction ?? 'UNKNOWN',
                processingTime: analysis.processingTime ?? payload.processing_time ?? payload.processingTime ?? 0,
                framePredictions: payload.frame_predictions || [],
                framesAnalyzed: payload.frames_analyzed || payload.framesAnalyzed || 0,
                frameCount: payload.frame_count || payload.frameCount || null,
                mediaType: payload.media_type || payload.mediaType || media.mediaType,
                metrics: payload.metrics || {},
                note: payload.note || null,
                visualizationPath: payload.visualization_path || null,
                analyzedAt: analysis.createdAt ?? analysis.updatedAt,
            };
        };
        
        // Separate completed and failed analyses
        const completedAnalyses = (latestAnalysisRun?.analyses?.filter(
            (a) => a.status === 'COMPLETED'
        ) || []).map(normalizeAnalysis);
        
        const failedAnalyses = (latestAnalysisRun?.analyses?.filter(
            (a) => a.status === 'FAILED'
        ) || []).map(normalizeAnalysis);
        
        // Calculate statistics
        const realDetections = completedAnalyses.filter((a) => a.prediction === 'REAL').length;
        const fakeDetections = completedAnalyses.filter((a) => a.prediction === 'FAKE').length;
        const totalModels = completedAnalyses.length + failedAnalyses.length;
        
        const avgConfidence = completedAnalyses.length > 0
            ? completedAnalyses.reduce((sum, a) => sum + (a.confidence || 0), 0) / completedAnalyses.length
            : 0;
        
        // Calculate average processing time
        const avgProcessingTime = completedAnalyses.length > 0
            ? completedAnalyses.reduce((sum, a) => sum + (a.processingTime || 0), 0) / completedAnalyses.length
            : 0;
        
        // Find highest and lowest confidence
        const highestConfidence = completedAnalyses.length > 0
            ? Math.max(...completedAnalyses.map(a => a.confidence || 0))
            : 0;
        
        const lowestConfidence = completedAnalyses.length > 0
            ? Math.min(...completedAnalyses.map(a => a.confidence || 0))
            : 0;
        
        // Sort by confidence (descending)
        const sortedAnalyses = [...completedAnalyses].sort(
            (a, b) => (b.confidence || 0) - (a.confidence || 0)
        );
        
        // Determine overall assessment based on majority vote
        const overallAssessment = fakeDetections > realDetections ? 'DEEPFAKE' : 'AUTHENTIC';
        
        // Determine risk level based on:
        // - Confidence spread (consistency)
        // - Number of fake detections
        // - Average confidence
        const confidenceSpread = highestConfidence - lowestConfidence;
        let riskLevel = 'LOW';
        
        if (fakeDetections > 0) {
            if (fakeDetections >= completedAnalyses.length * 0.6 && avgConfidence > 0.7) {
                riskLevel = 'HIGH';
            } else if (fakeDetections >= completedAnalyses.length * 0.4) {
                riskLevel = 'MEDIUM';
            } else {
                riskLevel = 'LOW';
            }
        }
        
        // Extract metadata from media
        const metadata = media.metadata || {};
        
        return {
            ...media,
            metadata,
            latestAnalysisRun,
            completedAnalyses: sortedAnalyses,
            failedAnalyses,
            totalModels,
            realDetections,
            fakeDetections,
            avgConfidence,
            avgProcessingTime,
            highestConfidence,
            lowestConfidence,
            confidenceSpread,
            overallAssessment,
            riskLevel,
            reportId: `RPT-${Date.now()}`,
            generatedAt: new Date().toISOString(),
        };
    }

    /**
     * Generate markdown report from prepared data
     * @param {Object} data - Prepared report data
     * @returns {string} Markdown content
     */
    generateMarkdownReport(data) {
        // Generate only the sections we want
        const sections = [
            this.generateCoverPage(data),
            this.generateTableOfContents(),
            this.generateExecutiveSummary(data),
            this.generateMediaInformation(data),
            this.generateAnalysisOverview(data),
            this.generateModelComparison(data),
            this.generateDetailedModelResults(data),
            this.generateTemporalAnalysis(data),
            // Technical metadata removed per user request
            this.generateMethodology(data),
            // Recommendations removed per user request
            this.generateFooter(data),
        ];
        
        return sections.filter(Boolean).join('\n\n');
    }

    /**
     * Generate filename for the PDF
     * @param {Object} media - Media object
     * @returns {string} Filename
     */
    generateFilename(media) {
        const date = new Date().toISOString().split('T')[0];
        const mediaName = (media.originalFilename || media.filename || 'media')
            .replace(/\.[^/.]+$/, '') // Remove extension
            .replace(/[^a-z0-9]/gi, '_') // Replace special chars
            .substring(0, 30); // Limit length
        
        return `Analysis_Report_${mediaName}_${date}.pdf`;
    }

    // Helper methods for markdown generation
    formatDate(date) {
        if (!date) return 'N/A';
        return new Date(date).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        });
    }

    formatBytes(bytes) {
        if (!bytes) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
    }

    formatDuration(seconds) {
        if (!seconds || seconds < 1) return 'less than a second';
        if (seconds < 60) return `${Math.round(seconds)}s`;
        
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.round(seconds % 60);
        
        if (minutes < 60) {
            return remainingSeconds === 0 ? `${minutes}m` : `${minutes}m ${remainingSeconds}s`;
        }
        
        const hours = Math.floor(minutes / 60);
        const remainingMinutes = minutes % 60;
        
        return remainingMinutes === 0 ? `${hours}h` : `${hours}h ${remainingMinutes}m`;
    }

    formatBitrate(bitsPerSecond) {
        if (!bitsPerSecond) return '0 bps';
        const kbps = bitsPerSecond / 1000;
        const mbps = kbps / 1000;
        
        if (mbps >= 1) {
            return `${mbps.toFixed(2)} Mbps`;
        } else {
            return `${kbps.toFixed(2)} Kbps`;
        }
    }

    // Markdown section generators (condensed versions - we'll expand these)
    generateCoverPage(data) {
        return `<div class="cover-page">

# Deepfake Detection Analysis Report

<div class="subtitle">Comprehensive Media Authenticity Assessment</div>

<div class="meta">

**Media ID:** \`${data.id || 'N/A'}\`

**Analysis Date:** ${this.formatDate(data.latestAnalysisRun?.createdAt || new Date())}

**System Version:** v2.0.0

Generated by VidVigilante Detection System

</div>

</div>`;
    }

    generateTableOfContents() {
        return `## Table of Contents

<div class="toc">

1. [Executive Summary](#executive-summary)
2. [Media Information](#media-information)
3. [Analysis Overview](#analysis-overview)
4. [Model Comparison](#model-comparison)
5. [Detailed Model Results](#detailed-model-results)
6. [Temporal Analysis](#temporal-analysis)
7. [Methodology](#methodology)

</div>

<div class="page-break"></div>`;
    }

    generateExecutiveSummary(data) {
        const isFake = data.fakeDetections > data.realDetections;
        const badgeClass = isFake ? 'badge-deepfake' : 'badge-authentic';
        const badgeText = isFake ? 'DEEPFAKE DETECTED' : 'AUTHENTIC CONTENT';
        const confidencePercent = `${Math.round(data.avgConfidence * 100)}%`;
        const successRate = data.totalModels > 0 
            ? Math.round((data.completedAnalyses.length / data.totalModels) * 100)
            : 0;
        
        return `## Executive Summary

<div class="info-box">

### Assessment Results

<div class="metadata-grid">

<div class="metadata-item">
<div class="metadata-label">Overall Verdict</div>
<div class="metadata-value"><span class="badge ${badgeClass}">${badgeText}</span></div>
</div>

<div class="metadata-item">
<div class="metadata-label">Confidence Level</div>
<div class="metadata-value">${confidencePercent}</div>
</div>

<div class="metadata-item">
<div class="metadata-label">Risk Assessment</div>
<div class="metadata-value"><span class="badge badge-${data.riskLevel.toLowerCase()}">${data.riskLevel} RISK</span></div>
</div>

<div class="metadata-item">
<div class="metadata-label">Analysis Success Rate</div>
<div class="metadata-value">${successRate}% (${data.completedAnalyses.length}/${data.totalModels})</div>
</div>

</div>

</div>

### Key Findings

This comprehensive analysis examined a **${data.mediaType?.toLowerCase() || 'media'}** file using ${data.totalModels} state-of-the-art deepfake detection models powered by advanced deep learning algorithms.

**Detection Summary:**

- **${data.realDetections} models** detected content as **AUTHENTIC**
- **${data.fakeDetections} models** flagged content as **DEEPFAKE**
- **Consensus:** ${Math.round((Math.max(data.realDetections, data.fakeDetections) / data.completedAnalyses.length) * 100)}% model agreement
- **Average confidence:** ${confidencePercent}
- **Confidence range:** ${Math.round(data.lowestConfidence * 100)}% - ${Math.round(data.highestConfidence * 100)}%
- **Processing time:** ${this.formatDuration(data.avgProcessingTime)} average per model

${data.failedAnalyses.length > 0 ? `
**⚠️ Note:** ${data.failedAnalyses.length} model(s) failed during analysis. This may indicate:
- Missing required media features (e.g., audio track for audio-visual models)
- Incompatible media format or encoding
- Technical issues during processing

Failed models: ${data.failedAnalyses.map(a => a.modelName).join(', ')}
` : ''}

### Interpretation

${isFake ? `<div class="danger-box">

**⚠️ CAUTION:** This media has been flagged as **potentially manipulated or synthetic**.

**Risk Level: ${data.riskLevel}**

${data.riskLevel === 'HIGH' ? `
This indicates a strong consensus among detection models with high confidence. The content should be treated as suspicious and should not be used for official purposes without additional verification.
` : data.riskLevel === 'MEDIUM' ? `
This indicates moderate suspicion with some disagreement among models. Additional forensic analysis is recommended before making definitive conclusions.
` : `
While some models detected manipulation, the overall confidence is relatively low. This could indicate subtle manipulation or potential false positives. Further investigation is warranted.
`}

**Recommended Actions:**
- Do not distribute or rely on this content for critical decisions
- Conduct manual forensic analysis if authenticity is crucial
- Cross-reference with original source material if available
- Consider metadata analysis and provenance verification

</div>` : `<div class="success-box">

**✓ PRELIMINARY ASSESSMENT:** The media appears to be **authentic** based on current analysis.

**Confidence: ${data.riskLevel === 'LOW' ? 'High' : 'Moderate'}**

The majority of detection models classified this content as authentic with ${data.avgConfidence > 0.8 ? 'high' : 'moderate'} confidence. ${data.confidenceSpread < 0.2 ? 'Model predictions show strong consistency.' : 'Some variation exists among model predictions, which is normal for authentic content.'}

**Important Note:**
- No detection system is perfect. Always consider the source and context.
- Authentic-looking content can still be misleading through editing or selective framing.
- For critical applications, consider additional verification methods.

</div>`}

<div class="page-break"></div>`;
    }

    generateMediaInformation(data) {
        // Extract metadata from media object - it's stored as JSON in metadata field
        const metadata = data.metadata || {};
        const format = metadata.format || {};
        const video = metadata.video || {};
        const audio = metadata.audio || {};
        
        // Safely extract numeric values with fallbacks
        const fileSize = Number(format.size || data.size) || 0;
        const duration = Number(format.duration) || 0;
        const bitRate = Number(format.bitRate) || 0;
        
        // Helper to parse frame rate (can be "30/1" string or number)
        const parseFrameRate = (fr) => {
            if (!fr) return null;
            if (typeof fr === 'number') return fr;
            if (typeof fr === 'string' && fr.includes('/')) {
                const [num, den] = fr.split('/').map(Number);
                return den !== 0 ? num / den : null;
            }
            return Number(fr) || null;
        };
        
        return `## Media Information

<div class="card">
<div class="card-header">Basic File Information</div>

<div class="metadata-grid">

<div class="metadata-item">
<div class="metadata-label">Filename</div>
<div class="metadata-value">${data.filename || 'Unknown'}</div>
</div>

<div class="metadata-item">
<div class="metadata-label">File Size</div>
<div class="metadata-value">${this.formatBytes(fileSize)}</div>
</div>

<div class="metadata-item">
<div class="metadata-label">Media Type</div>
<div class="metadata-value">${data.mediaType || 'VIDEO'}</div>
</div>

<div class="metadata-item">
<div class="metadata-label">MIME Type</div>
<div class="metadata-value">${data.mimetype || 'N/A'}</div>
</div>

<div class="metadata-item">
<div class="metadata-label">Duration</div>
<div class="metadata-value">${duration > 0 ? this.formatDuration(duration) : 'N/A'}</div>
</div>

<div class="metadata-item">
<div class="metadata-label">Bit Rate</div>
<div class="metadata-value">${bitRate > 0 ? this.formatBitrate(bitRate) : 'N/A'}</div>
</div>

<div class="metadata-item">
<div class="metadata-label">Upload Date</div>
<div class="metadata-value">${this.formatDate(data.createdAt)}</div>
</div>

<div class="metadata-item">
<div class="metadata-label">Media ID</div>
<div class="metadata-value">\`${data.id}\`</div>
</div>

</div>

</div>

${video.width ? `
<div class="card">
<div class="card-header">Video Stream Information</div>

<div class="metadata-grid">

<div class="metadata-item">
<div class="metadata-label">Resolution</div>
<div class="metadata-value">${video.width} × ${video.height}</div>
</div>

<div class="metadata-item">
<div class="metadata-label">Aspect Ratio</div>
<div class="metadata-value">${video.aspectRatio || 'N/A'}</div>
</div>

<div class="metadata-item">
<div class="metadata-label">Frame Rate</div>
<div class="metadata-value">${(() => {
    const fps = parseFrameRate(video.frameRate);
    return fps ? `${fps.toFixed(2)} fps` : (video.frameRate || 'N/A');
})()}</div>
</div>

<div class="metadata-item">
<div class="metadata-label">Video Codec</div>
<div class="metadata-value">${video.codecName || video.codecLongName || 'N/A'}</div>
</div>

<div class="metadata-item">
<div class="metadata-label">Pixel Format</div>
<div class="metadata-value">${video.pixelFormat || 'N/A'}</div>
</div>

<div class="metadata-item">
<div class="metadata-label">Video Bit Rate</div>
<div class="metadata-value">${video.bitRate && Number(video.bitRate) > 0 ? this.formatBitrate(Number(video.bitRate)) : 'N/A'}</div>
</div>

</div>

</div>
` : ''}

${audio.sampleRate ? `
<div class="card">
<div class="card-header">Audio Stream Information</div>

<div class="metadata-grid">

<div class="metadata-item">
<div class="metadata-label">Audio Present</div>
<div class="metadata-value">✓ Yes</div>
</div>

<div class="metadata-item">
<div class="metadata-label">Audio Codec</div>
<div class="metadata-value">${audio.codecName || audio.codecLongName || 'N/A'}</div>
</div>

<div class="metadata-item">
<div class="metadata-label">Sample Rate</div>
<div class="metadata-value">${audio.sampleRate ? `${Number(audio.sampleRate).toLocaleString()} Hz` : 'N/A'}</div>
</div>

<div class="metadata-item">
<div class="metadata-label">Channels</div>
<div class="metadata-value">${audio.channels || 'N/A'}${audio.channelLayout ? ` (${audio.channelLayout})` : ''}</div>
</div>

<div class="metadata-item">
<div class="metadata-label">Audio Bit Rate</div>
<div class="metadata-value">${audio.bitRate && Number(audio.bitRate) > 0 ? this.formatBitrate(Number(audio.bitRate)) : 'N/A'}</div>
</div>

<div class="metadata-item">
<div class="metadata-label">Sample Format</div>
<div class="metadata-value">${audio.sampleFormat || 'N/A'}</div>
</div>

</div>

</div>
` : data.mediaType === 'VIDEO' && data.hasAudio === false ? `
<div class="warning-box">

**⚠️ No Audio Track Detected**

This video does not contain an audio stream. Some audio-visual detection models may not be applicable.

</div>
` : ''}

<div class="page-break"></div>`;
    }

    generateAnalysisOverview(data) {
        return `## Analysis Overview

### Detection Process

The analysis pipeline processed the media through ${data.totalModels} specialized detection models.

### Confidence Distribution

<div class="info-box">

**Average Confidence:** ${Math.round(data.avgConfidence * 100)}%

**Models Completed:** ${data.completedAnalyses.length} / ${data.totalModels}

**Failed Models:** ${data.failedAnalyses.length}

</div>

<div class="page-break"></div>`;
    }

    generateModelComparison(data) {
        const allAnalyses = [...data.completedAnalyses, ...data.failedAnalyses];
        
        const rows = allAnalyses.map(analysis => {
            const prediction = analysis.prediction || 'N/A';
            const confidence = analysis.confidence ? `${Math.round(analysis.confidence * 100)}%` : 'N/A';
            
            // Get processing time - may be in milliseconds or seconds
            let processingTime = 'N/A';
            if (analysis.processingTime) {
                // If processingTime is greater than 1000, assume it's in milliseconds
                const timeInSeconds = analysis.processingTime > 1000 
                    ? analysis.processingTime / 1000 
                    : analysis.processingTime;
                processingTime = `${timeInSeconds.toFixed(1)}s`;
            }
            
            // Get model name and type from analysis object
            const modelName = analysis.modelName || analysis.model?.name || 'Unknown Model';
            const modelType = analysis.modelType || analysis.analysisTechnique || 'N/A';
            
            let badgeClass = 'badge-neutral';
            let badgeText = prediction;
            
            if (analysis.status === 'COMPLETED') {
                if (prediction === 'FAKE') {
                    badgeClass = 'badge-deepfake';
                    badgeText = 'Deepfake';
                } else if (prediction === 'REAL') {
                    badgeClass = 'badge-authentic';
                    badgeText = 'Authentic';
                }
            } else {
                badgeText = analysis.status;
            }
            
            return `| ${modelName} | ${modelType} | <span class="badge ${badgeClass}">${badgeText}</span> | ${confidence} | ${processingTime} |`;
        }).join('\n');
        
        return `## Model Comparison

| Model Name | Category | Prediction | Confidence | Time |
|------------|----------|------------|------------|------|
${rows}

<div class="page-break"></div>`;
    }

    generateDetailedModelResults(data) {
        if (data.completedAnalyses.length === 0) {
            return `## Detailed Model Results

<div class="warning-box">

**No completed analyses available.**

All models failed during processing. This may indicate incompatible media format or technical issues.

</div>

<div class="page-break"></div>`;
        }
        
        const modelDetails = data.completedAnalyses.map((analysis, index) => {
            const confidence = `${Math.round(analysis.confidence * 100)}%`;
            const badgeClass = analysis.prediction === 'FAKE' ? 'badge-deepfake' : 'badge-authentic';
            const badgeText = analysis.prediction === 'FAKE' ? 'DEEPFAKE DETECTED' : 'AUTHENTIC CONTENT';
            
            const modelName = analysis.modelName || 'Unknown Model';
            const processingTime = analysis.processingTime || 0;
            
            // Frame analysis statistics
            const framePredictions = analysis.framePredictions || [];
            const hasFrameData = framePredictions.length > 0;
            
            let frameStats = '';
            if (hasFrameData) {
                const fakeFrames = framePredictions.filter(f => f.prediction === 'FAKE').length;
                const realFrames = framePredictions.filter(f => f.prediction === 'REAL').length;
                const avgFrameScore = framePredictions.reduce((sum, f) => sum + f.score, 0) / framePredictions.length;
                const maxFrameScore = Math.max(...framePredictions.map(f => f.score));
                const minFrameScore = Math.min(...framePredictions.map(f => f.score));
                
                frameStats = `

**Temporal Analysis Summary:**
- Total frames analyzed: ${framePredictions.length}
- Frames classified as REAL: ${realFrames} (${((realFrames/framePredictions.length)*100).toFixed(1)}%)
- Frames classified as FAKE: ${fakeFrames} (${((fakeFrames/framePredictions.length)*100).toFixed(1)}%)
- Average frame score: ${(avgFrameScore * 100).toFixed(2)}%
- Score range: ${(minFrameScore * 100).toFixed(2)}% - ${(maxFrameScore * 100).toFixed(2)}%
- Confidence consistency: ${fakeFrames === 0 || realFrames === 0 ? 'High' : 'Variable'}`;
            }
            
            // Metrics from result payload
            const metrics = analysis.metrics || {};
            const metricsEntries = Object.entries(metrics);
            const hasMetrics = metricsEntries.length > 0;
            
            return `### ${index + 1}. ${modelName}

<div class="card avoid-break">

**Prediction:** <span class="badge ${badgeClass}">${badgeText}</span>

**Overall Confidence:** ${confidence}

**Processing Time:** ${processingTime.toFixed(3)}s

**Frames Analyzed:** ${analysis.framesAnalyzed || 'N/A'}

**Analysis Date:** ${this.formatDate(analysis.analyzedAt)}
${frameStats}
${hasMetrics ? `

**Advanced Metrics:**
${metricsEntries.map(([key, value]) => `- ${key.replace(/_/g, ' ')}: ${typeof value === 'number' ? value.toFixed(4) : value}`).join('\n')}` : ''}
${analysis.note ? `

**Note:** ${analysis.note}` : ''}

</div>`;
        }).join('\n\n');
        
        return `## Detailed Model Results

${modelDetails}

${data.completedAnalyses.some(a => a.framePredictions && a.framePredictions.length > 0) ? `

### Visualization Note

📊 **Interactive Charts Available Online**

For detailed frame-by-frame visualizations and interactive charts showing temporal analysis trends, model confidence comparisons, and processing time metrics, please view this analysis run in the web application at:

**Dashboard → Media Results → Run #${data.latestAnalysisRun?.runNumber || 'N'}**

The web interface provides:
- Frame-by-frame confidence trends (line charts)
- Model confidence comparison (bar charts)
- Prediction distribution (pie charts)  
- Processing time analysis (bar charts)
- Interactive legends and tooltips for detailed exploration

` : ''}

<div class="page-break"></div>`;
    }

    generateTemporalAnalysis(data) {
        const analysisRun = data.latestAnalysisRun;
        const startTime = analysisRun?.createdAt;
        const endTime = analysisRun?.updatedAt;
        
        // Calculate duration from timestamps if duration field is not available
        let durationSeconds = 0;
        if (analysisRun?.duration) {
            // Duration might be in milliseconds or seconds
            durationSeconds = analysisRun.duration > 1000 
                ? analysisRun.duration / 1000 
                : analysisRun.duration;
        } else if (startTime && endTime) {
            const start = new Date(startTime);
            const end = new Date(endTime);
            durationSeconds = (end - start) / 1000;
        }
        
        return `## Temporal Analysis

<div class="info-box">

**Analysis Timeline**

</div>

<div class="metadata-grid">

<div class="metadata-item">
<div class="metadata-label">Analysis Started</div>
<div class="metadata-value">${this.formatDate(startTime)}</div>
</div>

<div class="metadata-item">
<div class="metadata-label">Analysis Completed</div>
<div class="metadata-value">${this.formatDate(endTime)}</div>
</div>

<div class="metadata-item">
<div class="metadata-label">Total Duration</div>
<div class="metadata-value">${this.formatDuration(durationSeconds)}</div>
</div>

<div class="metadata-item">
<div class="metadata-label">Status</div>
<div class="metadata-value">${analysisRun?.status || 'UNKNOWN'}</div>
</div>

</div>

### Frame-by-Frame Analysis

<div class="info-box">

📊 **Visualization charts and frame-level predictions will be added in a future update.**

</div>

<div class="page-break"></div>`;
    }

    generateTechnicalMetadata(data) {
        return `## Technical Metadata

**Run ID:** \`${data.latestAnalysisRun?.id || 'N/A'}\`

**Total Duration:** ${this.formatDuration((data.latestAnalysisRun?.duration || 0) / 1000)}

**Status:** ${data.latestAnalysisRun?.status || 'UNKNOWN'}

### Processing Pipeline

| Stage | Status | Details |
|-------|--------|---------|
| Model Execution | ${data.completedAnalyses.length === data.totalModels ? '✓ Complete' : '⚠ Partial'} | ${data.completedAnalyses.length}/${data.totalModels} completed |
| Failed Analyses | ${data.failedAnalyses.length === 0 ? '✓ None' : `⚠ ${data.failedAnalyses.length}`} | ${data.totalRetries} retries |

<div class="page-break"></div>`;
    }

    generateMethodology() {
        return `## Methodology

The VidVigilante system employs a multi-stage detection pipeline combining spatial, temporal, and audio analysis using state-of-the-art deep learning models.

<div class="page-break"></div>`;
    }

    generateRecommendations(data) {
        const isFake = data.fakeDetections > data.realDetections;
        const highConfidence = data.avgConfidence > 0.85;
        
        if (isFake && highConfidence) {
            return `## Recommendations

<div class="danger-box">

**High Risk:** Do not use this media in official communications without additional verification.

</div>

<div class="page-break"></div>`;
        }
        
        return `## Recommendations

<div class="info-box">

**Assessment:** The media appears authentic. Always verify source and context.

</div>

<div class="page-break"></div>`;
    }

    generateFooter(data) {
        return `---

<div class="page-footer">

**Confidential Analysis Report** | Generated by VidVigilante v2.0.0 | ${this.formatDate(new Date())}

Report ID: ${data.reportId}

</div>`;
    }
}

export const pdfService = new PDFService();
