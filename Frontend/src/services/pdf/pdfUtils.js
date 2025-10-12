// src/services/pdf/pdfUtils.js
// Utility functions for PDF data processing and formatting

import { formatBytes, formatDate, formatProcessingTime } from '../../utils/formatters.js';

/**
 * Prepares and enriches media data for PDF generation
 * @param {Object} media - Media object from API
 * @param {Object} user - User object
 * @returns {Object} Enriched data object ready for PDF rendering
 */
export const prepareReportData = (media, user) => {
  // Get the latest analysis run (should be first in array)
  const latestAnalysisRun = media.analysisRuns?.[0];
  
  // Helper function to normalize analysis object
  const normalizeAnalysis = (analysis) => {
    return {
      ...analysis,
      // Ensure these fields exist at root level
      confidence: analysis.confidence ?? analysis.resultPayload?.confidence ?? 0,
      prediction: analysis.prediction ?? analysis.resultPayload?.prediction,
      retryCount: analysis.retryCount ?? analysis.resultPayload?.retryCount ?? 0,
      processingTime: analysis.processingTime ?? analysis.resultPayload?.processingTime,
      errorType: analysis.errorType ?? analysis.resultPayload?.errorType,
      framePredictions: analysis.framePredictions ?? analysis.resultPayload?.frame_predictions ?? [],
      modelType: analysis.modelType ?? analysis.resultPayload?.model_type,
      analysisTechnique: analysis.analysisTechnique ?? analysis.resultPayload?.analysis_technique,
      analyzedAt: analysis.analyzedAt ?? analysis.createdAt ?? analysis.updatedAt,
    };
  };
  
  // Separate completed and failed analyses and normalize them
  const completedAnalyses = (latestAnalysisRun?.analyses?.filter(
    (a) => a.status === 'COMPLETED'
  ) || []).map(normalizeAnalysis);
  
  const failedAnalyses = (latestAnalysisRun?.analyses?.filter(
    (a) => a.status === 'FAILED'
  ) || []).map(normalizeAnalysis);
  
  // Calculate detection statistics
  const realDetections = completedAnalyses.filter(
    (a) => a.prediction === 'REAL'
  ).length;
  
  const fakeDetections = completedAnalyses.filter(
    (a) => a.prediction === 'FAKE'
  ).length;
  
  const totalModels = completedAnalyses.length + failedAnalyses.length;
  
  // Calculate average confidence
  const avgConfidence = completedAnalyses.length > 0
    ? completedAnalyses.reduce((sum, a) => sum + (a.confidence || 0), 0) / completedAnalyses.length
    : 0;
  
  // Determine overall assessment
  const assessment = determineOverallAssessment(
    realDetections,
    fakeDetections,
    totalModels,
    media.status,
    latestAnalysisRun?.status
  );
  
  // Calculate retry statistics
  const totalRetries = failedAnalyses.reduce(
    (sum, a) => sum + (a.retryCount || 0), 0
  );
  
  // Sort analyses by confidence (descending) for display
  const sortedAnalyses = [...completedAnalyses].sort(
    (a, b) => (b.confidence || 0) - (a.confidence || 0)
  );
  
  return {
    // Original media data
    ...media,
    
    // User information
    user: user || null,
    
    // Analysis run data
    latestAnalysisRun,
    completedAnalyses: sortedAnalyses,
    failedAnalyses,
    
    // Statistics
    totalModels,
    realDetections,
    fakeDetections,
    avgConfidence,
    totalRetries,
    
    // Assessment
    overallAssessment: assessment.label,
    assessmentColor: assessment.color,
    riskLevel: assessment.risk,
    
    // Formatted data
    formattedSize: formatBytes(media.size),
    formattedUploadDate: formatDate(media.createdAt),
    formattedReportDate: formatDate(new Date()),
  };
};

/**
 * Determines overall assessment based on analysis results
 */
const determineOverallAssessment = (realCount, fakeCount, total, mediaStatus, runStatus) => {
  // Handle edge cases
  if (total === 0) {
    if (mediaStatus === 'FAILED' || runStatus === 'FAILED') {
      return {
        label: 'Analysis Failed',
        color: '#dc2626',
        risk: 'UNKNOWN'
      };
    }
    return {
      label: 'Pending Analysis',
      color: '#9ca3af',
      risk: 'UNKNOWN'
    };
  }
  
  // Calculate percentages
  const realPercent = realCount / total;
  const fakePercent = fakeCount / total;
  
  // Determine assessment based on consensus
  if (fakePercent >= 0.7) {
    return {
      label: 'Highly Likely Deepfake',
      color: '#dc2626',
      risk: 'CRITICAL'
    };
  } else if (fakePercent >= 0.5) {
    return {
      label: 'Likely Deepfake',
      color: '#ef4444',
      risk: 'HIGH'
    };
  } else if (fakePercent >= 0.3) {
    return {
      label: 'Possibly Manipulated',
      color: '#f59e0b',
      risk: 'MODERATE'
    };
  } else if (realPercent >= 0.7) {
    return {
      label: 'Highly Likely Authentic',
      color: '#16a34a',
      risk: 'LOW'
    };
  } else if (realPercent >= 0.5) {
    return {
      label: 'Likely Authentic',
      color: '#22c55e',
      risk: 'LOW'
    };
  } else {
    return {
      label: 'Inconclusive',
      color: '#9ca3af',
      risk: 'UNCERTAIN'
    };
  }
};

/**
 * Extracts temporal data from analyses for charting
 * @param {Array} analyses - Array of analysis objects
 * @returns {Object} Temporal data organized by model
 */
export const extractTemporalData = (analyses) => {
  const temporalData = {};
  
  analyses.forEach(analysis => {
    const framePredictions = analysis.resultPayload?.frame_predictions || [];
    
    if (framePredictions.length > 0) {
      temporalData[analysis.modelName] = {
        modelName: analysis.modelName,
        predictions: framePredictions.map(fp => ({
          index: fp.index,
          score: fp.score,
          prediction: fp.prediction
        })),
        metrics: analysis.resultPayload?.metrics || {}
      };
    }
  });
  
  return temporalData;
};

/**
 * Calculates confidence distribution statistics
 * @param {Array} analyses - Array of completed analysis objects
 * @returns {Object} Distribution statistics
 */
export const calculateConfidenceDistribution = (analyses) => {
  if (analyses.length === 0) {
    return {
      min: 0,
      max: 0,
      mean: 0,
      median: 0,
      stdDev: 0,
      bins: []
    };
  }
  
  const confidences = analyses.map(a => a.confidence || 0).sort((a, b) => a - b);
  
  // Calculate statistics
  const min = confidences[0];
  const max = confidences[confidences.length - 1];
  const mean = confidences.reduce((sum, c) => sum + c, 0) / confidences.length;
  const median = confidences[Math.floor(confidences.length / 2)];
  
  // Calculate standard deviation
  const variance = confidences.reduce((sum, c) => sum + Math.pow(c - mean, 2), 0) / confidences.length;
  const stdDev = Math.sqrt(variance);
  
  // Create histogram bins (0-50%, 50-60%, 60-70%, 70-80%, 80-90%, 90-100%)
  const bins = [
    { range: '0-50%', count: 0, color: '#dc2626' },
    { range: '50-60%', count: 0, color: '#f59e0b' },
    { range: '60-70%', count: 0, color: '#f59e0b' },
    { range: '70-80%', count: 0, color: '#22c55e' },
    { range: '80-90%', count: 0, color: '#22c55e' },
    { range: '90-100%', count: 0, color: '#16a34a' },
  ];
  
  confidences.forEach(c => {
    if (c < 0.5) bins[0].count++;
    else if (c < 0.6) bins[1].count++;
    else if (c < 0.7) bins[2].count++;
    else if (c < 0.8) bins[3].count++;
    else if (c < 0.9) bins[4].count++;
    else bins[5].count++;
  });
  
  return { min, max, mean, median, stdDev, bins };
};

/**
 * Calculates model agreement metrics
 * @param {Array} analyses - Array of completed analysis objects
 * @returns {Object} Agreement statistics
 */
export const calculateModelAgreement = (analyses) => {
  if (analyses.length === 0) {
    return {
      consensusPercent: 0,
      majorityPrediction: null,
      disagreementCount: 0
    };
  }
  
  const realCount = analyses.filter(a => a.prediction === 'REAL').length;
  const fakeCount = analyses.filter(a => a.prediction === 'FAKE').length;
  
  const majorityPrediction = realCount > fakeCount ? 'REAL' : 'FAKE';
  const majorityCount = Math.max(realCount, fakeCount);
  const consensusPercent = (majorityCount / analyses.length) * 100;
  const disagreementCount = Math.min(realCount, fakeCount);
  
  return {
    consensusPercent,
    majorityPrediction,
    disagreementCount,
    realCount,
    fakeCount
  };
};

/**
 * Groups analyses by media type (video, audio, image)
 * @param {Array} analyses - Array of analysis objects
 * @returns {Object} Analyses grouped by media type
 */
export const groupAnalysesByMediaType = (analyses) => {
  const grouped = {
    video: [],
    audio: [],
    image: [],
    unknown: []
  };
  
  analyses.forEach(analysis => {
    const mediaType = analysis.mediaType?.toLowerCase() || 
                     analysis.resultPayload?.media_type?.toLowerCase() ||
                     'unknown';
    
    if (grouped[mediaType]) {
      grouped[mediaType].push(analysis);
    } else {
      grouped.unknown.push(analysis);
    }
  });
  
  return grouped;
};

/**
 * Detects temporal anomalies in frame predictions
 * @param {Object} temporalData - Temporal data from extractTemporalData
 * @returns {Array} Array of detected anomalies
 */
export const detectTemporalAnomalies = (temporalData) => {
  const anomalies = [];
  
  Object.entries(temporalData).forEach(([modelName, data]) => {
    const predictions = data.predictions;
    
    if (predictions.length < 10) return; // Need sufficient data
    
    // Calculate rolling average with window of 5
    const windowSize = 5;
    for (let i = windowSize; i < predictions.length; i++) {
      const window = predictions.slice(i - windowSize, i);
      const avgScore = window.reduce((sum, p) => sum + p.score, 0) / windowSize;
      const currentScore = predictions[i].score;
      
      // Detect sudden jumps (> 0.3 difference)
      if (Math.abs(currentScore - avgScore) > 0.3) {
        anomalies.push({
          modelName,
          frameIndex: predictions[i].index,
          type: 'sudden_change',
          severity: Math.abs(currentScore - avgScore),
          description: `Sudden confidence change detected at frame ${predictions[i].index}`
        });
      }
    }
    
    // Detect oscillations (frequent prediction changes)
    let predictionChanges = 0;
    for (let i = 1; i < predictions.length; i++) {
      if (predictions[i].prediction !== predictions[i - 1].prediction) {
        predictionChanges++;
      }
    }
    
    const changeRate = predictionChanges / predictions.length;
    if (changeRate > 0.2) { // More than 20% of frames have prediction changes
      anomalies.push({
        modelName,
        type: 'high_oscillation',
        severity: changeRate,
        description: `High prediction oscillation detected (${(changeRate * 100).toFixed(1)}% frame changes)`
      });
    }
  });
  
  return anomalies;
};

/**
 * Generates table of contents with page numbers
 * @param {Object} data - Prepared report data
 * @returns {Array} Table of contents entries
 */
export const generateTableOfContents = (data) => {
  const toc = [
    { title: 'Executive Summary', page: 3 },
    { title: 'Media Information', page: 4 },
    { title: 'Model Comparison Matrix', page: 5 },
  ];
  
  let currentPage = 6;
  
  // Add detailed model analysis pages
  if (data.completedAnalyses.length > 0 || data.failedAnalyses.length > 0) {
    toc.push({ title: 'Detailed Model Analysis', page: currentPage, isSection: true });
    
    const allAnalyses = [...data.completedAnalyses, ...data.failedAnalyses];
    allAnalyses.forEach((analysis, index) => {
      toc.push({
        title: `${analysis.modelName}`,
        page: currentPage + index,
        isSubItem: true
      });
    });
    
    currentPage += allAnalyses.length;
  }
  
  // Add temporal analysis if available
  const hasTemporalData = data.completedAnalyses.some(
    a => a.resultPayload?.frame_predictions?.length > 0
  );
  
  if (hasTemporalData) {
    toc.push({ title: 'Temporal Analysis', page: currentPage });
    currentPage++;
  }
  
  // Add technical metadata
  toc.push({ title: 'Technical Metadata', page: currentPage });
  currentPage++;
  
  // Add methodology
  toc.push({ title: 'Methodology & Disclaimer', page: currentPage });
  
  return toc;
};

/**
 * Calculates total number of pages in the report
 * @param {Object} data - Prepared report data
 * @returns {number} Total page count
 */
export const calculateTotalPages = (data) => {
  let pageCount = 5; // Cover + TOC + Executive + Media + Comparison
  
  // Add pages for detailed analyses
  const allAnalyses = (data.completedAnalyses?.length || 0) + (data.failedAnalyses?.length || 0);
  pageCount += allAnalyses;
  
  // Add temporal analysis page if data exists
  const hasTemporalData = data.completedAnalyses?.some(
    a => a.resultPayload?.frame_predictions?.length > 0
  );
  if (hasTemporalData) pageCount++;
  
  // Add metadata and methodology pages
  pageCount += 2;
  
  return pageCount;
};

/**
 * Formats duration in seconds to human-readable string
 * @param {number} seconds - Duration in seconds
 * @returns {string} Formatted duration
 */
export const formatDuration = (seconds) => {
  if (!seconds || seconds < 0) return 'N/A';
  
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  
  if (mins > 0) {
    return `${mins}m ${secs}s`;
  }
  return `${secs}s`;
};

/**
 * Truncates text to fit within a certain width (approximation)
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length
 * @returns {string} Truncated text
 */
export const truncateText = (text, maxLength = 50) => {
  if (!text) return '';
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - 3) + '...';
};

/**
 * Generates a unique report ID
 * @returns {string} Report ID
 */
export const generateReportId = () => {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 7);
  return `RPT-${timestamp}-${random}`.toUpperCase();
};
