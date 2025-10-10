/**
 * User-friendly status messages and helpers
 * Converts technical statuses to human-readable text
 */

// ============================================
// STATUS MESSAGE MAPPING
// ============================================

export const statusMessages = {
  // Media/Video status
  UPLOADED: 'Ready to analyze',
  PROCESSING: 'Analyzing your video',
  QUEUED: 'Waiting in queue',
  COMPLETED: 'Analysis complete',
  FAILED: 'Analysis failed',
  PARTIALLY_ANALYZED: 'Partially analyzed',
  
  // Model status
  MODEL_PROCESSING: 'Processing',
  MODEL_COMPLETED: 'Complete',
  MODEL_FAILED: 'Failed',
  MODEL_QUEUED: 'Waiting',
};

/**
 * Get user-friendly status message
 * @param {string} status - Technical status from API
 * @param {object} context - Additional context (modelName, queuePosition, etc.)
 * @returns {string} Human-readable message
 */
export function getStatusMessage(status, context = {}) {
  switch (status?.toUpperCase()) {
    case 'UPLOADED':
      return 'Ready to analyze';
    
    case 'PROCESSING':
      if (context.modelName) {
        return `Processing with ${context.modelName}`;
      }
      return 'Analyzing your video';
    
    case 'QUEUED':
      if (context.queuePosition && context.queuePosition > 0) {
        return `Waiting in queue (${context.queuePosition} ahead)`;
      }
      return 'Waiting to start';
    
    case 'COMPLETED':
      if (context.duration) {
        return `Complete (${context.duration}s)`;
      }
      return 'Analysis complete';
    
    case 'FAILED':
      return 'Analysis failed';
    
    case 'PARTIALLY_ANALYZED':
      return 'Some models completed';
    
    default:
      return status || 'Unknown status';
  }
}

/**
 * Get progress description based on stage
 * @param {string} stage - Current processing stage
 * @param {number} progress - Progress percentage
 * @returns {string} Description
 */
export function getProgressDescription(stage, progress = 0) {
  const progressPercent = Math.round(progress);
  
  switch (stage) {
    case 'UPLOADING':
      return `Uploading... ${progressPercent}%`;
    
    case 'FRAME_EXTRACTION':
      return `Extracting frames... ${progressPercent}%`;
    
    case 'FRAME_ANALYSIS':
      return `Analyzing frames... ${progressPercent}%`;
    
    case 'WINDOW_PROCESSING':
      return `Processing windows... ${progressPercent}%`;
    
    case 'FINALIZING':
      return 'Finalizing results...';
    
    default:
      return `Processing... ${progressPercent}%`;
  }
}

/**
 * Get error message for user display
 * @param {object} error - Error object from API
 * @returns {object} { title, message, action }
 */
export function getErrorMessage(error) {
  const errorCode = error?.status || error?.code;
  const errorMessage = error?.message || '';
  
  // Network errors
  if (!errorCode || errorCode === 0) {
    return {
      title: 'Connection Error',
      message: 'Unable to connect to the server. Please check your internet connection.',
      action: 'Retry',
    };
  }
  
  // Client errors (4xx)
  if (errorCode >= 400 && errorCode < 500) {
    switch (errorCode) {
      case 400:
        return {
          title: 'Invalid Request',
          message: errorMessage || 'The file you uploaded is not supported. Please try a different file.',
          action: 'Try Again',
        };
      
      case 401:
      case 403:
        return {
          title: 'Authentication Required',
          message: 'Please sign in to continue.',
          action: 'Sign In',
        };
      
      case 404:
        return {
          title: 'Not Found',
          message: 'The requested resource could not be found.',
          action: 'Go Back',
        };
      
      case 413:
        return {
          title: 'File Too Large',
          message: 'The file you uploaded exceeds the maximum size limit. Please try a smaller file.',
          action: 'Choose Another File',
        };
      
      default:
        return {
          title: 'Request Error',
          message: errorMessage || 'Something went wrong with your request.',
          action: 'Try Again',
        };
    }
  }
  
  // Server errors (5xx)
  if (errorCode >= 500) {
    return {
      title: 'Server Error',
      message: 'Our servers are experiencing issues. Please try again later.',
      action: 'Retry',
    };
  }
  
  // Analysis-specific errors
  if (errorMessage.includes('unsupported format')) {
    return {
      title: 'Unsupported Format',
      message: 'This video format is not supported. Please use MP4, AVI, or MOV.',
      action: 'Choose Another File',
    };
  }
  
  if (errorMessage.includes('corrupted') || errorMessage.includes('invalid')) {
    return {
      title: 'Invalid File',
      message: 'The file appears to be corrupted or invalid. Please try a different file.',
      action: 'Choose Another File',
    };
  }
  
  // Default error
  return {
    title: 'Something Went Wrong',
    message: errorMessage || 'An unexpected error occurred. Please try again.',
    action: 'Retry',
  };
}

/**
 * Format duration in seconds to human-readable format
 * @param {number} seconds - Duration in seconds
 * @returns {string} Formatted duration
 */
export function formatDuration(seconds) {
  if (!seconds || seconds < 1) {
    return 'less than a second';
  }
  
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  }
  
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  
  if (minutes < 60) {
    if (remainingSeconds === 0) {
      return `${minutes}m`;
    }
    return `${minutes}m ${remainingSeconds}s`;
  }
  
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  
  if (remainingMinutes === 0) {
    return `${hours}h`;
  }
  return `${hours}h ${remainingMinutes}m`;
}

/**
 * Estimate remaining time based on progress
 * @param {number} startTime - Start timestamp (ms)
 * @param {number} currentProgress - Current progress (0-100)
 * @returns {string} Estimated time remaining
 */
export function estimateTimeRemaining(startTime, currentProgress) {
  if (!startTime || currentProgress <= 0) {
    return 'Calculating...';
  }
  
  if (currentProgress >= 100) {
    return 'Almost done';
  }
  
  const elapsed = Date.now() - startTime;
  const estimatedTotal = (elapsed / currentProgress) * 100;
  const remaining = estimatedTotal - elapsed;
  
  return formatDuration(remaining / 1000);
}

/**
 * Get color class for status
 * @param {string} status - Status string
 * @returns {string} Tailwind color classes
 */
export function getStatusColor(status) {
  switch (status?.toUpperCase()) {
    case 'COMPLETED':
      return 'text-green-500';
    
    case 'PROCESSING':
      return 'text-primary-main';
    
    case 'QUEUED':
      return 'text-light-tertiary dark:text-dark-tertiary';
    
    case 'FAILED':
      return 'text-red-500';
    
    case 'PARTIALLY_ANALYZED':
      return 'text-yellow-500';
    
    default:
      return 'text-light-muted-text dark:text-dark-muted-text';
  }
}

/**
 * Get icon name for status (for Lucide icons)
 * @param {string} status - Status string
 * @returns {string} Icon name
 */
export function getStatusIcon(status) {
  switch (status?.toUpperCase()) {
    case 'COMPLETED':
      return 'CheckCircle2';
    
    case 'PROCESSING':
      return 'Loader2';
    
    case 'QUEUED':
      return 'Clock';
    
    case 'FAILED':
      return 'XCircle';
    
    case 'PARTIALLY_ANALYZED':
      return 'AlertCircle';
    
    case 'UPLOADED':
      return 'Upload';
    
    default:
      return 'Circle';
  }
}

/**
 * Format file size to human-readable format
 * @param {number} bytes - File size in bytes
 * @returns {string} Formatted size
 */
export function formatFileSize(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const k = 1024;
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${units[i]}`;
}

/**
 * Format timestamp to relative time
 * @param {string|Date} timestamp - Timestamp
 * @returns {string} Relative time (e.g., "2 hours ago")
 */
export function formatRelativeTime(timestamp) {
  if (!timestamp) return '';
  
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now - date;
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHour / 24);
  
  if (diffSec < 60) {
    return 'just now';
  } else if (diffMin < 60) {
    return `${diffMin} minute${diffMin > 1 ? 's' : ''} ago`;
  } else if (diffHour < 24) {
    return `${diffHour} hour${diffHour > 1 ? 's' : ''} ago`;
  } else if (diffDay < 7) {
    return `${diffDay} day${diffDay > 1 ? 's' : ''} ago`;
  } else {
    return date.toLocaleDateString();
  }
}

export default {
  statusMessages,
  getStatusMessage,
  getProgressDescription,
  getErrorMessage,
  formatDuration,
  estimateTimeRemaining,
  getStatusColor,
  getStatusIcon,
  formatFileSize,
  formatRelativeTime,
};
