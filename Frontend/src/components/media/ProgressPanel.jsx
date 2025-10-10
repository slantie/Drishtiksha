import { useState, useEffect } from 'react';
import { 
  Loader2, 
  CheckCircle2, 
  Clock, 
  XCircle, 
  ChevronDown, 
  ChevronUp,
  X 
} from 'lucide-react';
import { spacing, typography, colors, padding, icon, transition } from '../../styles/tokens';
import { 
  getStatusMessage, 
  getProgressDescription, 
  formatDuration, 
  estimateTimeRemaining 
} from '../../utils/statusHelpers';

/**
 * Unified Progress Panel - Single source of truth for analysis progress
 * Replaces multiple toast spam with clean, persistent progress display
 */
export default function ProgressPanel({ 
  media, 
  models = [], 
  onClose,
  className = '' 
}) {
  const [isMinimized, setIsMinimized] = useState(false);
  const [startTime] = useState(Date.now());

  // Calculate overall progress from model statuses
  const completedModels = models.filter(m => m.status === 'COMPLETED').length;
  const failedModels = models.filter(m => m.status === 'FAILED').length;
  const totalModels = models.length;
  
  // Overall progress calculation
  const overallProgress = totalModels > 0 
    ? Math.round((completedModels / totalModels) * 100) 
    : 0;

  // Check if all processing is done
  const allComplete = completedModels + failedModels === totalModels;
  const hasFailures = failedModels > 0;

  // Get current processing model
  const processingModel = models.find(m => m.status === 'PROCESSING');

  // Auto-close after 5 seconds if all complete
  useEffect(() => {
    if (allComplete) {
      const timer = setTimeout(() => {
        onClose?.();
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [allComplete, onClose]);

  if (!media) return null;

  return (
    <div 
      className={`
        fixed bottom-4 right-4 
        w-full max-w-md
        bg-light-background dark:bg-dark-muted-background 
        shadow-2xl rounded-lg overflow-hidden 
        border ${colors.border.default}
        z-50
        ${transition.normal}
        ${className}
      `}
      style={{ maxHeight: isMinimized ? '80px' : '400px' }}
    >
      {/* Header */}
      <div 
        className={`
          ${padding.md} 
          bg-gradient-to-r from-primary-main to-purple-600 
          text-white
        `}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3 flex-1 min-w-0">
            {allComplete ? (
              hasFailures ? (
                <XCircle className={`${icon.md} flex-shrink-0`} />
              ) : (
                <CheckCircle2 className={`${icon.md} flex-shrink-0`} />
              )
            ) : (
              <Loader2 className={`${icon.md} animate-spin flex-shrink-0`} />
            )}
            <div className="flex-1 min-w-0">
              <p className="font-semibold text-sm">
                {allComplete 
                  ? (hasFailures ? 'Analysis Completed with Errors' : 'Analysis Complete') 
                  : 'Analyzing Video'}
              </p>
              <p className="text-xs opacity-90 truncate">
                {media.filename || 'Video file'}
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsMinimized(!isMinimized)}
              className={`
                ${padding.xs} 
                hover:bg-white/20 
                rounded 
                ${transition.fast}
              `}
              aria-label={isMinimized ? 'Expand' : 'Minimize'}
            >
              {isMinimized ? (
                <ChevronUp className={icon.sm} />
              ) : (
                <ChevronDown className={icon.sm} />
              )}
            </button>
            
            {allComplete && (
              <button
                onClick={onClose}
                className={`
                  ${padding.xs} 
                  hover:bg-white/20 
                  rounded 
                  ${transition.fast}
                `}
                aria-label="Close"
              >
                <X className={icon.sm} />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Content */}
      {!isMinimized && (
        <div className={`${padding.md} ${spacing.md}`}>
          {/* Overall Progress */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className={`${typography.small} font-medium ${colors.text.primary}`}>
                Overall Progress
              </span>
              <span className={`${typography.small} ${colors.text.muted}`}>
                {overallProgress}%
              </span>
            </div>
            
            {/* Progress Bar */}
            <div className="w-full bg-light-secondary dark:bg-dark-secondary rounded-full h-2">
              <div 
                className={`
                  h-2 rounded-full ${transition.normal}
                  ${allComplete 
                    ? (hasFailures ? 'bg-yellow-500' : 'bg-green-500')
                    : 'bg-primary-main'
                  }
                `}
                style={{ width: `${overallProgress}%` }}
              />
            </div>
            
            {/* Time estimate */}
            {!allComplete && overallProgress > 0 && (
              <p className={`${typography.tiny} ${colors.text.noisy} mt-1`}>
                Est. time remaining: {estimateTimeRemaining(startTime, overallProgress)}
              </p>
            )}
            
            {allComplete && (
              <p className={`${typography.tiny} ${colors.text.noisy} mt-1`}>
                Completed in {formatDuration((Date.now() - startTime) / 1000)}
              </p>
            )}
          </div>

          {/* Model Status List */}
          {totalModels > 0 && (
            <div>
              <h4 className={`${typography.small} font-semibold ${colors.text.primary} mb-2`}>
                Model Analysis
              </h4>
              
              <div className={spacing.xs}>
                {models.map((model) => (
                  <ModelStatusRow key={model.id} model={model} />
                ))}
              </div>
            </div>
          )}

          {/* Current Activity */}
          {processingModel && (
            <div 
              className={`
                ${padding.sm} 
                bg-light-muted-background dark:bg-dark-background 
                rounded 
                ${colors.border.default} 
                border
              `}
            >
              <p className={`${typography.tiny} ${colors.text.muted}`}>
                {getProgressDescription(
                  processingModel.stage, 
                  processingModel.progress || 0
                )}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Individual Model Status Row
 */
function ModelStatusRow({ model }) {
  const { name, status, progress, processingTime } = model;
  
  // Get display name (remove technical prefixes)
  const displayName = name
    .replace(/-V\d+$/, '') // Remove version suffix
    .replace(/-/g, ' ')    // Replace hyphens with spaces
    .split(' ')
    .map(word => word.charAt(0) + word.slice(1).toLowerCase())
    .join(' ');

  const getIcon = () => {
    switch (status) {
      case 'COMPLETED':
        return <CheckCircle2 className={`${icon.sm} text-green-500 flex-shrink-0`} />;
      case 'PROCESSING':
        return <Loader2 className={`${icon.sm} text-primary-main animate-spin flex-shrink-0`} />;
      case 'FAILED':
        return <XCircle className={`${icon.sm} text-red-500 flex-shrink-0`} />;
      case 'QUEUED':
      default:
        return <Clock className={`${icon.sm} text-light-tertiary dark:text-dark-tertiary flex-shrink-0`} />;
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'COMPLETED':
        return processingTime ? formatDuration(processingTime) : 'Complete';
      case 'PROCESSING':
        return progress ? `${Math.round(progress)}%` : 'Processing';
      case 'FAILED':
        return 'Failed';
      case 'QUEUED':
      default:
        return 'Queued';
    }
  };

  return (
    <div className="flex items-center gap-3">
      {getIcon()}
      <span className={`flex-1 ${typography.small} ${colors.text.primary} truncate`}>
        {displayName}
      </span>
      <span className={`${typography.tiny} ${colors.text.muted} flex-shrink-0`}>
        {getStatusText()}
      </span>
    </div>
  );
}
