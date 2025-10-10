/**
 * Enhanced hook to manage ProgressPanel display
 * Converts useAnalysisProgress data into ProgressPanel-compatible format
 */

import { useMemo } from 'react';
import { useAnalysisProgress } from './useAnalysisProgress';

export function useProgressPanel(mediaId, filename) {
  const analysisProgress = useAnalysisProgress(mediaId, filename);

  // Transform model progress data into ProgressPanel format
  const models = useMemo(() => {
    const modelProgressData = analysisProgress.modelProgress || {};
    
    return Object.values(modelProgressData)
      .filter((model) => {
        // Filter out non-model entries
        const modelName = model.modelName || '';
        return modelName && 
               !modelName.toLowerCase().includes('worker') && 
               !modelName.toLowerCase().includes('backend') &&
               modelName.trim() !== '';
      })
      .map((model) => {
        
        // Map phase to status
        let status = 'QUEUED';
        if (model.phase === 'queued') {
          status = 'QUEUED';
        } else if (model.phase === 'completed' || model.phase === 'complete') {
          status = 'COMPLETED';
        } else if (model.phase === 'failed') {
          status = 'FAILED';
        } else if (model.phase === 'analyzing' || model.phase === 'processing') {
          status = 'PROCESSING';
        }

        return {
          id: model.modelName,
          name: model.modelName,
          status,
          progress: model.progress || 0,
          total: model.total || 100,
          stage: model.details?.stage,
          processingTime: model.details?.processingTime,
        };
      });
  }, [analysisProgress.modelProgress]);

  // Create media object for ProgressPanel
  const media = useMemo(() => {
    if (!mediaId) return null;
    
    return {
      id: mediaId,
      filename: filename || 'Video file',
    };
  }, [mediaId, filename]);

  return {
    // ProgressPanel props
    media,
    models,
    isVisible: analysisProgress.isProgressVisible,
    onClose: analysisProgress.hideProgress,
    
    // Original analysis progress data (for backward compatibility)
    ...analysisProgress,
  };
}

export default useProgressPanel;
