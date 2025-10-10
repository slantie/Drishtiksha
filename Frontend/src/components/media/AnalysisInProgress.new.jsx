// src/components/media/AnalysisInProgress.jsx - Redesigned with design tokens

import React from "react";
import { Loader2, ExternalLink } from "lucide-react";
import { spacing, typography, colors, padding, card, icon } from "../../styles/tokens";
import { useProgressPanel } from "../../hooks/useProgressPanel";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/Card.jsx";
import { Button } from "../ui/Button.jsx";
import ProgressPanel from "./ProgressPanel";

/**
 * Simplified Analysis In Progress component
 * Shows minimal inline status + ProgressPanel for details
 */
export function AnalysisInProgress({ 
  mediaId, 
  filename,
  showDetailed = false 
}) {
  const progressPanel = useProgressPanel(mediaId, filename);

  // Calculate completion
  const completedCount = progressPanel.models.filter(
    m => m.status === 'COMPLETED'
  ).length;
  const totalCount = progressPanel.models.length;
  const allComplete = completedCount === totalCount && totalCount > 0;

  if (allComplete) {
    return null; // Hide when complete
  }

  return (
    <>
      {/* Inline status card */}
      <Card className={`${card.base} ${spacing.md}`}>
        <CardHeader>
          <CardTitle className={`flex items-center ${spacing.xs}`}>
            <Loader2 className={`${icon.md} animate-spin text-primary-main`} />
            <span className={typography.h4}>
              Analysis in Progress
            </span>
          </CardTitle>
        </CardHeader>
        
        <CardContent className={spacing.md}>
          {/* Simple progress summary */}
          <div className={spacing.sm}>
            <p className={`${typography.body} ${colors.text.muted}`}>
              Analyzing with {totalCount} model{totalCount !== 1 ? 's' : ''}...
            </p>
            
            <div className="flex items-center justify-between">
              <span className={`${typography.small} ${colors.text.noisy}`}>
                {completedCount} of {totalCount} complete
              </span>
              <span className={`${typography.small} ${colors.text.noisy}`}>
                {totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 0}%
              </span>
            </div>

            {/* Simple progress bar */}
            <div className="w-full bg-light-secondary dark:bg-dark-secondary rounded-full h-2 mt-2">
              <div 
                className="bg-primary-main h-2 rounded-full transition-all duration-300"
                style={{ 
                  width: `${totalCount > 0 ? (completedCount / totalCount) * 100 : 0}%` 
                }}
              />
            </div>
          </div>

          {/* Show details button */}
          {!showDetailed && (
            <Button
              variant="outline"
              size="sm"
              onClick={progressPanel.showProgress}
              className={`w-full ${spacing.xs}`}
            >
              <ExternalLink className={icon.sm} />
              <span>View Detailed Progress</span>
            </Button>
          )}
        </CardContent>
      </Card>

      {/* Floating progress panel */}
      {progressPanel.isVisible && (
        <ProgressPanel
          media={progressPanel.media}
          models={progressPanel.models}
          onClose={progressPanel.onClose}
        />
      )}
    </>
  );
}

export default AnalysisInProgress;
