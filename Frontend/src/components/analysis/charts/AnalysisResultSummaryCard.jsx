// src/components/analysis/charts/AnalysisResultSummaryCard.jsx

import React from "react";
import { Link } from "react-router-dom";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from "../../ui/Card";
import { Button } from "../../ui/Button";
import { Badge } from "../../ui/Badge";
import { Alert, AlertDescription, AlertTitle } from "../../ui/Alert";
import {
  Eye,
  ShieldCheck,
  ShieldAlert,
  Clock,
  AlertTriangle,
  Brain,
  FileVideo,
  FileAudio,
  FileImage,
  ChevronRight,
} from "lucide-react";
import { formatProcessingTime } from "../../../utils/formatters.js";

export const AnalysisResultSummaryCard = ({ analysis, mediaId }) => {
  // Card for a FAILED analysis
  if (analysis.status === "FAILED") {
    const retryCount = analysis.resultPayload?.retryCount || 0;
    
    return (
      <Card className="border-0 shadow-none">
        <CardContent className="p-2">
          <div className="flex items-start gap-4">
            <div className="flex-shrink-0 w-12 h-12 rounded-lg bg-red-100 dark:bg-red-900/20 flex items-center justify-center">
              <AlertTriangle className="h-6 w-6 text-red-600 dark:text-red-400" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <h4 className="font-semibold text-base truncate">{analysis.modelName}</h4>
                {retryCount > 0 && (
                  <Badge variant="secondary" size="sm" className="text-xs">
                    {retryCount} {retryCount === 1 ? 'retry' : 'retries'}
                  </Badge>
                )}
              </div>
              <p className="text-xs text-light-muted-text dark:text-dark-muted-text mb-2">
                {analysis.errorMessage}
              </p>
              
            </div>
            {mediaId && analysis.id && (
              <div className="flex-shrink-0">
                <Button asChild variant="outline" size="sm">
                  <Link to={`/results/${mediaId}/${analysis.id}`}>
                    <Eye className="h-4 w-4 mr-2" /> Details
                    <ChevronRight className="h-4 w-4 ml-1" />
                  </Link>
                </Button>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    );
  }

  // Card for a COMPLETED analysis
  const resultPayload = analysis.resultPayload;

  const prediction = resultPayload?.prediction || analysis.prediction || "N/A";
  const confidence = resultPayload?.confidence || analysis.confidence || 0;
  const processingTime =
    resultPayload?.processing_time || resultPayload?.processingTime || null;
  const mediaType = resultPayload?.media_type || "N/A";

  const isReal = prediction === "REAL";

  // Media type icon mapping
  const mediaTypeIcons = {
    video: FileVideo,
    audio: FileAudio,
    image: FileImage,
  };
  
  const MediaIcon = mediaTypeIcons[mediaType?.toLowerCase()] || FileVideo;

  return (
    <Card
      className={`transition-all shadow-none border-0`}
    >
      <CardContent className="p-2">
        <div className="flex items-center gap-4">
          {/* Icon Section */}
          <div 
            className={`flex-shrink-0 w-12 h-12 rounded-lg flex items-center justify-center ${
              isReal 
                ? "bg-green-100 dark:bg-green-900/20" 
                : "bg-red-100 dark:bg-red-900/20"
            }`}
          >
            {isReal ? (
              <ShieldCheck className="w-6 h-6 text-green-600 dark:text-green-400" />
            ) : (
              <ShieldAlert className="w-6 h-6 text-red-600 dark:text-red-400" />
            )}
          </div>

          {/* Model Info Section */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <h4 className="font-semibold text-base truncate">{analysis.modelName}</h4>
              {/* <Badge 
                variant={
                  mediaType?.toLowerCase() === "video" ? "purple" : 
                  mediaType?.toLowerCase() === "audio" ? "info" : 
                  "success"
                }
                size="sm"
              >
                <MediaIcon className="w-3 h-3 mr-1" />
                {mediaType}
              </Badge> */}
            </div>
          </div>

          {/* Results Section */}
          <div className="flex items-center gap-6">
            <div className="text-center">
              <p
                className={`text-xl font-bold ${
                  isReal ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"
                }`}
              >
                {prediction}
              </p>
              <p className="text-xs text-light-muted-text dark:text-dark-muted-text mt-1">
                Prediction
              </p>
            </div>
            <div className="text-center">
              <p className="text-xl font-bold">
                {(confidence * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-light-muted-text dark:text-dark-muted-text mt-1">
                Confidence
              </p>
            </div>
            <div className="text-center hidden sm:block">
              <p className="text-xl font-semibold">
                {formatProcessingTime(processingTime)}
              </p>
              <p className="text-xs text-light-muted-text dark:text-dark-muted-text mt-1 flex items-center justify-center gap-1">
                <Clock className="w-3 h-3" />
                Processing
              </p>
            </div>
          </div>

          {/* Action Button */}
          <div className="flex-shrink-0">
            <Button asChild variant="outline" size="sm">
              <Link to={`/results/${mediaId}/${analysis.id}`}>
                <Brain className="h-4 w-4 mr-2" /> View Report
                <ChevronRight className="h-4 w-4 ml-1" />
              </Link>
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};