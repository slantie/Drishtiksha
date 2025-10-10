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
} from "lucide-react";
import { formatProcessingTime } from "../../../utils/formatters.js";

export const AnalysisResultSummaryCard = ({ analysis, mediaId }) => {
  // `analysis` is the DeepfakeAnalysis object from the backend
  // It has properties like `id`, `modelName`, `prediction`, `confidence`, `status`, `errorMessage`, `resultPayload`.

  // Card for a FAILED analysis
  if (analysis.status === "FAILED") {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{analysis.modelName}</CardTitle>
          <CardDescription>An AI deepfake detection model.</CardDescription>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Analysis Failed</AlertTitle>
            <AlertDescription>
              {analysis.errorMessage ||
                "An unknown error occurred during analysis."}
            </AlertDescription>
          </Alert>
        </CardContent>
        {/* Potentially link to a detailed error report if one exists */}
        {mediaId && analysis.id && (
          <CardFooter>
            <Button asChild variant="outline" className="w-full">
              <Link to={`/results/${mediaId}/${analysis.id}`}>
                <Eye className="mr-2 h-4 w-4" /> View Error Details
              </Link>
            </Button>
          </CardFooter>
        )}
      </Card>
    );
  }

  // Card for a COMPLETED analysis
  const resultPayload = analysis.resultPayload;

  // Safely extract properties from resultPayload
  const prediction = resultPayload?.prediction || analysis.prediction || "N/A"; // Fallback to top-level if needed
  const confidence = resultPayload?.confidence || analysis.confidence || 0; // Fallback to top-level if needed
  const processingTime =
    resultPayload?.processing_time || resultPayload?.processingTime || null; // Backend might use different keys
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
      className={`transition-all hover:shadow-lg ${
        isReal ? "border-green-500/30" : "border-red-500/30"
      }`}
    >
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2">
              <CardTitle className="text-base truncate">{analysis.modelName}</CardTitle>
              <Badge 
                variant={
                  mediaType?.toLowerCase() === "video" ? "purple" : 
                  mediaType?.toLowerCase() === "audio" ? "info" : 
                  "success"
                }
                size="sm"
              >
                <MediaIcon className="w-3 h-3 mr-1" />
                {mediaType}
              </Badge>
            </div>
            <CardDescription className="text-xs">
              AI deepfake detection model
            </CardDescription>
          </div>
          {isReal ? (
            <ShieldCheck className="w-8 h-8 text-green-500 flex-shrink-0" />
          ) : (
            <ShieldAlert className="w-8 h-8 text-red-500 flex-shrink-0" />
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4 text-center">
          <div>
            <p
              className={`text-3xl font-bold ${
                isReal ? "text-green-600" : "text-red-600"
              }`}
            >
              {prediction}
            </p>
            <p className="text-xs text-light-muted-text dark:text-dark-muted-text">
              Prediction
            </p>
          </div>
          <div>
            <p className="text-3xl font-bold">
              {(confidence * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-light-muted-text dark:text-dark-muted-text">
              Confidence
            </p>
          </div>
        </div>
        <div className="text-xs text-light-muted-text dark:text-dark-muted-text pt-2 flex justify-between">
          <span className="flex items-center gap-2">
            <Clock className="inline w-3 h-3" />
            Processing Time
          </span>
          <span className="font-mono">
            {formatProcessingTime(processingTime)}
          </span>
        </div>
      </CardContent>
      <CardFooter>
        <Button asChild variant="outline" className="w-full">
          <Link to={`/results/${mediaId}/${analysis.id}`}>
            <Brain className="mr-2 h-4 w-4" /> View Detailed Report
          </Link>
        </Button>
      </CardFooter>
    </Card>
  );
};
