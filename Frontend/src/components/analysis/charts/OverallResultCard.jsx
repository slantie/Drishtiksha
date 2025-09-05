// src/components/analysis/charts/OverallResultCard.jsx

import React from "react";
import { Card } from "../../ui/Card";
import { CheckCircle, AlertCircle } from "lucide-react";

export const OverallResultCard = ({ result }) => {
  // `result` here is expected to be an object like: { prediction: "REAL"|"FAKE", confidence: 0.X }
  if (
    !result ||
    typeof result.prediction === "undefined" ||
    typeof result.confidence === "undefined"
  ) {
    return (
      <Card className="text-center p-6 border-yellow-500/30">
        <AlertCircle className="h-8 w-8 text-yellow-600 mx-auto mb-4" />
        <h2 className="text-xl font-bold mb-2">No Overall Result</h2>
        <p className="text-sm text-light-muted-text dark:text-dark-muted-text">
          Prediction or confidence score not available.
        </p>
      </Card>
    );
  }

  const isReal = result.prediction === "REAL";
  const confidence = result.confidence * 100; // Convert to percentage

  return (
    <Card
      className={`border-2 ${
        isReal ? "border-green-500/30" : "border-red-500/30"
      }`}
    >
      <div className="p-6 text-center">
        <div
          className={`inline-flex items-center justify-center w-16 h-16 rounded-full mb-4 ${
            isReal
              ? "bg-green-100 dark:bg-green-900/30"
              : "bg-red-100 dark:bg-red-900/30"
          }`}
        >
          {isReal ? (
            <CheckCircle className="h-8 w-8 text-green-600" />
          ) : (
            <AlertCircle className="h-8 w-8 text-red-600" />
          )}
        </div>

        <h2
          className={`text-4xl font-bold mb-2 ${
            isReal ? "text-green-600" : "text-red-600"
          }`}
        >
          {confidence.toFixed(1)}%
        </h2>

        <p className="text-xl font-semibold mb-1">
          {isReal ? "Authentic Content" : "Potential Deepfake"}
        </p>
        <p className="text-sm text-light-muted-text dark:text-dark-muted-text">
          Confidence Score
        </p>
      </div>
    </Card>
  );
};
