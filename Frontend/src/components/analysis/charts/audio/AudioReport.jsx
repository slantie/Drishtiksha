// src/components/analysis/charts/audio/AudioReport.jsx

import React from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "../../../ui/Card";
import { Music, Activity } from "lucide-react"; // Changed LineChartIcon to Music for audio context

// Import dedicated audio chart components
import { SpectrogramCard } from "./SpectrogramCard";
// import { AudioForensicsChart } from "./AudioForensicsChart"; // If you have a chart component, it would be used here

const InfoItem = ({ label, value }) => (
  <div className="flex justify-between items-center border-b border-light-secondary/50 dark:border-dark-secondary/50 py-2">
    <span className="text-light-muted-text dark:text-dark-muted-text">
      {label}
    </span>
    <span className="font-semibold font-mono">{value}</span>
  </div>
);

export const AudioReport = ({ result }) => {
  // `result` here is the analysis.resultPayload
  // Assuming backend's resultPayload for audio looks something like:
  // {
  //    "audio_analysis": {
  //        "pitch_info": { "mean_pitch_hz": 120.5, "pitch_stability_score": 0.8 },
  //        "energy_info": { "rms_energy": 0.05, "silence_ratio": 0.15 },
  //        "spectral_info": { "spectral_centroid": 1500, "spectral_contrast": 15 }
  //    },
  //    "visualization": { "spectrogram_url": "..." }
  // }
  const audioAnalysis = result.audio_analysis || {};
  const pitch = audioAnalysis.pitch_info || {};
  const energy = audioAnalysis.energy_info || {};
  const spectral = audioAnalysis.spectral_info || {};
  const visualization = result.visualization || {}; // Spectrogram URL directly from result.visualization

  if (
    Object.keys(audioAnalysis).length === 0 &&
    !visualization.spectrogram_url
  ) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Music className="h-5 w-5 text-primary-main" /> Audio Forensic
            Report
          </CardTitle>
          <CardDescription>
            Detailed metrics and visualizations for audio deepfake detection.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center p-8 text-light-muted-text dark:text-dark-muted-text">
            <Activity className="h-12 w-12 mx-auto mb-4" />
            <p className="text-lg font-semibold">No Detailed Audio Data</p>
            <p className="mt-2 text-sm">
              This model did not provide detailed audio forensics. This may be
              due to an incompatible media type (e.g., video without audio
              track), a processing error, or the model's capabilities.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Music className="h-5 w-5 text-primary-main" /> Audio Forensic Report
        </CardTitle>
        <CardDescription>
          Detailed metrics and visualizations for audio deepfake detection.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4 text-sm">
          {Object.keys(pitch).length > 0 && (
            <div>
              <h3 className="font-semibold mb-2">Pitch Analysis</h3>
              <InfoItem
                label="Mean Pitch (Hz)"
                value={pitch.mean_pitch_hz?.toFixed(2) || "N/A"}
              />
              <InfoItem
                label="Pitch Stability"
                value={
                  pitch.pitch_stability_score
                    ? `${(pitch.pitch_stability_score * 100).toFixed(1)}%`
                    : "N/A"
                }
              />
            </div>
          )}
          {Object.keys(energy).length > 0 && (
            <div>
              <h3 className="font-semibold mb-2">Energy Analysis</h3>
              <InfoItem
                label="RMS Energy"
                value={energy.rms_energy?.toFixed(4)}
              />
              <InfoItem
                label="Silence Ratio"
                value={
                  energy.silence_ratio
                    ? `${(energy.silence_ratio * 100).toFixed(1)}%`
                    : "N/A"
                }
              />
            </div>
          )}
          {Object.keys(spectral).length > 0 && (
            <div className="md:col-span-2">
              <h3 className="font-semibold mb-2">Spectral Analysis</h3>
              <InfoItem
                label="Spectral Centroid"
                value={spectral.spectral_centroid?.toFixed(2) || "N/A"}
              />
              <InfoItem
                label="Spectral Contrast"
                value={spectral.spectral_contrast?.toFixed(2) || "N/A"}
              />
            </div>
          )}
        </div>
        {visualization?.spectrogram_url && (
          <div className="pt-4">
            <SpectrogramCard
              url={visualization.spectrogram_url}
              title="Mel Spectrogram"
            />
          </div>
        )}
        {/* If you had a raw data heatmap for spectrogram: */}
        {/* {visualization?.raw_spectrogram_data && (
            <SpectrogramHeatmapChart data={visualization.raw_spectrogram_data} />
        )} */}
      </CardContent>
    </Card>
  );
};
