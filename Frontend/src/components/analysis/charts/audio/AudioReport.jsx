// src/components/analysis/charts/audio/AudioReport.jsx

import React from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "../../../ui/Card";
import { Music, Activity, Zap, BarChart3, Contrast } from "lucide-react";
import { SpectrogramCard } from "./SpectrogramCard";
import { AudioForensicsChart } from "./AudioForensicsChart";
import { SpectrogramHeatmapChart } from "./SpectrogramHeatmapChart";

export const AudioReport = ({ result }) => {
  // Correctly destructure data directly from the result payload
  const { pitch, energy, spectral, visualization } = result || {};

  // Check if any relevant data exists to determine if we should render the report
  const hasData = pitch || energy || spectral || visualization;

  if (!hasData) {
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
              This model did not provide detailed audio forensics.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Prepare data for AudioForensicsChart components using the correct keys
  const pitchData = {
    mean_pitch: {
      label: "Mean Pitch (Hz)",
      value: pitch?.mean_pitch_hz?.toFixed(2),
    },
    stability: {
      label: "Pitch Stability",
      value: pitch?.pitch_stability_score
        ? (pitch.pitch_stability_score * 100).toFixed(1)
        : null,
      unit: "%",
    },
  };

  const energyData = {
    rms: {
      label: "RMS Energy",
      value: energy?.rms_energy?.toFixed(4),
    },
    silence: {
      label: "Silence Ratio",
      value: energy?.silence_ratio
        ? (energy.silence_ratio * 100).toFixed(1)
        : null,
      unit: "%",
    },
  };

  const spectralData = {
    centroid: {
      label: "Spectral Centroid",
      value: spectral?.spectral_centroid?.toFixed(2),
    },
    contrast: {
      label: "Spectral Contrast",
      value: spectral?.spectral_contrast?.toFixed(2),
    },
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <AudioForensicsChart
          title="Pitch Analysis"
          icon={BarChart3}
          data={pitchData}
        />
        <AudioForensicsChart
          title="Energy Analysis"
          icon={Zap}
          data={energyData}
        />
        <AudioForensicsChart
          title="Spectral Analysis"
          icon={Contrast}
          data={spectralData}
        />
      </div>

      {/* Keep the static SpectrogramCard, it will show the empty state until the URL is fixed */}
      {visualization?.spectrogram_url && (
        <SpectrogramCard
          url={visualization.spectrogram_url}
          title="Mel Spectrogram"
        />
      )}

      {/* The interactive heatmap chart, which correctly uses the raw data */}
      {visualization?.spectrogram_data && (
        <SpectrogramHeatmapChart data={visualization.spectrogram_data} />
      )}
    </div>
  );
};
