// src/components/analysis/charts/audio/AudioReport.jsx

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../../../ui/Card";
import { Music } from "lucide-react";

const InfoItem = ({ label, value }) => (
  <div className="flex justify-between items-center border-b border-light-secondary/50 dark:border-dark-secondary/50 py-2">
    <span className="text-light-muted-text dark:text-dark-muted-text">
      {label}
    </span>
    <span className="font-semibold font-mono">{value}</span>
  </div>
);

export const AudioReport = ({ result }) => {
  const { pitch, energy, spectral, visualization } = result;
  if (!pitch) return <p>No detailed audio forensics available.</p>;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Music className="h-5 w-5 text-primary-main" /> Audio Forensic Report
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4 text-sm">
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
          <div>
            <h3 className="font-semibold mb-2">Energy Analysis</h3>
            <InfoItem
              label="RMS Energy"
              value={energy.rms_energy?.toFixed(4)}
            />
            <InfoItem
              label="Silence Ratio"
              value={`${(energy.silence_ratio * 100).toFixed(1)}%`}
            />
          </div>
          <div className="md:col-span-2">
            <h3 className="font-semibold mb-2">Spectral Analysis</h3>
            <InfoItem
              label="Spectral Centroid"
              value={spectral.spectral_centroid?.toFixed(2)}
            />
            <InfoItem
              label="Spectral Contrast"
              value={spectral.spectral_contrast?.toFixed(2)}
            />
          </div>
        </div>
        {visualization?.spectrogram_url && (
          <div className="pt-4">
            <h3 className="font-semibold mb-2">Mel Spectrogram</h3>
            <img
              src={visualization.spectrogram_url}
              alt="Mel Spectrogram"
              className="rounded-lg border dark:border-dark-secondary w-full"
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
};
