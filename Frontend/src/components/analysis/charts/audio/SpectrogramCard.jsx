// src/components/analysis/charts/audio/SpectrogramCard.jsx

import React from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "../../../ui/Card";
import { Image, Activity } from "lucide-react"; // Added Activity for empty state

export const SpectrogramCard = ({ url, title = "Mel Spectrogram" }) => {
  if (!url) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Image className="h-5 w-5 text-primary-main" /> {title}
          </CardTitle>
          <CardDescription>
            A visual representation of the audio's frequency spectrum over time,
            used for analysis.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center p-8 text-light-muted-text dark:text-dark-muted-text">
            <Activity className="h-12 w-12 mx-auto mb-4" />
            <p className="text-lg font-semibold">No Spectrogram Available</p>
            <p className="mt-2 text-sm">
              The audio analysis did not generate a spectrogram image.
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
          <Image className="h-5 w-5 text-primary-main" /> {title}
        </CardTitle>
        <CardDescription>
          A visual representation of the audio's frequency spectrum over time,
          used for analysis.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <img
          src={url}
          alt={title}
          className="rounded-lg border dark:border-dark-secondary w-full object-cover" // Added object-cover for better image fitting
        />
      </CardContent>
    </Card>
  );
};
