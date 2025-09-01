// src/components/analysis/charts/audio/SpectrogramCard.jsx

import React from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "../../../ui/Card";
import { Image } from "lucide-react";

export const SpectrogramCard = ({ url, title = "Mel Spectrogram" }) => {
  if (!url) return null;

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
          className="rounded-lg border dark:border-dark-secondary w-full"
        />
      </CardContent>
    </Card>
  );
};
