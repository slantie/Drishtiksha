// src/components/analysis/charts/audio/AudioForensicsChart.jsx

import React from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "../../../ui/Card";
import { Activity, Beaker } from "lucide-react"; // Beaker for a generic forensics icon

const InfoItem = ({ label, value, unit = "" }) => (
  <div className="flex justify-between items-center border-b border-light-secondary/50 dark:border-dark-secondary/50 py-2 last:border-b-0">
    <span className="text-light-muted-text dark:text-dark-muted-text">
      {label}
    </span>
    <span className="font-semibold font-mono">
      {value !== null && typeof value !== "undefined"
        ? `${value}${unit}`
        : "N/A"}
    </span>
  </div>
);

/**
 * A generic card to display a collection of audio forensic metrics.
 * @param {object} props
 * @param {string} props.title - The title of the card (e.g., "Pitch Analysis").
 * @param {object} props.data - An object where keys are metric names and values are objects like { label: "Mean Pitch", value: 120.5, unit: "Hz" }.
 * @param {React.ElementType} props.icon - The Lucide React icon component to display.
 * @param {string} [props.description] - Optional description for the card.
 */
export const AudioForensicsChart = ({
  title,
  data,
  icon: Icon,
  description,
}) => {
  const hasData = data && Object.keys(data).length > 0;

  if (!hasData) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {Icon && <Icon className="h-5 w-5 text-primary-main" />} {title}
          </CardTitle>
          {description && <CardDescription>{description}</CardDescription>}
        </CardHeader>
        <CardContent>
          <div className="text-center p-8 text-light-muted-text dark:text-dark-muted-text">
            <Activity className="h-12 w-12 mx-auto mb-4" />
            <p className="text-lg font-semibold">No {title} Data</p>
            <p className="mt-2 text-sm">
              Detailed {title.toLowerCase()} metrics are not available.
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
          {Icon && <Icon className="h-5 w-5 text-primary-main" />} {title}
        </CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        {Object.entries(data).map(([key, item]) => (
          <InfoItem
            key={key}
            label={item.label}
            value={item.value}
            unit={item.unit}
          />
        ))}
      </CardContent>
    </Card>
  );
};
