// src/components/analysis/charts/audio/AudioForensicsCard.jsx

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../../../ui/Card";

const InfoItem = ({ label, value, unit = "" }) => (
  <div className="flex justify-between items-center border-b border-light-secondary/50 dark:border-dark-secondary/50 py-2 last:border-b-0">
    <span className="text-light-muted-text dark:text-dark-muted-text">
      {label}
    </span>
    <span className="font-semibold font-mono">
      {value ? `${value}${unit}` : "N/A"}
    </span>
  </div>
);

export const AudioForensicsCard = ({ title, data, icon: Icon }) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Icon className="h-5 w-5 text-primary-main" /> {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        {Object.entries(data).map(([key, value]) => (
          <InfoItem
            key={key}
            label={value.label}
            value={value.value}
            unit={value.unit}
          />
        ))}
      </CardContent>
    </Card>
  );
};
