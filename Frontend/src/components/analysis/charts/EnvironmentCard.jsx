// src/components/analysis/charts/EnvironmentCard.jsx

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../../ui/Card";
import { Cpu } from "lucide-react";

export const EnvironmentCard = ({ result }) => {
  const { system_info: systemInfo, model_name: modelName } = result.metrics;
  if (!systemInfo && !modelName) return null;

  const InfoItem = ({ label, value }) =>
    value ? (
      <div className="flex justify-between">
        <span>{label}:</span>
        <span className="font-semibold font-mono">{value}</span>
      </div>
    ) : null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Cpu className="h-5 w-5 text-primary-main" /> Processing Environment
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <InfoItem label="Model Name" value={modelName} />
        <InfoItem label="Device" value={systemInfo?.device_info?.name} />
        <InfoItem label="Torch Version" value={systemInfo?.torch_version} />
        <InfoItem label="Python Version" value={systemInfo?.python_version} />
      </CardContent>
    </Card>
  );
};
