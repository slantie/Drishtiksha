// src/components/analysis/charts/EnvironmentCard.jsx

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../../ui/Card";
import { Cpu, Server, BrainCircuit } from "lucide-react"; // Added BrainCircuit for model

export const EnvironmentCard = ({ result }) => {
  // `result` here is expected to be an object like:
  // { modelName: "SIGLIP-LSTM-V4", systemInfo: { device_info: {...}, python_version: "...", torch_version: "..." } }
  const { modelName, systemInfo } = result || {};

  if (!systemInfo && !modelName) return null; // Render nothing if no relevant data

  // Destructure relevant properties from systemInfo for easier access
  const deviceInfo = systemInfo?.device_info;
  const pythonVersion = systemInfo?.python_version;
  const torchVersion = systemInfo?.torch_version;
  const platform = systemInfo?.platform; // Assuming platform is available

  const InfoItem = ({ label, value, icon: Icon }) =>
    value ? (
      <div className="flex justify-between items-center py-1">
        <span className="flex items-center gap-2 text-light-muted-text dark:text-dark-muted-text">
          {Icon && <Icon className="h-4 w-4" />} {label}:
        </span>
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
      <CardContent className="space-y-1 text-sm">
        <InfoItem label="Model" value={modelName} icon={BrainCircuit} />
        {deviceInfo && (
          <InfoItem
            label="Device"
            value={`${deviceInfo.type}: ${deviceInfo.name}`}
            icon={Server}
          />
        )}
        <InfoItem label="Torch Version" value={torchVersion} />
        <InfoItem label="Python Version" value={pythonVersion} />
        <InfoItem label="Platform" value={platform} />
      </CardContent>
    </Card>
  );
};
