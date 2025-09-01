// src/pages/Monitoring.jsx

import React from "react";
import {
  useServerStatusQuery,
  useServerHistoryQuery,
  useQueueStatusQuery,
} from "../hooks/useMonitoringQuery";
import {
  Server,
  CheckCircle,
  AlertTriangle,
  Cpu,
  Layers,
  RefreshCw,
} from "lucide-react";
import { PageHeader } from "../components/layout/PageHeader";
import { Button } from "../components/ui/Button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "../components/ui/Card";
import { DataTable } from "../components/ui/DataTable";
import { SkeletonCard } from "../components/ui/SkeletonCard";
import { Alert, AlertTitle, AlertDescription } from "../components/ui/Alert";
import { showToast } from "../utils/toast.js";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

// --- SUB-COMPONENTS (Refactored for new data props) ---

const formatUptime = (totalSeconds) => {
  if (totalSeconds == null || isNaN(totalSeconds) || totalSeconds < 0)
    return "N/A";
  const days = Math.floor(totalSeconds / 86400);
  const hours = Math.floor((totalSeconds % 86400) / 3600);
  return `${days}d ${hours}h`;
};

const RadialProgressChart = ({ percentage, label, color }) => {
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (percentage / 100) * circumference;
  return (
    <div className="relative flex-shrink-0 flex items-center justify-center w-28 h-28">
      <svg className="w-full h-full" viewBox="0 0 100 100">
        <circle
          className="text-light-hover dark:text-dark-hover"
          strokeWidth="10"
          stroke="currentColor"
          fill="transparent"
          r={radius}
          cx="50"
          cy="50"
        />
        <circle
          className={color}
          strokeWidth="10"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          stroke="currentColor"
          fill="transparent"
          r={radius}
          cx="50"
          cy="50"
          transform="rotate(-90 50 50)"
          style={{ transition: "stroke-dashoffset 0.5s ease-out" }}
        />
      </svg>
      <div className="absolute flex flex-col items-center justify-center">
        <span className="text-2xl font-bold">{percentage.toFixed(0)}%</span>
        <span className="text-xs text-light-muted-text dark:text-dark-muted-text">
          {label}
        </span>
      </div>
    </div>
  );
};

const ResourceCard = ({ title, icon: Icon, chartData, details }) => (
  <Card>
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <Icon className="h-5 w-5 text-primary-main" /> {title}
      </CardTitle>
    </CardHeader>
    <CardContent className="flex flex-col sm:flex-row items-center gap-4">
      {chartData && (
        <div className="flex-shrink-0">
          <RadialProgressChart
            percentage={chartData.percentage}
            label={chartData.label}
            color={chartData.color}
          />
        </div>
      )}
      <div className="w-full text-sm space-y-2">
        {details.map((item) => (
          <div
            key={item.label}
            className="flex justify-between items-center border-b border-light-secondary/50 dark:border-dark-secondary/50 py-1"
          >
            <span className="text-light-muted-text dark:text-dark-muted-text">
              {item.label}:
            </span>
            <span className="font-semibold ">{item.value}</span>
          </div>
        ))}
      </div>
    </CardContent>
  </Card>
);

const LiveStatusIndicator = ({ serverStatus }) => {
  const isHealthy = serverStatus?.status === "running";
  return (
    <Card
      className={`${isHealthy ? "border-green-500/50" : "border-red-500/50"}`}
    >
      <CardContent className="p-4 flex flex-col sm:flex-row justify-between items-center gap-4">
        <div className="flex items-center gap-4">
          {isHealthy ? (
            <CheckCircle className="h-8 w-8 text-green-500" />
          ) : (
            <AlertTriangle className="h-8 w-8 text-red-500" />
          )}
          <div>
            <h2 className="font-bold">ML Service Status</h2>
            <p
              className={`font-semibold ${
                isHealthy ? "text-green-600" : "text-red-600"
              }`}
            >
              {isHealthy ? "Healthy & Operational" : "Unavailable"}
            </p>
          </div>
        </div>
        <div className="text-sm text-center sm:text-right space-y-1">
          <div>
            Uptime:{" "}
            <span className="font-semibold">
              {formatUptime(serverStatus?.uptime_seconds)}
            </span>
          </div>
          <div>
            Response:{" "}
            <span className="font-semibold">
              {serverStatus?.responseTimeMs}ms
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

const HealthCheckHistoryChart = ({ data }) => {
  if (!data?.length)
    return (
      <Card className="p-6 text-center text-light-muted-text dark:text-dark-muted-text">
        No health check history available.
      </Card>
    );

  const displayData = data
    .map((item) => ({
      ...item.statsPayload,
      status: item.status,
      responseTime: item.responseTimeMs,
    }))
    .reverse()
    .slice(0, 50);

  const getStatusFillColor = (status) => {
    switch (status?.toUpperCase()) {
      case "HEALTHY":
        return "#22c55e";
      case "UNHEALTHY":
        return "#ef4444";
      default:
        return "#f59e0b";
    }
  };

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload?.[0]) {
      const data = payload[0].payload;
      return (
        <div className="p-2 bg-gray-800 text-white rounded-md shadow-lg border border-gray-700 text-xs">
          <p className="font-bold capitalize">
            Status: {data.status?.toLowerCase()}
          </p>
          <p>Response Time: {data.responseTime}ms</p>
        </div>
      );
    }
    return null;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Health Check History</CardTitle>
        <CardDescription>
          Response time of the last 50 server health checks.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={150}>
          <BarChart
            data={displayData}
            margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="currentColor"
              className="opacity-10"
            />
            <XAxis hide={true} />
            <YAxis unit="ms" tick={{ fill: "currentColor", fontSize: 11 }} />
            <Tooltip
              content={<CustomTooltip />}
              cursor={{ fill: "rgba(128,128,128,0.1)" }}
            />
            <Bar dataKey="responseTime" radius={[4, 4, 0, 0]}>
              {displayData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={getStatusFillColor(entry.status)}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

const MonitoringSkeleton = () => (
  <div className="space-y-4">
    <div className="flex justify-between items-center">
      <SkeletonCard className="h-10 w-64" />
      <SkeletonCard className="h-10 w-32" />
    </div>
    <SkeletonCard className="h-20" />
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <div className="lg:col-span-2 space-y-8">
        <SkeletonCard className="h-64" />
        <SkeletonCard className="h-80" />
      </div>
      <div className="lg:col-span-1 space-y-4">
        <SkeletonCard className="h-48" />
        <SkeletonCard className="h-48" />
      </div>
    </div>
  </div>
);

const Monitoring = () => {
  const {
    data: serverStatus,
    error: serverStatusError,
    isLoading: statusLoading,
    isRefetching: statusRefetching,
    refetch: refetchStatus,
  } = useServerStatusQuery();
  const { data: history = [], refetch: refetchHistory } = useServerHistoryQuery(
    { limit: 50 }
  );
  const { data: queueStatus, refetch: refetchQueue } = useQueueStatusQuery();

  const handleRefreshAll = () => {
    Promise.all([refetchStatus(), refetchHistory(), refetchQueue()])
      .then(() => showToast.success("Monitoring data updated."))
      .catch(() => showToast.error("Failed to refresh all data."));
  };

  if (statusLoading) return <MonitoringSkeleton />;

  if (serverStatusError) {
    return (
      <div className="space-y-4">
        <PageHeader
          title="System Monitoring"
          description="Live status and performance metrics for the analysis services."
          actions={
            <Button onClick={handleRefreshAll} variant="outline">
              <RefreshCw className="mr-2 h-4 w-4" /> Try Again
            </Button>
          }
        />
        <Alert variant="destructive">
          <AlertTriangle className="h-5 w-5" />
          <AlertTitle>Service Unavailable</AlertTitle>
          <AlertDescription>
            {serverStatusError.message ||
              "The monitoring service is currently unreachable."}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  // REFACTOR: The columns are now simpler and match the new `models_info` schema.
  const modelColumns = [
    {
      key: "name",
      header: "Model Name",
      render: (item) => <span className="font-semibold">{item.name}</span>,
    },
    {
      key: "loaded",
      header: "Status",
      render: (item) =>
        item.loaded ? (
          <span className="flex items-center gap-2 text-green-600">
            <CheckCircle className="w-4 h-4" />
            Loaded
          </span>
        ) : (
          <span className="flex items-center gap-2 text-red-600">
            <AlertTriangle className="w-4 h-4" />
            Not Loaded
          </span>
        ),
    },
    {
      key: "media_type",
      header: "Type",
      render: (item) => (item.isAudio ? "Audio" : "Video/Image"),
    },
    {
      key: "device",
      header: "Device",
      render: (item) => (
        <span className="font-mono uppercase">{item.device}</span>
      ),
    },
  ];

  const { device_info: deviceInfo, system_info: systemInfo } =
    serverStatus || {};

  return (
    <div className="space-y-6">
      <PageHeader
        title="System Monitoring"
        description="Live status and performance metrics for the analysis services."
        actions={
          <Button
            onClick={handleRefreshAll}
            isLoading={statusRefetching}
            variant="outline"
          >
            <RefreshCw className="mr-2 h-4 w-4" /> Refresh Data
          </Button>
        }
      />

      <LiveStatusIndicator serverStatus={serverStatus} />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <HealthCheckHistoryChart data={history} />
          <DataTable
            title="Loaded AI Models"
            columns={modelColumns}
            data={serverStatus?.models_info || []}
            loading={statusRefetching}
            showPagination={false}
          />
        </div>
        <div className="lg:col-span-1 space-y-6">
          {deviceInfo && (
            <ResourceCard
              title={deviceInfo.type === "cuda" ? "GPU Vitals" : "CPU Vitals"}
              icon={Cpu}
              chartData={
                deviceInfo.type === "cuda"
                  ? {
                      percentage: deviceInfo.memory_usage_percent,
                      label: "VRAM",
                      color: "text-green-500",
                    }
                  : null
              }
              details={[
                { label: "Name", value: deviceInfo.name },
                ...(deviceInfo.type === "cuda"
                  ? [
                      { label: "CUDA", value: deviceInfo.cuda_version },
                      {
                        label: "Used VRAM",
                        value: `${deviceInfo.used_memory?.toFixed(2)} GB`,
                      },
                      {
                        label: "Total VRAM",
                        value: `${deviceInfo.total_memory?.toFixed(2)} GB`,
                      },
                    ]
                  : []),
              ]}
            />
          )}
          {systemInfo && (
            <ResourceCard
              title="System Resources"
              icon={Server}
              chartData={{
                percentage: systemInfo.ram_usage_percent,
                label: "RAM",
                color: "text-indigo-500",
              }}
              details={[
                { label: "Platform", value: systemInfo.platform },
                { label: "Python", value: systemInfo.python_version },
                {
                  label: "Used RAM",
                  value: `${systemInfo.used_ram?.toFixed(2)} GB`,
                },
                {
                  label: "Total RAM",
                  value: `${systemInfo.total_ram?.toFixed(2)} GB`,
                },
              ]}
            />
          )}
          {queueStatus && (
            <ResourceCard
              title="Processing Queue"
              icon={Layers}
              details={[
                { label: "Pending", value: queueStatus.pending },
                { label: "Active", value: queueStatus.active },
                { label: "Completed", value: queueStatus.completed },
                { label: "Failed", value: queueStatus.failed },
              ]}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default Monitoring;
