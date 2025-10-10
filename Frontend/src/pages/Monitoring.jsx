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
  Gauge, // Added Gauge icon for overall status
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
import { Badge, DeviceBadge, VersionBadge } from "../components/ui/Badge";
import { showToast } from "../utils/toast.jsx";
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

// --- SUB-COMPONENTS ---

const formatUptime = (totalSeconds) => {
  if (totalSeconds == null || isNaN(totalSeconds) || totalSeconds < 0)
    return "N/A";
  const days = Math.floor(totalSeconds / 86400);
  const hours = Math.floor((totalSeconds % 86400) / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`; // Show minutes for shorter uptimes
};

const RadialProgressChart = ({ percentage, label, color }) => {
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  // Ensure percentage is clamped between 0 and 100
  const clampedPercentage = Math.max(0, Math.min(100, percentage));
  const offset = circumference - (clampedPercentage / 100) * circumference;

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
        <span className="text-2xl font-bold">
          {clampedPercentage.toFixed(0)}%
        </span>
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
            <span className="font-semibold ">{item.value || "N/A"}</span>
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
            <h2 className="font-bold text-lg">ML Service Status</h2>{" "}
            {/* Consistent title size */}
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
            Response Time:{" "}
            <span className="font-semibold">
              {serverStatus?.responseTimeMs
                ? `${serverStatus.responseTimeMs}ms`
                : "N/A"}
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
        <Gauge className="h-12 w-12 mx-auto mb-4 text-gray-400" />
        <p>No health check history available.</p>
      </Card>
    );

  const displayData = data
    .map((item) => ({
      ...item.statsPayload, // Contains deviceInfo, systemInfo etc from the Python service
      status: item.status, // Backend's 'HEALTHY', 'UNHEALTHY', etc.
      responseTime: item.responseTimeMs/1000,
      checkedAt: item.checkedAt,
    }))
    .reverse(); // Display oldest first for time series

  const getStatusFillColor = (status) => {
    switch (status?.toUpperCase()) {
      case "HEALTHY":
        return "#22c55e"; // Green
      case "UNHEALTHY":
        return "#ef4444"; // Red
      case "DEGRADED": // Added from backend schema
        return "#f59e0b"; // Amber/Yellow
      default:
        return "#94a3b8"; // Gray
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
          <p>Response Time: {data.responseTime/1000}s</p>
          <p>Time: {new Date(data.checkedAt).toLocaleTimeString()}</p>{" "}
          {/* Show time */}
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
          Response time of the last {data.length} server health checks.
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
            <XAxis hide={true} dataKey="checkedAt" />{" "}
            {/* Hide but provide dataKey for tooltip */}
            <YAxis unit="s" tick={{ fill: "currentColor", fontSize: 11 }} />
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
  <div className="space-y-6 w-full max-w-full mx-auto">
    {" "}
    {/* Consistent spacing, full width */}
    <SkeletonCard className="h-24 w-full" /> {/* PageHeader skeleton */}
    <SkeletonCard className="h-20 w-full" />{" "}
    {/* LiveStatusIndicator skeleton */}
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {" "}
      {/* Consistent gap */}
      <div className="lg:col-span-2 space-y-6">
        {" "}
        {/* Consistent vertical spacing */}
        <SkeletonCard className="h-64" />{" "}
        {/* HealthCheckHistoryChart skeleton */}
        <SkeletonCard className="h-80" /> {/* DataTable skeleton */}
      </div>
      <div className="lg:col-span-1 space-y-6">
        {" "}
        {/* Consistent vertical spacing */}
        <SkeletonCard className="h-48" /> {/* ResourceCard skeleton */}
        <SkeletonCard className="h-48" /> {/* ResourceCard skeleton */}
        <SkeletonCard className="h-48" /> {/* ResourceCard skeleton */}
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
    { limit: 50 } // Pass limit to hook
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
      <div className="space-y-6 w-full max-w-full mx-auto">
        {" "}
        {/* Consistent spacing, full width */}
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
              "The monitoring service is currently unreachable. Please ensure the ML backend is running."}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  // Helper function to extract version from model name
  const extractVersion = (modelName) => {
    if (!modelName) return null;
    // Extract version from patterns like: EFFICIENTNET-B7-V1, SIGLIP-LSTM-V3, etc.
    const match = modelName.match(/-V(\d+)$/i);
    return match ? match[1] : null;
  };

  // Helper function to determine media types
  const getMediaTypes = (item) => {
    const types = [];
    if (item.isVideo) types.push("Video");
    if (item.isAudio) types.push("Audio");
    if (item.isImage) types.push("Image");
    return types.length > 0 ? types : ["Unknown"];
  };

  const modelColumns = [
    {
      key: "name",
      header: "Model Name",
      render: (item) => (
        <div className="flex flex-col gap-1">
          <span className="font-semibold">{item.name}</span>
          <span className="text-xs text-light-muted-text dark:text-dark-muted-text">
            {item.description || "No description"}
          </span>
        </div>
      ),
      sortable: true,
      filterable: true,
    },
    {
      key: "loaded",
      header: "Status",
      render: (item) =>
        item.loaded ? (
          <Badge variant="success" icon={CheckCircle} iconSize="w-3.5 h-3.5">
            Loaded
          </Badge>
        ) : (
          <Badge variant="danger" icon={AlertTriangle} iconSize="w-3.5 h-3.5">
            Not Loaded
          </Badge>
        ),
      sortable: true,
      filterable: true,
    },
    {
      key: "mediaType",
      header: "Media Type",
      render: (item) => {
        const types = getMediaTypes(item);
        const colorMap = {
          Video: "info",
          Audio: "purple",
          Image: "pink",
        };
        return (
          <div className="flex flex-wrap gap-1">
            {types.map((type) => (
              <Badge 
                key={type} 
                variant={colorMap[type] || "default"}
                size="sm"
              >
                {type}
              </Badge>
            ))}
          </div>
        );
      },
      sortable: true,
      filterable: true,
    },
    {
      key: "device",
      header: "Device",
      render: (item) => (
        <DeviceBadge device={item.device?.toUpperCase()} />
      ),
      sortable: true,
      filterable: true,
    },
    {
      key: "version",
      header: "Version",
      render: (item) => {
        const version = extractVersion(item.name);
        return version ? (
          <VersionBadge version={version} />
        ) : (
          <span className="text-xs text-light-muted-text dark:text-dark-muted-text">
            N/A
          </span>
        );
      },
      sortable: true,
      filterable: false,
    },
  ];

  const { device_info: deviceInfo, system_info: systemInfo } =
    serverStatus || {};

  return (
    <div className="space-y-6 w-full max-w-full mx-auto">
      {" "}
      {/* Consistent vertical spacing, full width */}
      <PageHeader
        title="System Monitoring"
        description="Live status and performance metrics for the analysis services."
        actions={
          <Button
            onClick={handleRefreshAll}
            isLoading={statusRefetching}
            variant="outline"
            aria-label="Refresh all monitoring data"
          >
            <RefreshCw className="mr-2 h-4 w-4" /> Refresh Data
          </Button>
        }
      />
      <LiveStatusIndicator serverStatus={serverStatus} />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {" "}
        {/* Consistent gap */}
        <div className="lg:col-span-2 space-y-6">
          {" "}
          {/* Consistent vertical spacing */}
          <HealthCheckHistoryChart data={history} />
          <DataTable
            title="Loaded AI Models"
            description="Currently loaded models in the ML service and their capabilities."
            columns={modelColumns}
            data={serverStatus?.models_info || []}
            loading={statusRefetching}
            showPagination={false}
            showSearch={true}
            searchPlaceholder="Search models..."
          />
        </div>
        <div className="lg:col-span-1 space-y-6">
          {" "}
          {/* Consistent vertical spacing */}
          {deviceInfo && (
            <ResourceCard
              title={deviceInfo.type === "cuda" ? "GPU Vitals" : "CPU Vitals"}
              icon={Cpu}
              chartData={
                deviceInfo.type === "cuda" &&
                deviceInfo.memory_usage_percent !== undefined
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
                      { label: "CUDA Version", value: deviceInfo.cuda_version },
                      {
                        label: "Used VRAM",
                        value: `${deviceInfo.used_memory?.toFixed(2)} GB`,
                      },
                      {
                        label: "Total VRAM",
                        value: `${deviceInfo.total_memory?.toFixed(2)} GB`,
                      },
                      {
                        label: "GPU Temp",
                        value: deviceInfo.temperature
                          ? `${deviceInfo.temperature}°C`
                          : "N/A",
                      },
                    ]
                  : [
                      // CPU specific details
                      { label: "Cores", value: deviceInfo.core_count },
                      { label: "Threads", value: deviceInfo.thread_count },
                      {
                        label: "CPU Usage",
                        value: deviceInfo.cpu_usage_percent
                          ? `${deviceInfo.cpu_usage_percent.toFixed(1)}%`
                          : "N/A",
                      },
                    ]),
              ]}
            />
          )}
          {systemInfo && (
            <ResourceCard
              title="System Resources"
              icon={Server}
              chartData={
                systemInfo.ram_usage_percent !== undefined
                  ? {
                      percentage: systemInfo.ram_usage_percent,
                      label: "RAM",
                      color: "text-indigo-500",
                    }
                  : null
              }
              details={[
                { label: "Platform", value: systemInfo.platform },
                { label: "OS", value: systemInfo.os_type }, // Assuming os_type exists
                { label: "Python", value: systemInfo.python_version },
                {
                  label: "Used RAM",
                  value: `${systemInfo.used_ram?.toFixed(2)} GB`,
                },
                {
                  label: "Total RAM",
                  value: `${systemInfo.total_ram?.toFixed(2)} GB`,
                },
                {
                  label: "CPU Load",
                  value: systemInfo.cpu_load_percent
                    ? `${systemInfo.cpu_load_percent.toFixed(1)}%`
                    : "N/A",
                },
              ]}
            />
          )}
          {queueStatus && (
            <ResourceCard
              title="Processing Queue"
              icon={Layers}
              details={[
                { label: "Pending Jobs", value: queueStatus.pending },
                { label: "Active Jobs", value: queueStatus.active },
                { label: "Completed Jobs", value: queueStatus.completed },
                { label: "Failed Jobs", value: queueStatus.failed },
              ]}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default Monitoring;
