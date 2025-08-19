// src/pages/Monitoring.jsx

import React from "react";
import {
    useServerStatusQuery,
    useServerHistoryQuery,
    useQueueStatusQuery,
    useAnalysisStatsQuery,
} from "../hooks/useMonitoringQuery";
import {
    Server,
    CheckCircle,
    AlertTriangle,
    Cpu,
    PieChart,
    Activity,
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
import { showToast } from "../utils/toast";
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

// --- HELPER FUNCTIONS & SUB-COMPONENTS (Restored & Refined) ---

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
                <span className="text-2xl font-bold">
                    {percentage.toFixed(0)}%
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
                        <span className="font-semibold ">{item.value}</span>
                    </div>
                ))}
            </div>
        </CardContent>
    </Card>
);

const LiveStatusIndicator = ({ status, responseTime, uptimeSeconds }) => {
    const isHealthy = status === "running";
    return (
        <Card
            className={`${isHealthy ? "border-green-500" : "border-red-500"}`}
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
                            {isHealthy
                                ? "Healthy & Operational"
                                : "Unavailable"}
                        </p>
                    </div>
                </div>
                <div className="text-sm text-light-muted-text dark:text-dark-muted-text text-center sm:text-right space-y-1">
                    <div>
                        Uptime:{" "}
                        <span className="font-semibold text-light-text dark:text-dark-text">
                            {formatUptime(uptimeSeconds)}
                        </span>
                    </div>
                    <div>
                        Response:{" "}
                        <span className="font-semibold text-light-text dark:text-dark-text">
                            {responseTime}ms
                        </span>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
};

const HealthCheckHistoryChart = ({ data }) => {
    if (!data || data.length === 0) {
        return (
            <Card className="p-6 text-center text-light-muted-text dark:text-dark-muted-text">
                No health check history available.
            </Card>
        );
    }

    // REFACTOR: Data preparation is the same, ensuring we show the latest 50 checks.
    const displayData = [...data].reverse().slice(0, 50);

    // REFACTOR: Color logic now returns hex codes for the SVG 'fill' property.
    const getStatusFillColor = (status) => {
        switch (status?.toUpperCase()) {
            case "HEALTHY":
                return "#22c55e"; // green-500
            case "UNHEALTHY":
                return "#ef4444"; // red-500
            default:
                return "#f59e0b"; // amber-500 for DEGRADED or other statuses
        }
    };

    // REFACTOR: A custom tooltip component for Recharts to display formatted data.
    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload;
            return (
                <div className="p-2 bg-gray-800 text-white rounded-md shadow-lg border border-gray-700 text-xs">
                    <p className="font-bold capitalize">
                        Status: {data.status}
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
                    Response time of the last 50 server health checks. (Newest
                    on right)
                </CardDescription>
            </CardHeader>
            <CardContent>
                {/* REFACTOR: The entire chart is now built with Recharts components. */}
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
                        <YAxis
                            unit="ms"
                            tick={{ fill: "currentColor", fontSize: 11 }}
                        />
                        <Tooltip
                            content={<CustomTooltip />}
                            cursor={{ fill: "rgba(128,128,128,0.1)" }}
                        />
                        <Bar
                            dataKey="responseTime"
                            radius={[4, 4, 0, 0]}
                            barSize={25}
                            alignmentBaseline="middle"
                        >
                            {displayData.map((entry, index) => (
                                <Cell
                                    key={`bar-cell-${index}`}
                                    fill={getStatusFillColor(entry.status)}
                                    className="cursor-pointer"
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
    // --- DATA FETCHING (Preserved) ---
    const {
        data: serverStatus,
        error: serverStatusError,
        isLoading: statusLoading,
        isRefetching: statusRefetching,
        refetch: refetchStatus,
    } = useServerStatusQuery();
    const { data: history = [], refetch: refetchHistory } =
        useServerHistoryQuery({ limit: 50 });
    const { data: queueStatus, refetch: refetchQueue } = useQueueStatusQuery();
    const { data: analysisStats, refetch: refetchStats } =
        useAnalysisStatsQuery({ timeframe: "24h" });

    const handleRefreshAll = () =>
        Promise.all([
            refetchStatus(),
            refetchHistory(),
            refetchQueue(),
            refetchStats(),
        ])
            .then(() => showToast.success("Monitoring data updated."))
            .catch(() => showToast.error("Failed to refresh data."));

    // --- RENDER LOGIC ---
    if (statusLoading) return <MonitoringSkeleton />;

    // REFACTOR: Added a dedicated error state for when the backend is unreachable (e.g., 503 error).
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
                            "The monitoring service is currently unreachable. Please try again later."}
                    </AlertDescription>
                </Alert>
            </div>
        );
    }

    const modelColumns = [
        {
            key: "name",
            header: "Model Name",
            render: (item) => (
                <span className="font-semibold">{item.name}</span>
            ),
        },
        {
            key: "loaded",
            header: "Status",
            render: (item) =>
                item.loaded ? (
                    <span className="flex items-center gap-2 text-green-600">
                        <CheckCircle className="w-4 h-4" /> Loaded
                    </span>
                ) : (
                    <span className="flex items-center gap-2 text-red-600">
                        <AlertTriangle className="w-4 h-4" /> Not Loaded
                    </span>
                ),
        },
        {
            key: "isDetailed",
            header: "Detailed Analysis",
            render: (item) =>
                item.isDetailed ? (
                    <span className="flex items-center gap-2 text-blue-600">
                        <CheckCircle className="w-4 h-4" /> Supported
                    </span>
                ) : (
                    <span className="flex items-center gap-2 text-gray-500">
                        <AlertTriangle className="w-4 h-4" /> Not Supported
                    </span>
                ),
        },
        {
            key: "device",
            header: "Device",
            render: (item) => <span className=" uppercase">{item.device}</span>,
        },
        {
            key: "memoryUsageMb",
            header: "Memory",
            render: (item) => (
                <span className="">
                    {item.memoryUsageMb
                        ? `${item.memoryUsageMb.toFixed(1)} MB`
                        : "N/A"}
                </span>
            ),
        },
    ];

    return (
        <div className="space-y-4">
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

            {serverStatus && (
                <LiveStatusIndicator
                    status={serverStatus.status}
                    responseTime={serverStatus.responseTime}
                    uptimeSeconds={serverStatus.uptimeSeconds}
                />
            )}

            {/* REFACTOR: The main grid now has a breakpoint for medium screens for better tablet responsiveness. */}
            <div className="grid grid-cols-1 md:grid-cols-2 items-start space-y-4">
                <div className="md:col-span-2 lg:col-span-2 space-y-4">
                    <HealthCheckHistoryChart data={history} />
                    <DataTable
                        title="Loaded AI Models"
                        columns={modelColumns}
                        data={serverStatus?.modelsInfo || []}
                        loading={statusRefetching}
                        showPagination={false}
                    />
                </div>
                <div className="col-span-4 space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* REFACTOR: Now using the new, responsive ResourceCard for all system info. */}
                        {serverStatus?.deviceInfo && (
                            <ResourceCard
                                title={
                                    serverStatus.deviceInfo.type === "cuda"
                                        ? "GPU Vitals"
                                        : "CPU Vitals"
                                }
                                icon={Cpu}
                                chartData={
                                    serverStatus.deviceInfo.type === "cuda"
                                        ? {
                                              percentage:
                                                  serverStatus.deviceInfo
                                                      .memoryUsagePercent,
                                              label: "VRAM",
                                              color: "text-green-500",
                                          }
                                        : null
                                }
                                details={[
                                    {
                                        label: "Name",
                                        value: serverStatus.deviceInfo.name,
                                    },
                                    ...(serverStatus.deviceInfo.type === "cuda"
                                        ? [
                                              {
                                                  label: "CUDA",
                                                  value: serverStatus.deviceInfo
                                                      .cudaVersion,
                                              },
                                              {
                                                  label: "Used",
                                                  value: `${serverStatus.deviceInfo.usedMemory?.toFixed(
                                                      2
                                                  )} GB`,
                                              },
                                              {
                                                  label: "Total",
                                                  value: `${serverStatus.deviceInfo.totalMemory?.toFixed(
                                                      2
                                                  )} GB`,
                                              },
                                          ]
                                        : []),
                                ]}
                            />
                        )}
                        {serverStatus?.systemInfo && (
                            <ResourceCard
                                title="System Resources"
                                icon={Server}
                                chartData={{
                                    percentage:
                                        serverStatus.systemInfo.ramUsagePercent,
                                    label: "RAM",
                                    color: "text-indigo-500",
                                }}
                                details={[
                                    {
                                        label: "Platform",
                                        value: serverStatus.systemInfo.platform,
                                    },
                                    {
                                        label: "Python",
                                        value: serverStatus.systemInfo
                                            .pythonVersion,
                                    },
                                    {
                                        label: "Used",
                                        value: `${serverStatus.systemInfo.usedRam?.toFixed(
                                            2
                                        )} GB`,
                                    },
                                    {
                                        label: "Total",
                                        value: `${serverStatus.systemInfo.totalRam?.toFixed(
                                            2
                                        )} GB`,
                                    },
                                ]}
                            />
                        )}
                        {analysisStats && (
                            <ResourceCard
                                title="Analysis Stats (24h)"
                                icon={PieChart}
                                details={[
                                    {
                                        label: "Total",
                                        value: analysisStats.total,
                                    },
                                    {
                                        label: "Successful",
                                        value: analysisStats.successful,
                                    },
                                    {
                                        label: "Failed",
                                        value: analysisStats.failed,
                                    },
                                    {
                                        label: "Avg. Time",
                                        value: `${analysisStats.avgProcessingTime.toFixed(
                                            2
                                        )}s`,
                                    },
                                ]}
                            />
                        )}
                        {queueStatus && (
                            <ResourceCard
                                title="Processing Queue"
                                icon={Layers}
                                details={[
                                    {
                                        label: "Pending",
                                        value: queueStatus.pendingJobs,
                                    },
                                    {
                                        label: "Active",
                                        value: queueStatus.activeJobs,
                                    },
                                    {
                                        label: "Completed",
                                        value: queueStatus.completedJobs,
                                    },
                                    {
                                        label: "Failed",
                                        value: queueStatus.failedJobs,
                                    },
                                ]}
                            />
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Monitoring;
