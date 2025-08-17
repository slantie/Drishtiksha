// src/pages/Monitoring.jsx

import React, { useState } from "react";
import { Card } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { SkeletonCard } from "../components/ui/SkeletonCard";
import {
    useServerStatusQuery,
    useServerHistoryQuery,
    useQueueStatusQuery,
    useAnalysisStatsQuery,
} from "../hooks/useMonitoringQuery";
import {
    Server,
    Clock,
    CheckCircle,
    AlertTriangle,
    Cpu,
    PieChart,
    Activity,
    Layers,
    RefreshCw,
    Loader2,
} from "lucide-react";
import { showToast } from "../utils/toast";
import { DataTable } from "../components/ui/DataTable";

// --- HELPER FUNCTIONS & COMPONENTS ---

const formatUptime = (totalSeconds) => {
    if (totalSeconds == null || isNaN(totalSeconds) || totalSeconds < 0)
        return "N/A";
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = Math.floor(totalSeconds % 60);
    return `${hours}h ${minutes}m ${seconds}s`;
};

const ProgressBar = ({ value, total, colorClass }) => {
    const percentage = total > 0 ? (value / total) * 100 : 0;
    return (
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
            <div
                className={`${colorClass} h-2.5 rounded-full`}
                style={{ width: `${percentage}%` }}
            ></div>
        </div>
    );
};

const RadialProgressChart = ({ percentage, label, color }) => {
    const radius = 50;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (percentage / 100) * circumference;
    return (
        <div className="relative flex items-center justify-center w-32 h-32">
            <svg className="w-full h-full" viewBox="0 0 120 120">
                <circle
                    className="text-gray-200 dark:text-gray-700"
                    strokeWidth="10"
                    stroke="currentColor"
                    fill="transparent"
                    r={radius}
                    cx="60"
                    cy="60"
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
                    cx="60"
                    cy="60"
                    transform="rotate(-90 60 60)"
                    style={{ transition: "stroke-dashoffset 0.5s ease-out" }}
                />
            </svg>
            <div className="absolute flex flex-col items-center justify-center">
                <span className="text-2xl font-bold">
                    {percentage.toFixed(0)}%
                </span>
                <span className="text-xs text-gray-500">{label}</span>
            </div>
        </div>
    );
};

const MonitoringHeader = ({ onRefresh, isRefetching }) => {
    const [isManualRefresh, setIsManualRefresh] = useState(false);
    const handleRefresh = async () => {
        setIsManualRefresh(true);
        try {
            await onRefresh();
            showToast.success("Monitoring data updated.");
        } catch (error) {
            showToast.error("Failed to refresh data.");
            console.error("Refresh error:", error);
        } finally {
            setIsManualRefresh(false);
        }
    };
    const isLoading = isRefetching || isManualRefresh;
    return (
        <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold">System Monitoring</h1>
            <Button
                onClick={handleRefresh}
                variant="outline"
                disabled={isLoading}
            >
                <>
                    {isLoading ? (
                        <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    ) : (
                        <RefreshCw className="mr-2 h-5 w-5" />
                    )}
                    {isLoading ? "Refreshing..." : "Refresh Data"}
                </>
            </Button>
        </div>
    );
};

const LiveStatusIndicator = ({ status, responseTime, uptimeSeconds }) => {
    const isHealthy = status === "running";
    return (
        <div className="p-4 rounded-lg bg-light-muted-background dark:bg-dark-muted-background flex flex-col sm:flex-row justify-between items-center gap-4">
            <div className="flex items-center gap-4">
                <div
                    className={`w-4 h-4 rounded-full animate-pulse ${
                        isHealthy ? "bg-green-500" : "bg-red-500"
                    }`}
                ></div>
                <div>
                    <h2 className="text-xl font-bold">ML Service Status</h2>
                    <p
                        className={`font-semibold ${
                            isHealthy ? "text-green-600" : "text-red-600"
                        }`}
                    >
                        {isHealthy ? "Healthy & Operational" : "Unavailable"}
                    </p>
                </div>
            </div>
            <div className="text-center sm:text-right">
                <div className="text-sm text-gray-500">
                    Uptime:{" "}
                    <span className="font-bold text-light-text dark:text-dark-text">
                        {formatUptime(uptimeSeconds)}
                    </span>
                </div>
                <div className="text-sm text-gray-500">
                    Response:{" "}
                    <span className="font-bold text-light-text dark:text-dark-text">
                        {responseTime}ms
                    </span>
                </div>
            </div>
        </div>
    );
};

const GpuInfoCard = ({ gpuInfo }) => {
    if (!gpuInfo || gpuInfo.type !== "cuda")
        return (
            <Card>
                <div className="p-6">
                    <h3 className="text-lg font-bold flex items-center gap-2">
                        <Cpu className="h-5 w-5" />
                        GPU Vitals
                    </h3>
                    <p className="text-sm text-gray-500 mt-2">
                        No CUDA-enabled GPU detected.
                    </p>
                </div>
            </Card>
        );
    return (
        <Card>
            <div className="p-6">
                <h3 className="text-lg font-bold flex items-center gap-2 mb-4">
                    <Cpu className="h-5 w-5 text-green-500" />
                    GPU Vitals
                </h3>
                <div className="flex items-center gap-4">
                    <RadialProgressChart
                        percentage={gpuInfo.memoryUsagePercent}
                        label="VRAM"
                        color="text-green-500"
                    />
                    <div className="text-sm space-y-2 flex-1">
                        <p className="font-semibold text-base truncate">
                            {gpuInfo.name}
                        </p>
                        <p className="font-semibold text-base truncate">
                            CUDA Version: {gpuInfo.cudaVersion}
                        </p>
                        <div className="flex justify-between">
                            <span>Used:</span>{" "}
                            <span className="font-mono">
                                {gpuInfo.usedMemory?.toFixed(2)} GB
                            </span>
                        </div>
                        <div className="flex justify-between">
                            <span>Total:</span>{" "}
                            <span className="font-mono">
                                {gpuInfo.totalMemory?.toFixed(2)} GB
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </Card>
    );
};

// CPU Info Card

const CpuInfoCard = ({ cpuInfo }) => {
    if (!cpuInfo)
        return (
            <Card>
                <div className="p-6">
                    <h3 className="text-lg font-bold flex items-center gap-2">
                        <Cpu className="h-5 w-5" />
                        CPU Vitals
                    </h3>
                    <p className="text-sm text-gray-500 mt-2">
                        No CPU information available.
                    </p>
                </div>
            </Card>
        );
    return (
        <Card>
            <div className="p-6">
                <h3 className="text-lg font-bold flex items-center gap-2">
                    <Cpu className="h-5 w-5 text-green-500" />
                    CPU: {cpuInfo.name}
                </h3>
            </div>
        </Card>
    );
};

// *** RESTORED HEALTH CHECK HISTORY BAR CHART ***
const HealthCheckHistoryChart = ({ data }) => {
    if (!data || data.length === 0)
        return (
            <Card className="p-6 text-center text-gray-500">
                No health check history available.
            </Card>
        );

    const displayData = [...data];
    while (displayData.length < 50) {
        displayData.unshift({
            status: "NODATA",
            createdAt: new Date().toISOString(),
            responseTime: 0,
        });
    }
    const finalData = displayData.slice(-50);

    const getStatusColor = (status) => {
        switch (status?.toUpperCase()) {
            case "HEALTHY":
                return "bg-green-500";
            case "UNHEALTHY":
                return "bg-red-500";
            case "DEGRADED":
                return "bg-yellow-500";
            default:
                return "bg-gray-300 dark:bg-gray-700";
        }
    };

    return (
        <Card>
            <div className="p-6">
                <h3 className="text-lg font-bold mb-2">Health Check History</h3>
                <p className="text-sm text-gray-500 mb-4">
                    Visualizing the status of the last 50 server health checks.
                    (Newest on the right)
                </p>
                <div className="flex items-center gap-[0.25rem] w-full h-20 bg-light-muted-background dark:bg-dark-muted-background p-2 rounded-lg">
                    {finalData.map((item, index) => (
                        <div
                            key={index}
                            className="flex-1 h-full group relative"
                        >
                            <div
                                className={`w-full h-full rounded-sm ${getStatusColor(
                                    item.status
                                )}`}
                            ></div>
                            <div className="absolute bottom-full mb-2 w-max p-2 text-xs bg-gray-800 text-white rounded-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none -translate-x-1/2 left-1/2 z-10">
                                Status: {item.status}
                                <br />
                                Response: {item.responseTime}ms
                            </div>
                        </div>
                    ))}
                </div>
                <div className="flex items-center gap-4 text-xs text-gray-500 mt-2">
                    <div className="flex items-center gap-1.5">
                        <div className="w-3 h-3 rounded-full bg-green-500"></div>
                        Healthy
                    </div>
                    <div className="flex items-center gap-1.5">
                        <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                        Degraded
                    </div>
                    <div className="flex items-center gap-1.5">
                        <div className="w-3 h-3 rounded-full bg-red-500"></div>
                        Unhealthy
                    </div>
                    <div className="flex items-center gap-1.5">
                        <div className="w-3 h-3 rounded-full bg-gray-300 dark:bg-gray-700"></div>
                        No Data
                    </div>
                </div>
            </div>
        </Card>
    );
};

const SystemInfoCard = ({ systemInfo }) => {
    if (!systemInfo) return null;
    return (
        <Card>
            <div className="p-6">
                <h3 className="text-lg font-bold flex items-center gap-2 mb-4">
                    <Server className="h-5 w-5 text-indigo-500" />
                    System Resources
                </h3>
                <div className="flex items-center gap-4">
                    <RadialProgressChart
                        percentage={systemInfo.ramUsagePercent}
                        label="RAM"
                        color="text-indigo-500"
                    />
                    <div className="text-sm space-y-2 flex-1">
                        <p className="font-semibold text-base">
                            {systemInfo.platform.includes("Windows")
                                ? "Platform: Windows"
                                : "Platform: Linux"}
                        </p>
                        <p className="font-semibold text-base">
                            Python Version: {systemInfo.pythonVersion}
                        </p>
                        <div className="flex justify-between">
                            <span>Used:</span>{" "}
                            <span className="font-mono">
                                {systemInfo.usedRam?.toFixed(2)} GB
                            </span>
                        </div>
                        <div className="flex justify-between">
                            <span>Total:</span>{" "}
                            <span className="font-mono">
                                {systemInfo.totalRam?.toFixed(2)} GB
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </Card>
    );
};

// Analysis Stats Card
const AnalysisStatsCard = ({ stats }) => {
    if (!stats) return null;
    const successRate =
        stats.total > 0 ? (stats.successful / stats.total) * 100 : 100;
    return (
        <Card>
            <div className="p-6">
                <h3 className="text-lg font-bold flex items-center gap-2 mb-4">
                    <PieChart className="h-5 w-5 text-amber-500" />
                    Model Analysis Performance (24h)
                </h3>
                <div className="space-y-4 text-sm">
                    <div>
                        <div className="flex justify-between mb-1">
                            <span>Success Rate</span>
                            <span className="font-mono">
                                {successRate.toFixed(1)}%
                            </span>
                        </div>
                        <ProgressBar
                            value={successRate}
                            total={100}
                            colorClass="bg-amber-500"
                        />
                    </div>
                    <div className="flex justify-between border-t pt-2 mt-2 dark:border-gray-700">
                        <span>Total Analyses:</span>{" "}
                        <span className="font-mono">{stats.total}</span>
                    </div>
                    <div className="flex justify-between">
                        <span>Successful:</span>{" "}
                        <span className="font-mono text-green-600">
                            {stats.successful}
                        </span>
                    </div>
                    <div className="flex justify-between">
                        <span>Failed:</span>{" "}
                        <span className="font-mono text-red-600">
                            {stats.failed}
                        </span>
                    </div>
                    <div className="flex justify-between">
                        <span>Avg. Analysis Response Time:</span>{" "}
                        <span className="font-mono">
                            {stats.avgProcessingTime.toFixed(2)}s
                        </span>
                    </div>
                </div>
            </div>
        </Card>
    );
};

const MonitoringSkeleton = () => (
    <div className="space-y-6">
        <SkeletonCard className="h-24" />
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
                <SkeletonCard className="h-96" />
                <SkeletonCard className="h-64" />
            </div>
            <div className="lg:col-span-1 space-y-6">
                <SkeletonCard className="h-48" />
                <SkeletonCard className="h-48" />
                <SkeletonCard className="h-48" />
            </div>
        </div>
    </div>
);

const Monitoring = () => {
    const {
        data: serverStatus,
        isLoading: statusLoading,
        isRefetching: statusRefetching,
        refetch: refetchStatus,
    } = useServerStatusQuery();
    const {
        data: history = [],
        isLoading: historyLoading,
        isRefetching: historyRefetching,
        refetch: refetchHistory,
    } = useServerHistoryQuery({ limit: 50 });
    const {
        data: queueStatus,
        isLoading: queueLoading,
        isRefetching: queueRefetching,
        refetch: refetchQueue,
    } = useQueueStatusQuery();
    const {
        data: analysisStats,
        isLoading: statsLoading,
        isRefetching: statsRefetching,
        refetch: refetchStats,
    } = useAnalysisStatsQuery({ timeframe: "24h" });

    const isLoading =
        statusLoading || historyLoading || queueLoading || statsLoading;
    const isRefetching =
        statusRefetching ||
        historyRefetching ||
        queueRefetching ||
        statsRefetching;

    const handleRefreshAll = () =>
        Promise.all([
            refetchStatus(),
            refetchHistory(),
            refetchQueue(),
            refetchStats(),
        ]);

    if (isLoading && !serverStatus)
        return (
            <div className="space-y-6">
                <MonitoringHeader onRefresh={() => {}} isRefetching={true} />
                <MonitoringSkeleton />
            </div>
        );

    const modelColumns = [
        {
            key: "name",
            header: "Model Name",
            accessor: (item) => (
                <span className="font-semibold">{item.name}</span>
            ),
        },
        {
            key: "loaded",
            header: "Status",
            render: (item) =>
                item.loaded ? (
                    <span className="flex items-center text-green-600">
                        <CheckCircle className="w-4 h-4 mr-2" /> Loaded
                    </span>
                ) : (
                    <span className="flex items-center text-red-600">
                        <AlertTriangle className="w-4 h-4 mr-2" /> Not Loaded
                    </span>
                ),
        },
        {
            key: "device",
            header: "Device",
            accessor: (item) => (
                <span className="font-mono uppercase">{item.device}</span>
            ),
        },
        {
            key: "memoryUsageMb",
            header: "Memory Usage",
            accessor: (item) => (
                <span className="font-mono">
                    {item.memoryUsageMb
                        ? `${item.memoryUsageMb.toFixed(1)} MB`
                        : "N/A"}
                </span>
            ),
        },
    ];

    return (
        <div className="space-y-6">
            <MonitoringHeader
                onRefresh={handleRefreshAll}
                isRefetching={isRefetching}
            />
            {serverStatus && (
                <LiveStatusIndicator
                    status={serverStatus.status}
                    responseTime={serverStatus.responseTime}
                    uptimeSeconds={serverStatus.uptimeSeconds}
                />
            )}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 space-y-6">
                    <HealthCheckHistoryChart data={history} />
                    <DataTable
                        title="Model Status & Specifications"
                        data={serverStatus?.modelsInfo || []}
                        columns={modelColumns}
                        pageSize={5}
                        showPagination={false}
                        loading={isRefetching}
                    />
                </div>
                <div className="lg:col-span-1 space-y-6">
                    {serverStatus &&
                    serverStatus.deviceInfo &&
                    serverStatus.deviceInfo.type === "cuda" ? (
                        <GpuInfoCard gpuInfo={serverStatus?.deviceInfo} />
                    ) : (
                        serverStatus &&
                        serverStatus.deviceInfo && (
                            <CpuInfoCard cpuInfo={serverStatus?.deviceInfo} />
                        )
                    )}
                    {serverStatus && (
                        <SystemInfoCard systemInfo={serverStatus?.systemInfo} />
                    )}
                    {analysisStats && (
                        <AnalysisStatsCard stats={analysisStats} />
                    )}
                    <Card>
                        <div className="p-6">
                            <h3 className="text-lg font-bold flex items-center gap-2 mb-4">
                                <Layers className="h-5 w-5 text-blue-500" />
                                Model Processing Queue ( Lifetime )
                            </h3>
                            <div className="space-y-3 text-sm">
                                <div className="space-y-2 flex items-center justify-between">
                                    <div className="flex justify-between">
                                        <span>Queued Jobs: </span>
                                        <span className=" font-bold">
                                            {queueStatus?.pendingJobs ?? 0}
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Active Jobs: </span>
                                        <span className=" font-bold">
                                            {queueStatus?.activeJobs ?? 0}
                                        </span>
                                    </div>
                                </div>
                                <div className="space-y-2 flex items-center justify-between">
                                    <div className="flex justify-between dark:border-gray-700">
                                        <span>Completed Jobs: </span>
                                        <span className="">
                                            {queueStatus?.completedJobs ?? 0}
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Failed Jobs: </span>
                                        <span className=" text-red-500">
                                            {queueStatus?.failedJobs ?? 0}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </Card>
                </div>
            </div>
        </div>
    );
};

export default Monitoring;
