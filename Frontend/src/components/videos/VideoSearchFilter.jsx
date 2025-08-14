// src/components/videos/VideoSearchFilter.jsx

import { useMemo } from "react";
import { Search, X } from "lucide-react";
import { Input } from "../ui/Input";
import { Select } from "../ui/Select";

const VideoSearchFilter = ({
    searchTerm,
    statusFilter,
    sizeFilter,
    sortOrder,
    videos,
    onSearchChange,
    onStatusFilterChange,
    onSizeFilterChange,
    onSortOrderChange,
}) => {
    // Extract unique statuses from videos
    const statuses = useMemo(() => {
        const statusSet = new Set();
        videos.forEach((video) => {
            if (video.status) {
                statusSet.add(video.status);
            }
        });
        return Array.from(statusSet).sort();
    }, [videos]);

    // Define size ranges for filtering
    const sizeRanges = [
        {
            label: "Small (< 10MB)",
            value: "small",
            min: 0,
            max: 10 * 1024 * 1024,
        },
        {
            label: "Medium (10MB - 100MB)",
            value: "medium",
            min: 10 * 1024 * 1024,
            max: 100 * 1024 * 1024,
        },
        {
            label: "Large (> 100MB)",
            value: "large",
            min: 100 * 1024 * 1024,
            max: Infinity,
        },
    ];

    // Clear all filters
    const handleClearFilters = () => {
        onSearchChange("");
        onStatusFilterChange("ALL");
        onSizeFilterChange("ALL");
        onSortOrderChange("desc"); // Reset to default (latest first)
    };

    // Check if any filters are active
    const hasActiveFilters =
        searchTerm !== "" ||
        statusFilter !== "ALL" ||
        sizeFilter !== "ALL" ||
        sortOrder !== "desc"; // Changed from "asc" to "desc"

    return (
        <div className="bg-light-background dark:bg-dark-background p-6 rounded-xl shadow-sm border border-light-secondary dark:border-dark-secondary">
            <div className="space-y-4">
                {/* Filter Controls */}
                <div className="flex flex-col lg:flex-row gap-4">
                    {/* Search Input */}
                    <div className="flex-1">
                        <Input
                            type="text"
                            placeholder="Search by filename, description..."
                            value={searchTerm}
                            onChange={(e) => onSearchChange(e.target.value)}
                            className="w-full"
                            leftIcon={
                                <Search className="w-4 h-4 text-light-muted-text dark:text-dark-muted-text" />
                            }
                        />
                    </div>

                    {/* Filter Controls Row */}
                    <div className="flex flex-wrap items-center gap-3">
                        {/* Sort Order Toggle Button */}
                        {/* <button
                            onClick={() =>
                                onSortOrderChange(
                                    sortOrder === "desc" ? "asc" : "desc"
                                )
                            }
                            className="px-4 py-2 border border-light-secondary dark:border-dark-secondary 
                                     bg-light-background dark:bg-dark-background 
                                     text-light-text dark:text-dark-text 
                                     rounded-lg text-sm font-medium
                                     hover:bg-light-muted-background dark:hover:bg-dark-muted-background 
                                     transition-colors"
                        >
                            {sortOrder === "desc"
                                ? "Latest First"
                                : "Oldest First"}
                        </button> */}

                        {/* Status Filter */}
                        <Select
                            id="status-select"
                            name="status-select"
                            value={statusFilter}
                            onChange={(e) =>
                                onStatusFilterChange(e.target.value)
                            }
                            className="min-w-[120px]"
                        >
                            <option value="ALL">All Status</option>
                            {statuses.map((status) => (
                                <option key={status} value={status}>
                                    {status}
                                </option>
                            ))}
                        </Select>

                        {/* Size Filter */}
                        <Select
                            id="size-select"
                            name="size-select"
                            value={sizeFilter}
                            onChange={(e) => onSizeFilterChange(e.target.value)}
                            className="min-w-[140px]"
                        >
                            <option value="ALL">All Sizes</option>
                            {sizeRanges.map((range) => (
                                <option key={range.value} value={range.value}>
                                    {range.label}
                                </option>
                            ))}
                        </Select>

                        {/* Clear Filters Button */}
                        <button
                            onClick={handleClearFilters}
                            disabled={!hasActiveFilters}
                            className="flex items-center gap-2 px-4 py-2 
                                     bg-transparent border border-red-500 dark:border-red-400
                                     text-red-600 dark:text-red-400 
                                     rounded-lg text-sm font-medium
                                     transition-all duration-200
                                     disabled:opacity-50 disabled:cursor-not-allowed
                                     hover:bg-red-50 dark:hover:bg-red-900/20
                                     hover:shadow-sm"
                        >
                            <X className="w-4 h-4" />
                            Clear Filters
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export { VideoSearchFilter };
