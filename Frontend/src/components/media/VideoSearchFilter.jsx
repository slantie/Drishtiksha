// src/components/media/VideoSearchFilter.jsx

import React, { useMemo } from "react";
import { Search, X } from "lucide-react";
import { Input } from "../ui/Input";
import {
    Select,
    SelectTrigger,
    SelectContent,
    SelectItem,
    SelectValue,
} from "../ui/Select"; // REFACTOR: Using our new custom Select component.
import { Button } from "../ui/Button"; // REFACTOR: Using our new Button component.
import { Card, CardContent } from "../ui/Card";

export const VideoSearchFilter = ({
    searchTerm,
    statusFilter,
    onSearchChange,
    onStatusFilterChange,
    videos,
}) => {
    // REFACTOR: Logic is preserved.
    const statuses = useMemo(() => {
        const statusSet = new Set(
            videos.map((video) => video.status).filter(Boolean)
        );
        return Array.from(statusSet).sort();
    }, [videos]);

    const handleClearFilters = () => {
        onSearchChange("");
        onStatusFilterChange("ALL");
    };

    const hasActiveFilters = searchTerm !== "" || statusFilter !== "ALL";

    return (
        // REFACTOR: Wrapped in a Card for consistent layout.
        <Card>
            <CardContent className="p-4">
                <div className="flex flex-col md:flex-row gap-3">
                    <div className="flex-grow">
                        <Input
                            type="text"
                            placeholder="Search by filename..."
                            value={searchTerm}
                            onChange={(e) => onSearchChange(e.target.value)}
                            leftIcon={<Search />}
                            rightIcon={<></>}
                            className="rounded-full"
                        />
                    </div>

                    <div className="flex items-center gap-3">
                        <Select
                            value={statusFilter}
                            onValueChange={onStatusFilterChange}
                        >
                            <SelectTrigger className="w-full md:w-[180px]">
                                <SelectValue placeholder="Filter by Status" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="ALL">
                                    All Statuses
                                </SelectItem>
                                {statuses.map((status) => (
                                    <SelectItem
                                        key={status}
                                        value={status}
                                        className="capitalize"
                                    >
                                        {status.replace("_", " ").toLowerCase()}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>

                        {hasActiveFilters && (
                            <Button
                                variant="ghost"
                                onClick={handleClearFilters}
                            >
                                <X className="mr-2 h-4 w-4" /> Clear All Filters
                            </Button>
                        )}
                    </div>
                </div>
            </CardContent>
        </Card>
    );
};
