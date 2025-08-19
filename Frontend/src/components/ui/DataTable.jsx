// src/components/ui/DataTable.jsx

import React, { useState, useEffect, useMemo } from "react";
import PropTypes from "prop-types";
import {
    ChevronUp,
    ChevronDown,
    ChevronLeft,
    ChevronRight,
    Search,
} from "lucide-react";
import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
    CardDescription,
    CardFooter,
} from "./Card";
import { Input } from "./Input";
import { Button } from "./Button";
import { EmptyState } from "./EmptyState";
import { cn } from "../../lib/utils";

// REFACTOR: A dedicated skeleton row component for a clean loading state.
const SkeletonRow = ({ cells }) => (
    <tr className="animate-pulse">
        {Array.from({ length: cells }).map((_, i) => (
            <td key={i} className="px-6 py-4 whitespace-nowrap">
                <div className="h-4 bg-light-hover dark:bg-dark-hover rounded-md"></div>
            </td>
        ))}
    </tr>
);

export function DataTable({
    data,
    columns,
    loading,
    onRowClick,
    // REFACTOR: Changed prop from a simple message to a rich object for the EmptyState component.
    emptyState = {
        icon: Search,
        title: "No Results Found",
        message:
            "Your search did not return any results. Please try a different query.",
    },
    pageSize = 10,
    showPagination = true,
    showSearch = true,
    searchPlaceholder = "Search...",
    title,
    description,
    headerActions,
    disableInternalSorting = false,
}) {
    // REFACTOR: All state management and memoized logic is preserved exactly as you wrote it.
    const [searchTerm, setSearchTerm] = useState("");
    const [sortConfig, setSortConfig] = useState(null);
    const [currentPage, setCurrentPage] = useState(1);

    const handleSort = (key) => {
        let direction = "asc";
        if (
            sortConfig &&
            sortConfig.key === key &&
            sortConfig.direction === "asc"
        ) {
            direction = "desc";
        }
        setSortConfig({ key, direction });
    };

    const processedData = useMemo(() => {
        let filteredData = [...data];
        if (showSearch && searchTerm.trim()) {
            const lowercasedSearchTerm = searchTerm.toLowerCase();
            filteredData = filteredData.filter((item) =>
                columns.some((column) => {
                    if (column.filterable === false) return false;
                    const value = column.accessor
                        ? column.accessor(item)
                        : item[column.key];
                    if (
                        typeof value === "string" ||
                        typeof value === "number"
                    ) {
                        return String(value)
                            .toLowerCase()
                            .includes(lowercasedSearchTerm);
                    }
                    return false;
                })
            );
        }
        if (!disableInternalSorting && sortConfig !== null) {
            filteredData.sort((a, b) => {
                const column = columns.find((c) => c.key === sortConfig.key);
                const aValue = column.accessor
                    ? column.accessor(a)
                    : a[sortConfig.key];
                const bValue = column.accessor
                    ? column.accessor(b)
                    : b[sortConfig.key];
                if (aValue < bValue)
                    return sortConfig.direction === "asc" ? -1 : 1;
                if (aValue > bValue)
                    return sortConfig.direction === "asc" ? 1 : -1;
                return 0;
            });
        }
        return filteredData;
    }, [
        data,
        searchTerm,
        sortConfig,
        columns,
        showSearch,
        disableInternalSorting,
    ]);

    useEffect(() => {
        setCurrentPage(1);
    }, [searchTerm, pageSize]);

    const totalPages = Math.ceil(processedData.length / pageSize);
    const paginatedData = useMemo(() => {
        if (!showPagination) return processedData;
        const startIndex = (currentPage - 1) * pageSize;
        return processedData.slice(startIndex, startIndex + pageSize);
    }, [processedData, currentPage, pageSize, showPagination]);

    const getSortIcon = (key) => {
        if (!sortConfig || sortConfig.key !== key) return null;
        return sortConfig.direction === "asc" ? (
            <ChevronUp className="h-4 w-4 ml-1" />
        ) : (
            <ChevronDown className="h-4 w-4 ml-1" />
        );
    };

    return (
        <Card>
            {(title || showSearch || headerActions) && (
                // REFACTOR: Using CardHeader and sub-components for consistent layout.
                <CardHeader className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b-0 pb-4">
                    <div>
                        {title && <CardTitle>{title}</CardTitle>}
                        {description && (
                            <CardDescription className="mt-1">
                                {description}
                            </CardDescription>
                        )}
                    </div>
                    <div className="flex items-center gap-2 w-[400px]">
                        {showSearch && (
                            <div className="w-full">
                                <Input
                                    leftIcon={<Search />}
                                    type="text"
                                    placeholder={searchPlaceholder}
                                    value={searchTerm}
                                    onChange={(e) =>
                                        setSearchTerm(e.target.value)
                                    }
                                    className="rounded-full"
                                    rightIcon={<></>}
                                />
                            </div>
                        )}
                        {headerActions}
                    </div>
                </CardHeader>
            )}

            <CardContent className="p-0">
                <div className="overflow-x-auto">
                    <table className="min-w-full">
                        <thead className="bg-light-muted-background dark:bg-dark-secondary">
                            <tr>
                                {columns.map((column) => (
                                    <th
                                        key={column.key}
                                        onClick={() =>
                                            column.sortable &&
                                            handleSort(column.key)
                                        }
                                        className={cn(
                                            "px-6 py-3 text-left text-xs font-semibold text-light-muted-text dark:text-dark-muted-text uppercase tracking-wider",
                                            column.sortable &&
                                                "cursor-pointer hover:bg-light-hover dark:hover:bg-dark-hover transition-colors"
                                        )}
                                    >
                                        <div className="flex items-center">
                                            {column.header}
                                            {column.sortable &&
                                                getSortIcon(column.key)}
                                        </div>
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-light-secondary dark:divide-dark-secondary">
                            {loading ? (
                                Array.from({ length: pageSize }).map((_, i) => (
                                    <SkeletonRow
                                        key={i}
                                        cells={columns.length}
                                    />
                                ))
                            ) : paginatedData.length === 0 ? (
                                <tr>
                                    <td colSpan={columns.length}>
                                        <div className="py-12">
                                            <EmptyState {...emptyState} />
                                        </div>
                                    </td>
                                </tr>
                            ) : (
                                paginatedData.map((item, index) => (
                                    <tr
                                        key={item.id || index}
                                        onClick={() =>
                                            onRowClick && onRowClick(item)
                                        }
                                        className={cn(
                                            onRowClick &&
                                                "cursor-pointer hover:bg-light-hover dark:hover:bg-dark-hover transition-colors"
                                        )}
                                    >
                                        {columns.map((column) => (
                                            <td
                                                key={column.key}
                                                className="px-6 py-4 whitespace-nowrap text-sm text-light-text dark:text-dark-text"
                                            >
                                                {column.render
                                                    ? column.render(item)
                                                    : column.accessor
                                                    ? column.accessor(item)
                                                    : item[column.key]}
                                            </td>
                                        ))}
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </CardContent>

            {showPagination && totalPages > 1 && (
                <CardFooter className="justify-between">
                    <div className="text-sm text-light-muted-text dark:text-dark-muted-text">
                        Page {currentPage} of {totalPages}
                    </div>
                    <div className="flex items-center gap-2">
                        <Button
                            variant="outline"
                            size="sm"
                            className="rounded-full"
                            onClick={() =>
                                setCurrentPage((p) => Math.max(1, p - 1))
                            }
                            disabled={currentPage === 1}
                        >
                            <ChevronLeft className="h-4 w-4 mr-1" /> Previous
                        </Button>

                        {/* Page number buttons: show current page plus one on each side (when available) */}
                        {(() => {
                            const pages = [];
                            const start = Math.max(1, currentPage - 1);
                            const end = Math.min(totalPages, currentPage + 1);
                            for (let p = start; p <= end; p++) pages.push(p);
                            // If at the start and we have space, ensure up to 3 buttons when possible
                            if (start === 1 && end < Math.min(3, totalPages)) {
                                const extraEnd = Math.min(3, totalPages);
                                for (let p = end + 1; p <= extraEnd; p++) pages.push(p);
                            }
                            // If at the end and we have space, ensure up to 3 buttons when possible
                            if (end === totalPages && start > Math.max(1, totalPages - 2)) {
                                const extraStart = Math.max(1, totalPages - 2);
                                for (let p = extraStart; p < start; p++) pages.unshift(p);
                            }
                            return pages.map((p) => (
                                <Button
                                    key={p}
                                    variant={p === currentPage ? "default" : "outline"}
                                    size="sm"
                                    className={cn(
                                        "rounded-full h-8 w-8 p-0 flex items-center justify-center",
                                        p === currentPage && "pointer-events-none"
                                    )}
                                    onClick={() => setCurrentPage(p)}
                                    aria-current={p === currentPage ? "page" : undefined}
                                >
                                    {p}
                                </Button>
                            ));
                        })()}

                        <Button
                            variant="outline"
                            size="sm"
                            className="rounded-full"
                            onClick={() =>
                                setCurrentPage((p) =>
                                    Math.min(totalPages, p + 1)
                                )
                            }
                            disabled={currentPage === totalPages}
                        >
                            Next <ChevronRight className="h-4 w-4 ml-1" />
                        </Button>
                    </div>
                </CardFooter>
            )}
        </Card>
    );
}
