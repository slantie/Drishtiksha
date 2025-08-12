// src/components/ui/DataTable.jsx

import React, { useState, useEffect, useMemo } from "react";
import PropTypes from "prop-types";
import {
    ChevronUp,
    ChevronDown,
    ChevronLeft,
    ChevronRight,
    Search,
    Loader2,
} from "lucide-react";
import { Card } from "./Card";
import { Input } from "./Input";

export function DataTable({
    data,
    columns,
    loading,
    emptyMessage = "No data available",
    pageSize = 10,
    showPagination = true,
    showSearch = true,
    searchPlaceholder = "Search...",
    onRowClick,
    showCard = true,
}) {
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

        if (searchTerm.trim()) {
            const lowercasedSearchTerm = searchTerm.toLowerCase();
            filteredData = filteredData.filter((item) =>
                columns.some((column) => {
                    if (!column.filterable) return false;
                    const value = column.accessor
                        ? column.accessor(item)
                        : item[column.key];
                    // For components, we can't search, so we check if it's a string/number
                    if (
                        typeof value === "string" ||
                        typeof value === "number"
                    ) {
                        return String(value)
                            .toLowerCase()
                            .includes(lowercasedSearchTerm);
                    }
                    // Fallback for simple object properties if accessor is not used for display
                    if (
                        typeof item[column.key] === "string" ||
                        typeof item[column.key] === "number"
                    ) {
                        return String(item[column.key])
                            .toLowerCase()
                            .includes(lowercasedSearchTerm);
                    }
                    return false;
                })
            );
        }

        if (sortConfig !== null) {
            filteredData.sort((a, b) => {
                const column = columns.find((c) => c.key === sortConfig.key);
                // Use accessor to get the sortable value
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
    }, [data, searchTerm, sortConfig, columns]);

    useEffect(() => {
        setCurrentPage(1);
    }, [searchTerm, pageSize]);

    const totalPages = Math.ceil(processedData.length / pageSize);
    const paginatedData = useMemo(() => {
        const startIndex = (currentPage - 1) * pageSize;
        return processedData.slice(startIndex, startIndex + pageSize);
    }, [processedData, currentPage, pageSize]);

    const getSortIcon = (key) => {
        if (!sortConfig || sortConfig.key !== key) return null;
        return sortConfig.direction === "asc" ? (
            <ChevronUp className="w-4 h-4 ml-1" />
        ) : (
            <ChevronDown className="w-4 h-4 ml-1" />
        );
    };

    const tableContent = (
        <div className="w-full">
            {showSearch && (
                <div className="p-4 border-b border-light-secondary dark:border-dark-secondary">
                    <Input
                        leftIcon={<Search className="w-5 h-5" />}
                        type="text"
                        placeholder={searchPlaceholder}
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                    />
                </div>
            )}
            <div className="overflow-x-auto">
                <table className="min-w-full w-full table-auto">
                    <thead>
                        <tr>
                            {columns.map((column) => (
                                <th
                                    key={column.key}
                                    onClick={() =>
                                        column.sortable &&
                                        handleSort(column.key)
                                    }
                                    className={`px-6 py-3 text-left text-sm font-semibold uppercase tracking-wider ${
                                        column.sortable ? "cursor-pointer" : ""
                                    }`}
                                >
                                    <div className="flex items-center">
                                        {column.header}{" "}
                                        {column.sortable &&
                                            getSortIcon(column.key)}
                                    </div>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-light-secondary dark:divide-dark-secondary">
                        {loading ? (
                            <tr>
                                <td
                                    colSpan={columns.length}
                                    className="text-center p-8"
                                >
                                    <Loader2 className="w-8 h-8 animate-spin mx-auto" />
                                </td>
                            </tr>
                        ) : paginatedData.length === 0 ? (
                            <tr>
                                <td
                                    colSpan={columns.length}
                                    className="text-center p-8"
                                >
                                    {emptyMessage}
                                </td>
                            </tr>
                        ) : (
                            paginatedData.map((item, index) => (
                                <tr
                                    key={item.id || index}
                                    onClick={() =>
                                        onRowClick && onRowClick(item)
                                    }
                                    className={`${
                                        onRowClick ? "cursor-pointer" : ""
                                    } hover:bg-light-muted-background dark:hover:bg-dark-muted-background`}
                                >
                                    {columns.map((column) => (
                                        <td
                                            key={column.key}
                                            className="px-6 py-4 whitespace-nowrap"
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
            {showPagination && totalPages > 1 && (
                <div className="flex items-center justify-between p-4 border-t border-light-secondary dark:border-dark-secondary">
                    <span>
                        Page {currentPage} of {totalPages}
                    </span>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() =>
                                setCurrentPage((p) => Math.max(1, p - 1))
                            }
                            disabled={currentPage === 1}
                            className="p-2 rounded-md disabled:opacity-50"
                        >
                            <ChevronLeft />
                        </button>
                        <button
                            onClick={() =>
                                setCurrentPage((p) =>
                                    Math.min(totalPages, p + 1)
                                )
                            }
                            disabled={currentPage === totalPages}
                            className="p-2 rounded-md disabled:opacity-50"
                        >
                            <ChevronRight />
                        </button>
                    </div>
                </div>
            )}
        </div>
    );

    return showCard ? <Card>{tableContent}</Card> : tableContent;
}

// PropTypes and defaultProps remain the same
DataTable.propTypes = {
    data: PropTypes.arrayOf(PropTypes.object).isRequired,
    columns: PropTypes.arrayOf(
        PropTypes.shape({
            key: PropTypes.string.isRequired,
            header: PropTypes.oneOfType([PropTypes.string, PropTypes.node])
                .isRequired,
            accessor: PropTypes.func,
            render: PropTypes.func, // New prop for custom rendering
            sortable: PropTypes.bool,
            filterable: PropTypes.bool,
        })
    ).isRequired,
    loading: PropTypes.bool,
    emptyMessage: PropTypes.string,
    pageSize: PropTypes.number,
    showPagination: PropTypes.bool,
    showSearch: PropTypes.bool,
    searchPlaceholder: PropTypes.string,
    onRowClick: PropTypes.func,
    showCard: PropTypes.bool,
};
