/**
 * @file src/components/ui/DataTable.jsx
 * @description Advanced, flexible data table component with sorting, filtering, pagination, responsive design, and file preview support.
 * This version includes extensive logging for debugging.
 */

"use client";

import React, { useState, useEffect, useMemo } from "react";
import PropTypes from "prop-types";
import {
    ChevronUp,
    ChevronDown,
    ChevronLeft,
    ChevronRight,
    Table,
    Search,
    Loader2
} from "lucide-react";
import { Card } from "./Card";
import { Input } from "./Input";

/**
 * A highly customizable and reusable data table component.
 *
 * @param {object} props - The component props.
 * @param {Array<object>} props.data - The data array to be displayed.
 * @param {boolean} [props.showserial=false] - Whether to show a serial number column.
 * @param {Array<object>} props.columns - Array of column definitions.
 * @param {boolean} [props.loading=false] - A boolean to show a loading spinner.
 * @param {string} [props.emptyMessage="No data available"] - Message to display when the data is empty.
 * @param {number} [props.pageSize=10] - Number of items per page.
 * @param {boolean} [props.showPagination=true] - Whether to show pagination controls.
 * @param {boolean} [props.showSearch=true] - Whether to show the search input.
 * @param {string} [props.searchPlaceholder="Search..."] - Placeholder for the search input.
 * @param {Function} [props.onRowClick] - Callback function for row clicks.
 * @param {string} [props.className] - Custom class for the main container.
 * @param {string|Function} [props.rowClassName] - Custom class for table rows.
 * @param {string} [props.headerClassName] - Custom class for the table header.
 * @param {string} [props.cellClassName] - Custom class for table cells.
 * @param {boolean} [props.stickyHeader=false] - Whether the table header should be sticky.
 * @param {string} [props.maxHeight] - Max height for the table body, enabling scrolling.
 * @param {boolean} [props.previewMode=false] - Special mode for file previews.
 * @param {string} [props.previewTitle="File Preview"] - Title for the preview mode.
 * @param {object} [props.previewIcon=Table] - Icon for the preview mode.
 * @param {boolean} [props.showCard=false] - Whether to wrap the table in a Card component.
 * @param {boolean} [props.autoGenerateColumns=false] - Whether to automatically generate columns from data keys.
 */
export function DataTable({
    data,
    showserial,
    columns: providedColumns,
    loading,
    emptyMessage,
    pageSize,
    showPagination,
    showSearch,
    searchPlaceholder,
    onRowClick,
    className,
    rowClassName,
    headerClassName,
    cellClassName,
    stickyHeader,
    maxHeight,
    previewMode,
    previewTitle,
    previewIcon: PreviewIcon,
    showCard,
    autoGenerateColumns,
}) {
    const [searchTerm, setSearchTerm] = useState("");
    const [sortConfig, setSortConfig] = useState(null);
    const [currentPage, setCurrentPage] = useState(1);

    console.log("DATATABLE SUPER-DEBUG: Component rendered. Data prop length:", data.length);
    console.log("DATATABLE SUPER-DEBUG: Current state -> SearchTerm:", searchTerm, "| SortConfig:", sortConfig, "| CurrentPage:", currentPage);

    // Memoize the column definitions to prevent re-creation on every render
    const columns = useMemo(() => {
        let generatedColumns = providedColumns;
        if (autoGenerateColumns && data.length > 0) {
            const firstRow = data[0];
            generatedColumns = Object.keys(firstRow).map((key) => ({
                key,
                header: key.charAt(0).toUpperCase() + key.slice(1),
                sortable: true,
                accessor: (item) => item[key] || "",
                align: "left",
            }));
        }

        if (showserial) {
            return [
                {
                    key: "srNo",
                    header: "Sr. No.",
                    accessor: (_item, index) => (
                        <span className="text-light-text dark:text-dark-text">
                            {(currentPage - 1) * pageSize + (index + 1)}
                        </span>
                    ),
                    align: "left",
                },
                ...generatedColumns,
            ];
        }
        return generatedColumns;
    }, [autoGenerateColumns, data, providedColumns, showserial, currentPage, pageSize]);

    // Handle sorting logic for a given column key
    const handleSort = (key) => {
        let direction = "asc";
        if (sortConfig && sortConfig.key === key && sortConfig.direction === "asc") {
            direction = "desc";
        }
        setSortConfig({ key, direction });
    };

    // Memoize the filtered and sorted data
    const processedData = useMemo(() => {
        console.log("DATATABLE SUPER-DEBUG: Starting data processing (filtering/sorting).");
        let filteredData = [...data];

        if (searchTerm.trim() !== "") {
            const lowercasedSearchTerm = searchTerm.toLowerCase();
            
            filteredData = filteredData.filter((item) =>
                columns.some((column) => {
                    if (!column.filterable) {
                        return false;
                    }
                    const value = column.accessor ? column.accessor(item) : item[column.key];
                    return value != null && String(value).toLowerCase().includes(lowercasedSearchTerm);
                })
            );
        }

        if (sortConfig !== null) {
            filteredData.sort((a, b) => {
                const aValue = a[sortConfig.key];
                const bValue = b[sortConfig.key];
                if (aValue < bValue) return sortConfig.direction === "asc" ? -1 : 1;
                if (aValue > bValue) return sortConfig.direction === "asc" ? 1 : -1;
                return 0;
            });
        }
        console.log("DATATABLE SUPER-DEBUG: Filtered and sorted data length:", filteredData.length);
        return filteredData;
    }, [data, searchTerm, sortConfig, columns]);

    const totalPages = Math.ceil(processedData.length / pageSize);
    
    // Memoize the paginated data
    const paginatedData = useMemo(() => {
        console.log("DATATABLE SUPER-DEBUG: Starting pagination logic.");
        console.log("DATATABLE SUPER-DEBUG: processedData length:", processedData.length);
        console.log("DATATABLE SUPER-DEBUG: currentPage:", currentPage, "| pageSize:", pageSize);
        const startIndex = (currentPage - 1) * pageSize;
        const endIndex = startIndex + pageSize;
        const slicedData = processedData.slice(startIndex, endIndex);
        console.log("DATATABLE SUPER-DEBUG: Slicing from index", startIndex, "to", endIndex, ". Resulting data length:", slicedData.length);
        return slicedData;
    }, [processedData, currentPage, pageSize]);
    
    // Reset to page 1 whenever the search term or page size changes
    useEffect(() => {
        console.log("DATATABLE SUPER-DEBUG: useEffect triggered. Resetting currentPage to 1.");
        setCurrentPage(1);
    }, [searchTerm, pageSize]);

    // Render the sort icon based on the current sort configuration
    const getSortIcon = (key) => {
        if (!sortConfig || sortConfig.key !== key) return null;
        return sortConfig.direction === "asc" ? <ChevronUp className="w-4 h-4 ml-1" /> : <ChevronDown className="w-4 h-4 ml-1" />;
    };

    // Handle preview mode for empty data
    if (previewMode && (!data || data.length === 0)) {
        return (
            <div className="h-[700px] flex flex-col items-center justify-center bg-light-muted-background dark:bg-dark-muted-background rounded-2xl border border-light-secondary dark:border-dark-secondary p-8 text-center">
                <div className="w-24 h-24 text-light-muted-text dark:text-dark-noisy-text">
                    <PreviewIcon className="w-full h-full" />
                </div>
                <h3 className="text-xl font-semibold mt-4 text-light-text dark:text-dark-text">{previewTitle}</h3>
                <p className="text-light-muted-text dark:text-dark-muted-text mt-2">
                    {emptyMessage || "Select a file and click preview to see the content"}
                </p>
            </div>
        );
    }

    const tableContent = (
        <div className={`w-full ${className}`}>
            {/* Search Input */}
            {showSearch && (
                <div className="p-4 border-b border-light-secondary dark:border-dark-secondary">
                    <Input
                        leftIcon={<Search className="w-5 h-5 text-light-muted-text dark:text-dark-noisy-text" />}
                        type="text"
                        placeholder={searchPlaceholder}
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full"
                    />
                </div>
            )}
            {/* Table Content */}
            <div className="overflow-x-auto" style={{ maxHeight: maxHeight || "none" }}>
                <table className="min-w-full w-full table-auto">
                    {/* Table Header */}
                    <thead className={`${stickyHeader ? "sticky top-0 z-10" : ""} bg-light-background dark:bg-dark-background ${headerClassName}`}>
                        <tr>
                            {columns.map((column, colIndex) => (
                                <th
                                    key={column.key || colIndex}
                                    onClick={() => column.sortable && handleSort(column.key)}
                                    className={`px-6 py-3 text-sm font-semibold text-light-muted-text dark:text-dark-noisy-text uppercase tracking-wider
                                    ${column.sortable ? "cursor-pointer select-none" : ""}
                                    ${column.align === 'center' ? 'text-center' : column.align === 'right' ? 'text-right' : 'text-left'}
                                    ${column.className || ""}`}
                                    style={{ width: column.width || "auto" }}
                                >
                                    <div className={`flex items-center ${column.align === 'center' ? 'justify-center' : column.align === 'right' ? 'justify-end' : ''}`}>
                                        {column.header}
                                        {column.sortable && getSortIcon(column.key)}
                                    </div>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    {/* Table Body */}
                    <tbody className="bg-light-background dark:bg-dark-background divide-y divide-light-secondary dark:divide-dark-secondary">
                        {loading ? (
                            <tr>
                                <td colSpan={columns.length} className="text-center p-8">
                                    <div className="flex justify-center">
                                        <Loader2 className="w-8 h-8 animate-spin text-primary-main"/>
                                    </div>
                                </td>
                            </tr>
                        ) : paginatedData.length === 0 ? (
                            <tr>
                                <td colSpan={columns.length} className="text-center p-8 text-light-muted-text dark:text-dark-noisy-text">
                                    {emptyMessage}
                                </td>
                            </tr>
                        ) : (
                            paginatedData.map((item, rowIndex) => {
                                console.log(`DATATABLE SUPER-DEBUG:   Rendering row ${rowIndex} with data:`, item);
                                return (
                                    <tr
                                        key={item.id || rowIndex}
                                        className={`hover:bg-light-muted-background dark:hover:bg-dark-muted-background transition-colors
                                        ${onRowClick ? "cursor-pointer" : ""}
                                        ${typeof rowClassName === "function" ? rowClassName(item, rowIndex) : rowClassName}`}
                                        onClick={() => onRowClick && onRowClick(item, rowIndex)}
                                    >
                                        {columns.map((column, colIndex) => {
                                            const cellValue = column.accessor ? column.accessor(item, rowIndex) : item[column.key];
                                            console.log(`DATATABLE SUPER-DEBUG:     Rendering cell ${colIndex} (key: ${column.key}) with value:`, cellValue);
                                            return (
                                                <td
                                                    key={colIndex}
                                                    className={`px-6 py-4 text-sm font-medium text-light-text dark:text-dark-text
                                                    ${cellClassName}
                                                    ${column.align === 'center' ? 'text-center' : column.align === 'right' ? 'text-right' : 'text-left'}`}
                                                >
                                                    {cellValue}
                                                </td>
                                            );
                                        })}
                                    </tr>
                                );
                            })
                        )}
                    </tbody>
                </table>
            </div>
            {/* Pagination Controls */}
            {showPagination && totalPages > 1 && (
                <div className="flex items-center justify-between p-4 border-t border-light-secondary dark:border-dark-secondary">
                    <span className="text-sm font-semibold text-light-muted-text dark:text-dark-noisy-text">
                        Page {currentPage} of {totalPages}
                    </span>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                            disabled={currentPage === 1}
                            className="p-2 rounded-md bg-light-muted-background dark:bg-dark-secondary disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <ChevronLeft className="w-5 h-5 text-light-text dark:text-dark-text" />
                        </button>
                        {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
                            <button
                                key={page}
                                onClick={() => setCurrentPage(page)}
                                className={`px-4 py-2 rounded-md text-sm font-medium
                                ${currentPage === page
                                    ? "bg-primary-main text-white"
                                    : "bg-light-muted-background dark:bg-dark-secondary text-light-text dark:text-dark-text hover:bg-light-hover dark:hover:bg-dark-hover"}`
                                }
                            >
                                {page}
                            </button>
                        ))}
                        <button
                            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                            disabled={currentPage === totalPages}
                            className="p-2 rounded-md bg-light-muted-background dark:bg-dark-secondary disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <ChevronRight className="w-5 h-5 text-light-text dark:text-dark-text" />
                        </button>
                    </div>
                </div>
            )}
        </div>
    );

    return showCard ? <Card>{tableContent}</Card> : tableContent;
}

DataTable.propTypes = {
    data: PropTypes.arrayOf(PropTypes.object).isRequired,
    showserial: PropTypes.bool,
    columns: PropTypes.arrayOf(
        PropTypes.shape({
            key: PropTypes.string.isRequired,
            header: PropTypes.oneOfType([PropTypes.string, PropTypes.node]).isRequired,
            accessor: PropTypes.func,
            sortable: PropTypes.bool,
            filterable: PropTypes.bool,
            width: PropTypes.string,
            align: PropTypes.oneOf(['left', 'center', 'right']),
            className: PropTypes.string,
        })
    ),
    loading: PropTypes.bool,
    emptyMessage: PropTypes.string,
    pageSize: PropTypes.number,
    showPagination: PropTypes.bool,
    showSearch: PropTypes.bool,
    searchPlaceholder: PropTypes.string,
    onRowClick: PropTypes.func,
    className: PropTypes.string,
    rowClassName: PropTypes.oneOfType([PropTypes.string, PropTypes.func]),
    headerClassName: PropTypes.string,
    cellClassName: PropTypes.string,
    stickyHeader: PropTypes.bool,
    maxHeight: PropTypes.string,
    previewMode: PropTypes.bool,
    previewTitle: PropTypes.string,
    previewIcon: PropTypes.elementType,
    showCard: PropTypes.bool,
    autoGenerateColumns: PropTypes.bool,
};

DataTable.defaultProps = {
    showserial: false,
    columns: [],
    loading: false,
    emptyMessage: "No data available",
    pageSize: 10,
    showPagination: true,
    showSearch: true,
    searchPlaceholder: "Search...",
    onRowClick: undefined,
    className: "",
    rowClassName: "",
    headerClassName: "",
    cellClassName: "",
    stickyHeader: false,
    maxHeight: undefined,
    previewMode: false,
    previewTitle: "File Preview",
    previewIcon: Table,
    showCard: false,
    autoGenerateColumns: false,
};

export default DataTable;
