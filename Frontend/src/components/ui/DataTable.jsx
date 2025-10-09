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
    // Only sort if column is sortable
    const column = columns.find((c) => c.key === key);
    if (!column || !column.sortable) return;

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
          // Only filter if column is filterable (or if filterable is undefined/true)
          if (column.filterable === false) return false;
          const value = column.accessor
            ? column.accessor(item)
            : item[column.key];
          if (typeof value === "string" || typeof value === "number") {
            return String(value).toLowerCase().includes(lowercasedSearchTerm);
          }
          return false;
        })
      );
    }
    if (!disableInternalSorting && sortConfig !== null) {
      filteredData.sort((a, b) => {
        const column = columns.find((c) => c.key === sortConfig.key);
        if (!column) return 0; // Should not happen if sortConfig.key is valid

        const aValue = column.accessor ? column.accessor(a) : a[sortConfig.key];
        const bValue = column.accessor ? column.accessor(b) : b[sortConfig.key];

        // Handle null/undefined values gracefully
        if (aValue === null || typeof aValue === "undefined")
          return sortConfig.direction === "asc" ? 1 : -1;
        if (bValue === null || typeof bValue === "undefined")
          return sortConfig.direction === "asc" ? -1 : 1;

        if (aValue < bValue) return sortConfig.direction === "asc" ? -1 : 1;
        if (aValue > bValue) return sortConfig.direction === "asc" ? 1 : -1;
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

  // Reset to first page when search term changes
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

  // New pagination logic to show a limited number of page buttons
  const getPageButtons = () => {
    const pageButtons = [];
    const maxButtons = 5; // Max number of page buttons to show (e.g., 1 ... 5 6 [7] 8 9 ... 15)

    if (totalPages <= maxButtons) {
      for (let i = 1; i <= totalPages; i++) {
        pageButtons.push(i);
      }
    } else {
      pageButtons.push(1); // Always show first page
      if (currentPage > 2) pageButtons.push("..."); // Ellipsis after first page

      const start = Math.max(2, currentPage - 1);
      const end = Math.min(totalPages - 1, currentPage + 1);

      for (let i = start; i <= end; i++) {
        if (i !== 1 && i !== totalPages) pageButtons.push(i);
      }

      if (currentPage < totalPages - 1) pageButtons.push("..."); // Ellipsis before last page
      if (totalPages > 1) pageButtons.push(totalPages); // Always show last page
    }
    return Array.from(new Set(pageButtons)); // Remove duplicates from ...
  };

  return (
    <Card>
      {(title || showSearch || headerActions) && (
        <CardHeader className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b-0 pb-4">
          <div>
            {title && <CardTitle>{title}</CardTitle>}
            {description && (
              <CardDescription className="mt-1">{description}</CardDescription>
            )}
          </div>
          <div className="flex items-center flex-wrap gap-2 md:w-[400px] w-full justify-end">
            {" "}
            {/* Added flex-wrap and justify-end */}
            {showSearch && (
              <div className="flex-grow md:flex-grow-0">
                {" "}
                {/* Allow search to grow on small screens */}
                <Input
                  leftIcon={<Search className="h-5 w-5" />} // Consistent icon size
                  type="text"
                  placeholder={searchPlaceholder}
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="rounded-full"
                  rightIcon={<></>} // Explicitly null
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
                    onClick={() => handleSort(column.key)}
                    className={cn(
                      "px-6 py-3 text-left text-xs font-semibold text-light-muted-text dark:text-dark-muted-text uppercase tracking-wider",
                      column.sortable && // Only apply cursor if sortable
                        "cursor-pointer hover:bg-light-hover dark:hover:bg-dark-hover transition-colors"
                    )}
                  >
                    <div className="flex items-center">
                      {column.header}
                      {column.sortable && // Only render icon if sortable
                        getSortIcon(column.key)}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-light-secondary dark:divide-dark-secondary">
              {loading ? (
                Array.from({ length: pageSize }).map((_, i) => (
                  <SkeletonRow key={i} cells={columns.length} />
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
                    key={item.id || `row-${index}`} // Ensure unique key
                    onClick={() => onRowClick && onRowClick(item)}
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
        <CardFooter className="justify-between flex-wrap gap-2">
          {" "}
          {/* Added flex-wrap */}
          <div className="text-sm text-light-muted-text dark:text-dark-muted-text">
            Page {currentPage} of {totalPages}
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              className="rounded-full"
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
            >
              <ChevronLeft className="h-4 w-4 mr-1" /> Previous
            </Button>

            {/* Updated Page number buttons logic */}
            {getPageButtons().map((p, index) =>
              p === "..." ? (
                <span
                  key={`ellipsis-${index}`}
                  className="px-2 text-sm text-light-muted-text dark:text-dark-muted-text"
                >
                  ...
                </span>
              ) : (
                <Button
                  key={`page-${p}`}
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
              )
            )}

            <Button
              variant="outline"
              size="sm"
              className="rounded-full"
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
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

DataTable.propTypes = {
  data: PropTypes.array.isRequired,
  columns: PropTypes.arrayOf(
    PropTypes.shape({
      key: PropTypes.string.isRequired,
      header: PropTypes.node.isRequired,
      accessor: PropTypes.func, // Optional function to access nested data
      render: PropTypes.func, // Optional function for custom cell rendering
      sortable: PropTypes.bool,
      filterable: PropTypes.bool, // New prop to control if column is searchable
    })
  ).isRequired,
  loading: PropTypes.bool,
  onRowClick: PropTypes.func,
  emptyState: PropTypes.shape({
    icon: PropTypes.elementType.isRequired,
    title: PropTypes.string.isRequired,
    message: PropTypes.string.isRequired,
    action: PropTypes.node,
  }),
  pageSize: PropTypes.number,
  showPagination: PropTypes.bool,
  showSearch: PropTypes.bool,
  searchPlaceholder: PropTypes.string,
  title: PropTypes.node,
  description: PropTypes.node,
  headerActions: PropTypes.node,
  disableInternalSorting: PropTypes.bool,
};
