// src/components/layout/PageHeader.jsx

// NEW COMPONENT: Standardizes the header for all main pages.
import React from "react";
import { Card } from "../ui/Card";

export const PageHeader = ({ title, description, actions }) => {
    return (
        <Card className="p-4 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div className="space-y-1">
                <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
                {description && (
                    <p className="text-sm text-light-muted-text dark:text-dark-muted-text">
                        {description}
                    </p>
                )}
            </div>
            {actions && (
                <div className="flex items-center gap-2">{actions}</div>
            )}
        </Card>
    );
};
