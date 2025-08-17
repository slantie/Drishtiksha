// src/components/ui/EmptyState.jsx

import React from "react";

export const EmptyState = ({ icon: Icon, title, message, action }) => {
    return (
        <div className="text-center py-12">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-light-muted-background dark:bg-dark-secondary rounded-full mb-4">
                <Icon className="h-8 w-8 text-light-muted-text dark:text-dark-muted-text" />
            </div>
            <h3 className="text-lg font-semibold">{title}</h3>
            <p className="mt-1 text-sm text-light-muted-text dark:text-dark-muted-text">
                {message}
            </p>
            {action && <div className="mt-6">{action}</div>}
        </div>
    );
};
