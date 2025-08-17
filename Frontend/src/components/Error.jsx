// src/components/Error.jsx

import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { AlertTriangle, Home, RefreshCw, ArrowLeft } from "lucide-react";
import { Button } from "./ui/Button"; // REFACTOR: Using our standardized Button component.

function Error() {
    const navigate = useNavigate();

    const handleRefresh = () => {
        window.location.reload();
    };

    const goBack = () => {
        navigate(-1); // REFACTOR: Using navigate(-1) is a safer, more React-friendly way to go back.
    };

    // REFACTOR: The entire component is redesigned for a cleaner, more modern look.
    // It now uses a Card as a container and our primary/secondary button pattern.
    return (
        <div className="flex items-center justify-center py-12">
            <div className="text-center space-y-8 max-w-lg">
                <div className="flex justify-center">
                    <div className="flex items-center justify-center w-24 h-24 bg-red-100 dark:bg-red-900/30 rounded-full">
                        <AlertTriangle className="w-12 h-12 text-red-500" />
                    </div>
                </div>

                <div className="space-y-2">
                    <h1 className="text-6xl font-bold tracking-tighter text-light-text dark:text-dark-text">
                        404
                    </h1>
                    <h2 className="text-2xl font-semibold text-light-text dark:text-dark-text">
                        Page Not Found
                    </h2>
                    <p className="text-lg text-light-muted-text dark:text-dark-muted-text leading-relaxed">
                        Sorry, we couldn't find the page you're looking for. It
                        might have been moved or deleted.
                    </p>
                </div>

                <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                    <Button asChild>
                        <Link to="/">
                            <Home className="w-4 h-4 mr-2" /> Go Home
                        </Link>
                    </Button>
                    <Button variant="outline" onClick={goBack}>
                        <ArrowLeft className="w-4 h-4 mr-2" /> Go Back
                    </Button>
                    <Button variant="ghost" onClick={handleRefresh}>
                        <RefreshCw className="w-4 h-4 mr-2" /> Refresh
                    </Button>
                </div>
            </div>
        </div>
    );
}

export default Error;
