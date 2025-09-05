// src/components/Error.jsx

import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { AlertTriangle, Home, RefreshCw, ArrowLeft } from "lucide-react";
import { Button } from "./ui/Button";

function Error() {
  const navigate = useNavigate();

  const handleRefresh = () => window.location.reload();
  const goBack = () => navigate(-1);

  return (
    <div className="flex min-h-[calc(100vh-200px)] items-center justify-center py-12 px-4">
      {" "}
      {/* Added min-h and px-4 */}
      <div className="text-center space-y-8 max-w-lg">
        <div className="flex justify-center">
          <div className="flex items-center justify-center w-24 h-24 bg-red-100 dark:bg-red-900/30 rounded-full">
            <AlertTriangle className="w-12 h-12 text-red-500" />
          </div>
        </div>
        <div className="space-y-2">
          <h1 className="text-6xl font-bold tracking-tighter">404</h1>
          <h2 className="text-2xl font-semibold">Page Not Found</h2>
          <p className="text-lg text-light-muted-text dark:text-dark-muted-text leading-relaxed">
            Sorry, we couldn't find the page you're looking for. It might have
            been moved or deleted.
          </p>
        </div>
        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
          <Button asChild>
            <Link to="/">
              <Home className="w-5 h-5 mr-2" /> Go Home
            </Link>
          </Button>
          <Button variant="outline" onClick={goBack}>
            <ArrowLeft className="w-5 h-5 mr-2" /> Go Back
          </Button>
          <Button
            variant="ghost"
            onClick={handleRefresh}
            aria-label="Refresh page"
          >
            <RefreshCw className="w-5 h-5 mr-2" /> Refresh
          </Button>
        </div>
      </div>
    </div>
  );
}

export default Error;
