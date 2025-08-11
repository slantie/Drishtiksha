import { AlertTriangle, Home, RefreshCw, ArrowLeft } from "lucide-react";

function Error() {
  const handleRefresh = () => {
    window.location.reload();
  };

  const goBack = () => {
    window.history.back();
  };

  return (
    <div className="flex items-center justify-center py-8 ">
      <div className="text-center space-y-8">
        <div className="relative">
          <div className="w-24 h-24 mx-auto bg-light-highlight dark:bg-dark-highlight rounded-full flex items-center justify-center shadow-2xl">
            <AlertTriangle className="w-12 h-12 text-white" />
          </div>
        </div>

        <div className="space-y-4">
          <h1 className="text-6xl font-bold text-light-text dark:text-dark-text">
            404
          </h1>
          <h2 className="text-2xl font-bold text-light-text dark:text-dark-text">
            Page Not Found
          </h2>
          <p className="text-lg text-light-muted-text dark:text-dark-muted-text max-w-md mx-auto leading-relaxed">
            Sorry, we couldn't find the page you're looking for. It might have
            been moved, deleted, or you entered the wrong URL.
          </p>
        </div>

        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">

          <button
            onClick={() => window.location.href = "/"}
            className="group bg-light-muted-background dark:bg-dark-muted-background border-2 border-light-muted-text/20 dark:border-dark-muted-text/20 hover:border-light-highlight dark:hover:border-dark-highlight font-semibold py-3 px-8 rounded-xl transition-all duration-300 hover:shadow-lg flex items-center space-x-2 text-light-text dark:text-dark-text"
          >
            <Home className="w-5 h-5 group-hover:scale-110 transition-transform duration-300" />
            <span>Home</span>
          </button>

          <button
            onClick={handleRefresh}
            className="group bg-light-muted-background dark:bg-dark-muted-background border-2 border-light-muted-text/20 dark:border-dark-muted-text/20 hover:border-light-highlight dark:hover:border-dark-highlight font-semibold py-3 px-8 rounded-xl transition-all duration-300 hover:shadow-lg flex items-center space-x-2 text-light-text dark:text-dark-text"
          >
            <RefreshCw className="w-5 h-5 group-hover:rotate-180 transition-transform duration-500" />
            <span>Refresh</span>
          </button>

          <button
            onClick={goBack}
            className="group bg-light-muted-background dark:bg-dark-muted-background border-2 border-light-muted-text/20 dark:border-dark-muted-text/20 hover:border-light-highlight dark:hover:border-dark-highlight font-semibold py-3 px-8 rounded-xl transition-all duration-300 hover:shadow-lg flex items-center space-x-2 text-light-text dark:text-dark-text"
          >
            <ArrowLeft className="w-5 h-5 group-hover:-translate-x-1 transition-transform duration-300" />
            <span>Back</span>
          </button>
        </div>
      </div>
    </div>
  );
}

export default Error;
