import React, { useState, useEffect } from "react";
import {
  Terminal,
  Copy,
  Check,
  AlertCircle,
  Server,
  Database,
  Code,
  Zap,
  Settings,
  Play,
  ExternalLink,
  ChevronRight,
  CheckCircle,
  Clock,
  Folder,
  GitBranch,
  Download,
  RefreshCw,
  Globe,
  Package,
  Rocket,
  CheckSquare,
  Square,
  ChevronLeft,
  Monitor,
  Wifi,
  WifiOff,
  HeartPulse,
} from "lucide-react";
import { showToast } from "../utils/toast";

const projectName = import.meta.env.VITE_PROJECT_NAME || "StarterKit";
const backendUrl = import.meta.env.VITE_BACKEND_URL;

function Setup() {
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState(new Set());
  const [copiedCommands, setCopiedCommands] = useState(new Set());
  const [backendStatus, setBackendStatus] = useState("checking");
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    checkBackendConnection();
  }, []);

  const checkBackendConnection = async () => {
    setIsRefreshing(true);
    try {
      const response = await fetch(`${backendUrl}/api/health`);
      if (response.ok) {
        setBackendStatus("connected");
        setCompletedSteps((prev) => new Set([...prev, 5])); // Last step (verification)
        showToast.success("Backend connected successfully!");
      } else {
        setBackendStatus("error");
      }
    } catch (error) {
      setBackendStatus("disconnected");
      console.error("Error checking backend connection:", error);
    } finally {
      setIsRefreshing(false);
    }
  };

  const copyToClipboard = (text, commandId) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedCommands((prev) => new Set([...prev, commandId]));
      showToast.success("Copied to clipboard!", { duration: 2000 });

      setTimeout(() => {
        setCopiedCommands((prev) => {
          const newSet = new Set(prev);
          newSet.delete(commandId);
          return newSet;
        });
      }, 3000);
    });
  };

  const toggleStepComplete = (stepIndex) => {
    setCompletedSteps((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(stepIndex)) {
        newSet.delete(stepIndex);
        showToast.info(`Step ${stepIndex + 1} marked as incomplete`);
      } else {
        newSet.add(stepIndex);
        showToast.success(`Step ${stepIndex + 1} completed!`);
      }
      return newSet;
    });
  };

  const setupSteps = [
    {
      id: "prerequisites",
      title: "Prerequisites",
      icon: <Settings className="w-6 h-6" />,
      description: "Install required development tools",
      estimatedTime: "15 minutes",
      items: [
        {
          name: "Node.js",
          version: "v18.0.0 or higher",
          command: "node --version",
          link: "https://nodejs.org/",
          description:
            "JavaScript runtime environment for running the application",
          icon: <Globe className="w-5 h-5" />,
        },
        {
          name: "npm",
          version: "v8.0.0 or higher",
          command: "npm --version",
          link: "https://www.npmjs.com/",
          description: "Package manager for installing dependencies",
          icon: <Package className="w-5 h-5" />,
        },
        {
          name: "Git",
          version: "Latest version",
          command: "git --version",
          link: "https://git-scm.com/",
          description: "Version control system for code management",
          icon: <GitBranch className="w-5 h-5" />,
        },
      ],
    },
    {
      id: "clone",
      title: "Clone Repository",
      icon: <GitBranch className="w-6 h-6" />,
      description: "Download the project source code",
      estimatedTime: "2 minutes",
      commands: [
        {
          title: "Clone the repository",
          command: "git clone https://github.com/slantie/StarterKit.git",
          description:
            "",
          category: "setup",
        },
        {
          title: "Navigate to project directory",
          command: "cd StarterKit-main",
          description: "",
          category: "navigation",
        },
      ],
    },
    {
      id: "frontend",
      title: "Frontend Setup",
      icon: <Monitor className="w-6 h-6" />,
      description: "Configure the React frontend application",
      estimatedTime: "5 minutes",
      commands: [
        {
          title: "Navigate to frontend directory",
          command: "cd Frontend",
          description: "Change to the frontend folder",
          category: "navigation",
        },
        {
          title: "Install dependencies",
          command: "npm install",
          description: "Install all required packages and dependencies",
          category: "install",
        },
        {
          title: "Create environment file",
          command: "cp .env.example .env",
          description: "Copy environment template for configuration",
          category: "config",
        },
        {
          title: "Start development server",
          command: "npm run dev",
          description: "Launch the frontend application on localhost:5173",
          category: "start",
        },
      ],
    },
    {
      id: "backend",
      title: "Backend Setup",
      icon: <Server className="w-6 h-6" />,
      description: "Configure the Node.js backend server",
      estimatedTime: "5 minutes",
      commands: [
        {
          title: "Navigate to backend directory",
          command: "cd ../Backend",
          description: "Change to the backend folder",
          category: "navigation",
        },
        {
          title: "Install dependencies",
          command: "npm install",
          description: "Install all required packages and dependencies",
          category: "install",
        },
        {
          title: "Create environment file",
          command: "cp .env.example .env",
          description: "Copy environment template for configuration",
          category: "config",
        },
        {
          title: "Start development server",
          command: "npm run dev",
          description: "Launch the backend server on localhost:3000",
          category: "start",
        },
      ],
    },
    {
      id: "database",
      title: "Database Setup",
      icon: <Database className="w-6 h-6" />,
      description: "Configure database connection and schema",
      estimatedTime: "10 minutes",
      commands: [
        {
          title: "Generate Prisma Client",
          command: "npm run db:generate",
          description: "Generate the Prisma client for database operations",
          category: "setup",
        },
        {
          title: "Push database schema",
          command: "npm run db:push",
          description: "Create database tables based on schema",
          category: "setup",
        },
        {
          title: "Open Prisma Studio (Optional)",
          command: "npm run db:studio",
          description: "Launch database GUI for viewing data",
          category: "optional",
        },
      ],
    },
    {
      id: "verification",
      title: "Verification",
      icon: <CheckCircle className="w-6 h-6" />,
      description: "Test your complete setup",
      estimatedTime: "3 minutes",
      verifications: [
        {
          name: "Frontend Application",
          url: "http://localhost:5173",
          description: "React development server with hot reload",
          status: "active",
        },
        {
          name: "Backend API",
          url: `${backendUrl}/api/health`,
          description: "Express server health check endpoint",
          status: backendStatus,
        },
        {
          name: "Database Connection",
          description: "Prisma database connection established",
          status: "pending",
        },
        {
          name: "Authentication System",
          url: `${backendUrl}/api/auth/login`,
          description: "User authentication endpoints",
          status: "pending",
        },
      ],
    },
  ];

  const envVariables = [
    {
      file: "Frontend/.env",
      icon: <Monitor className="w-5 h-5" />,
      variables: [
        {
          key: "VITE_PROJECT_NAME",
          value: '"Your Project Name"',
          description: "Display name for your application",
        },
        {
          key: "VITE_BACKEND_URL",
          value: '"http://localhost:3000"',
          description: "Backend API URL",
        },
        {
          key: "VITE_NODE_ENV",
          value: '"development"',
          description: "Environment mode",
        },
      ],
    },
    {
      file: "Backend/.env",
      icon: <Server className="w-5 h-5" />,
      variables: [
        { key: "PORT", value: "3000", description: "Server port number" },
        {
          key: "DATABASE_URL",
          value: '"file:./dev.db"',
          description: "SQLite database file path",
        },
        {
          key: "JWT_SECRET",
          value: '"your-super-secret-jwt-key-here"',
          description: "JSON Web Token secret for authentication",
        },
        {
          key: "NODE_ENV",
          value: '"development"',
          description: "Environment mode",
        },
        {
          key: "FRONTEND_URL",
          value: '"http://localhost:5173"',
          description: "Frontend URL for CORS",
        },
      ],
    },
  ];

  const getStatusIcon = (status) => {
    switch (status) {
      case "connected":
      case "active":
        return <HeartPulse className="w-4 h-4 text-green-500" />;
      case "checking":
        return <RefreshCw className="w-4 h-4 text-yellow-500 animate-spin" />;
      case "disconnected":
      case "error":
        return <WifiOff className="w-4 h-4 text-red-500" />;
      default:
        return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case "connected":
      case "active":
        return "text-green-600 dark:text-green-400";
      case "checking":
        return "text-yellow-600 dark:text-yellow-400";
      case "disconnected":
      case "error":
        return "text-red-600 dark:text-red-400";
      default:
        return "text-gray-600 dark:text-gray-400";
    }
  };

  const completionPercentage = Math.round(
    (completedSteps.size / setupSteps.length) * 100
  );

  return (
    <div className="min-h-screen bg-light-background dark:bg-dark-background">
      {/* Enhanced Hero Section */}
      <div className="bg-gradient-to-r from-light-highlight/5 to-light-highlight/10 dark:from-dark-highlight/5 dark:to-dark-highlight/10 border-b border-light-muted-text/20 dark:border-dark-muted-text/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center">
            <div className="flex items-center justify-center mb-6">
              <div className="p-4 bg-light-highlight dark:bg-dark-highlight rounded-2xl shadow-lg">
                <Terminal className="w-12 h-12 text-white" />
              </div>
            </div>

            <h1 className="text-5xl font-bold text-light-text dark:text-dark-text mb-4">
              Setup Guide
            </h1>
            <p className="text-xl text-light-muted-text dark:text-dark-muted-text max-w-3xl mx-auto mb-8">
              Complete setup instructions for{" "}
              <span className="text-light-highlight dark:text-dark-highlight font-semibold">
                {projectName}
              </span>
            </p>

            {/* Progress Bar */}
            <div className="max-w-md mx-auto mb-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-light-text dark:text-dark-text">
                  Progress
                </span>
                <span className="text-sm font-medium text-light-highlight dark:text-dark-highlight">
                  {completionPercentage}%
                </span>
              </div>
              <div className="w-full bg-light-muted-background dark:bg-dark-muted-background rounded-full h-3">
                <div
                  className="bg-gradient-to-r from-light-highlight to-light-highlight/80 dark:from-dark-highlight dark:to-dark-highlight/80 h-3 rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${completionPercentage}%` }}
                ></div>
              </div>
              <p className="text-xs text-light-muted-text dark:text-dark-muted-text mt-2">
                {completedSteps.size} of {setupSteps.length} steps completed
              </p>
            </div>

            {/* Status Indicators */}
            <div className="flex items-center justify-center space-x-8 text-sm">
              <div className="flex items-center space-x-1">
                <HeartPulse className="w-4 h-4 text-green-500" />
                <span className="text-light-muted-text dark:text-dark-muted-text">
                  Frontend:{" "}
                </span>
                <span className="text-green-500">Active</span>
              </div>
              <div className="flex items-center space-x-1">
                {getStatusIcon(backendStatus)}
                <span>Backend: </span>
                <span className={`${getStatusColor(backendStatus)}`}>
                  {backendStatus === "connected"
                    ? "Connected"
                    : backendStatus === "checking"
                    ? "Checking..."
                    : "Disconnected"}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid lg:grid-cols-4 gap-8">
          {/* Enhanced Progress Sidebar */}
          <div className="lg:col-span-1">
            <div className="sticky top-8 space-y-6">
              {/* Step Progress */}
              <div className="bg-light-background dark:bg-dark-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-bold text-light-text dark:text-dark-text mb-4 flex items-center">
                  <CheckSquare className="w-5 h-5 mr-2 text-light-highlight dark:text-dark-highlight" />
                  Steps
                </h3>
                <div className="space-y-3">
                  {setupSteps.map((step, index) => (
                    <div
                      key={step.id}
                      className={`flex items-center space-x-3 p-3 rounded-lg transition-all duration-300 cursor-pointer border ${
                        currentStep === index
                          ? "bg-light-highlight/10 dark:bg-dark-highlight/10 border-light-highlight/30 dark:border-dark-highlight/30 shadow-md"
                          : "border-transparent hover:bg-light-muted-background/50 dark:hover:bg-dark-muted-background/50"
                      }`}
                      onClick={() => setCurrentStep(index)}
                    >
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleStepComplete(index);
                        }}
                        className={`p-1 rounded transition-colors ${
                          completedSteps.has(index)
                            ? "text-green-600 dark:text-green-400 hover:text-green-700 dark:hover:text-green-300"
                            : "text-light-muted-text dark:text-dark-muted-text hover:text-light-highlight dark:hover:text-dark-highlight"
                        }`}
                      >
                        {completedSteps.has(index) ? (
                          <CheckSquare className="w-5 h-5" />
                        ) : (
                          <Square className="w-5 h-5" />
                        )}
                      </button>

                      <div
                        className={`p-2 rounded-lg ${
                          completedSteps.has(index)
                            ? "bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400"
                            : currentStep === index
                            ? "bg-light-highlight/20 dark:bg-dark-highlight/20 text-light-highlight dark:text-dark-highlight"
                            : "bg-light-muted-background dark:bg-dark-muted-background text-light-muted-text dark:text-dark-muted-text"
                        }`}
                      >
                        {step.icon}
                      </div>

                      <div className="flex-1 min-w-0">
                        <p
                          className={`text-sm font-medium ${
                            completedSteps.has(index)
                              ? "text-green-600 dark:text-green-400"
                              : "text-light-text dark:text-dark-text"
                          }`}
                        >
                          {step.title}
                        </p>
                        <p className="text-xs text-light-muted-text dark:text-dark-muted-text truncate">
                          {step.estimatedTime}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Quick Actions */}
              <div className="bg-light-background dark:bg-dark-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-bold text-light-text dark:text-dark-text mb-4 flex items-center">
                  <Rocket className="w-5 h-5 mr-2 text-light-highlight dark:text-dark-highlight" />
                  Quick Links
                </h3>
                <div className="space-y-3">
                  <a
                    href="http://localhost:5173"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center space-x-3 p-3 bg-light-muted-background dark:bg-dark-muted-background rounded-lg hover:bg-light-highlight/10 dark:hover:bg-dark-highlight/10 transition-colors group"
                  >
                    <Monitor className="w-5 h-5 text-light-highlight dark:text-dark-highlight" />
                    <div className="flex-1">
                      <p className="text-sm font-medium text-light-text dark:text-dark-text group-hover:text-light-highlight dark:group-hover:text-dark-highlight">
                        Frontend
                      </p>
                      <p className="text-xs text-light-muted-text dark:text-dark-muted-text">
                        :5173
                      </p>
                    </div>
                    <ExternalLink className="w-4 h-4 text-light-muted-text dark:text-dark-muted-text group-hover:text-light-highlight dark:group-hover:text-dark-highlight" />
                  </a>

                  <a
                    href={`${backendUrl}/api/health`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center space-x-3 p-3 bg-light-muted-background dark:bg-dark-muted-background rounded-lg hover:bg-light-highlight/10 dark:hover:bg-dark-highlight/10 transition-colors group"
                  >
                    <Server className="w-5 h-5 text-light-highlight dark:text-dark-highlight" />
                    <div className="flex-1">
                      <p className="text-sm font-medium text-light-text dark:text-dark-text group-hover:text-light-highlight dark:group-hover:text-dark-highlight">
                        Backend
                      </p>
                      <p className="text-xs text-light-muted-text dark:text-dark-muted-text">
                        :3000
                      </p>
                    </div>
                    <ExternalLink className="w-4 h-4 text-light-muted-text dark:text-dark-muted-text group-hover:text-light-highlight dark:group-hover:text-dark-highlight" />
                  </a>

                  <button
                    onClick={checkBackendConnection}
                    disabled={isRefreshing}
                    className="w-full flex items-center space-x-3 p-3 bg-light-muted-background dark:bg-dark-muted-background rounded-lg hover:bg-light-highlight/10 dark:hover:bg-dark-highlight/10 transition-colors group disabled:opacity-50"
                  >
                    <RefreshCw
                      className={`w-5 h-5 text-light-highlight dark:text-dark-highlight ${
                        isRefreshing ? "animate-spin" : ""
                      }`}
                    />
                    <div className="flex-1 text-left">
                      <p className="text-sm font-medium text-light-text dark:text-dark-text group-hover:text-light-highlight dark:group-hover:text-dark-highlight">
                        Refresh Status
                      </p>
                      <p className="text-xs text-light-muted-text dark:text-dark-muted-text">
                        Check connections
                      </p>
                    </div>
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Enhanced Main Content */}
          <div className="lg:col-span-3 space-y-8">
            {/* Current Step */}
            <div className="bg-light-background dark:bg-dark-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-xl shadow-lg overflow-hidden">
              <div className="bg-gradient-to-r from-light-highlight/5 to-light-highlight/10 dark:from-dark-highlight/5 dark:to-dark-highlight/10 p-6 border-b border-light-muted-text/20 dark:border-dark-muted-text/20">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="p-3 bg-light-highlight dark:bg-dark-highlight rounded-xl shadow-lg">
                      <div className="text-white">
                        {setupSteps[currentStep]?.icon}
                      </div>
                    </div>
                    <div>
                      <h2 className="text-2xl font-bold text-light-text dark:text-dark-text">
                        {setupSteps[currentStep]?.title}
                      </h2>
                      <p className="text-light-muted-text dark:text-dark-muted-text">
                        {setupSteps[currentStep]?.description}
                      </p>
                      <div className="flex items-center space-x-4 mt-2">
                        <span className="text-sm text-light-muted-text dark:text-dark-muted-text flex items-center">
                          <Clock className="w-4 h-4 mr-1" />
                          {setupSteps[currentStep]?.estimatedTime}
                        </span>
                        <span className="text-sm text-light-muted-text dark:text-dark-muted-text">
                          Step {currentStep + 1} of {setupSteps.length}
                        </span>
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={() => toggleStepComplete(currentStep)}
                    className={`px-6 py-3 rounded-lg font-medium text-sm transition-all duration-300 flex items-center space-x-2 ${
                      completedSteps.has(currentStep)
                        ? "bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 border border-green-200 dark:border-green-800"
                        : "bg-light-highlight dark:bg-dark-highlight text-white hover:opacity-90 shadow-lg"
                    }`}
                  >
                    {completedSteps.has(currentStep) ? (
                      <>
                        <Check className="w-4 h-4" />
                        <span>Completed</span>
                      </>
                    ) : (
                      <>
                        <Square className="w-4 h-4" />
                        <span>Mark Complete</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              <div className="p-6">
                {renderStepContent(
                  setupSteps[currentStep],
                  copiedCommands,
                  copyToClipboard,
                  backendStatus
                )}
              </div>
            </div>

            {/* Enhanced Navigation */}
            <div className="flex justify-between items-center">
              <button
                onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
                disabled={currentStep === 0}
                className="flex items-center space-x-2 px-6 py-3 bg-light-muted-background dark:bg-dark-muted-background rounded-lg font-medium text-sm text-light-text dark:text-dark-text transition-all duration-300 hover:bg-light-muted-background/80 dark:hover:bg-dark-muted-background/80 disabled:opacity-50 disabled:cursor-not-allowed border border-light-muted-text/20 dark:border-dark-muted-text/20"
              >
                <ChevronLeft className="w-4 h-4" />
                <span>Previous</span>
              </button>

              <div className="flex items-center space-x-2">
                {setupSteps.map((_, index) => (
                  <button
                    key={index}
                    onClick={() => setCurrentStep(index)}
                    className={`w-3 h-3 rounded-full transition-all duration-300 ${
                      currentStep === index
                        ? "bg-light-highlight dark:bg-dark-highlight scale-125"
                        : completedSteps.has(index)
                        ? "bg-green-400 dark:bg-green-500"
                        : "bg-light-muted-background dark:bg-dark-muted-background hover:bg-light-muted-text/30 dark:hover:bg-dark-muted-text/30"
                    }`}
                  />
                ))}
              </div>

              <button
                onClick={() =>
                  setCurrentStep(
                    Math.min(setupSteps.length - 1, currentStep + 1)
                  )
                }
                disabled={currentStep === setupSteps.length - 1}
                className="flex items-center space-x-2 px-6 py-3 bg-light-highlight dark:bg-dark-highlight text-white rounded-lg font-medium text-sm transition-all duration-300 hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg"
              >
                <span>Next</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            {/* Enhanced Environment Variables Section */}
            <div className="space-y-6">
              <div className="text-center">
                <h3 className="text-2xl font-bold text-light-text dark:text-dark-text mb-2">
                  Environment Configuration
                </h3>
                <p className="text-light-muted-text dark:text-dark-muted-text">
                  Configure your application settings with these environment
                  variables
                </p>
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                {envVariables.map((envFile, index) => (
                  <div
                    key={index}
                    className="bg-light-background dark:bg-dark-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-xl shadow-lg overflow-hidden"
                  >
                    <div className="bg-gradient-to-r from-light-highlight/5 to-light-highlight/10 dark:from-dark-highlight/5 dark:to-dark-highlight/10 p-4 border-b border-light-muted-text/20 dark:border-dark-muted-text/20">
                      <div className="flex items-center space-x-3">
                        <div className="p-2 bg-light-highlight dark:bg-dark-highlight rounded-lg">
                          <div className="text-white">{envFile.icon}</div>
                        </div>
                        <div>
                          <h4 className="font-bold text-lg text-light-text dark:text-dark-text">
                            {envFile.file}
                          </h4>
                          <p className="text-sm text-light-muted-text dark:text-dark-muted-text">
                            {envFile.variables.length} variables
                          </p>
                        </div>
                      </div>
                    </div>
                    <div className="p-4">
                      <div className="space-y-4">
                        {envFile.variables.map((variable, varIndex) => (
                          <div
                            key={varIndex}
                            className="p-4 bg-light-muted-background dark:bg-dark-muted-background rounded-lg border border-light-muted-text/10 dark:border-dark-muted-text/10"
                          >
                            <div className="flex items-center justify-between mb-2">
                              <code className="text-sm font-bold text-light-highlight dark:text-dark-highlight bg-light-highlight/10 dark:bg-dark-highlight/10 px-2 py-1 rounded">
                                {variable.key}
                              </code>
                              <button
                                onClick={() =>
                                  copyToClipboard(
                                    `${variable.key}=${variable.value}`,
                                    `env-${index}-${varIndex}`
                                  )
                                }
                                className="p-2 hover:bg-light-background dark:hover:bg-dark-background rounded-lg transition-colors border border-light-muted-text/20 dark:border-dark-muted-text/20 group"
                              >
                                {copiedCommands.has(
                                  `env-${index}-${varIndex}`
                                ) ? (
                                  <Check className="w-4 h-4 text-green-500" />
                                ) : (
                                  <Copy className="w-4 h-4 text-light-muted-text dark:text-dark-muted-text group-hover:text-light-highlight dark:group-hover:text-dark-highlight" />
                                )}
                              </button>
                            </div>
                            <code className="text-xs text-light-muted-text dark:text-dark-muted-text block mb-2 bg-light-background dark:bg-dark-background px-2 py-1 rounded">
                              = {variable.value}
                            </code>
                            <p className="text-xs text-light-muted-text dark:text-dark-muted-text">
                              {variable.description}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Enhanced helper function to render step content
function renderStepContent(step, copiedCommands, copyToClipboard) {
  if (!step) return null;

  const getCategoryIcon = (category) => {
    switch (category) {
      case "navigation":
        return <Folder className="w-4 h-4" />;
      case "install":
        return <Package className="w-4 h-4" />;
      case "config":
        return <Settings className="w-4 h-4" />;
      case "start":
        return <Rocket className="w-4 h-4" />;
      case "setup":
        return <Zap className="w-4 h-4" />;
      case "optional":
        return <Play className="w-4 h-4" />;
      default:
        return <Terminal className="w-4 h-4" />;
    }
  };

  const getCategoryColor = (category) => {
    switch (category) {
      case "navigation":
        return "bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400";
      case "install":
        return "bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400";
      case "config":
        return "bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400";
      case "start":
        return "bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400";
      case "setup":
        return "bg-light-highlight/10 dark:bg-dark-highlight/10 text-light-highlight dark:text-dark-highlight";
      case "optional":
        return "bg-gray-100 dark:bg-gray-900/30 text-gray-600 dark:text-gray-400";
      default:
        return "bg-light-muted-background dark:bg-dark-muted-background text-light-text dark:text-dark-text";
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case "connected":
      case "active":
        return <Wifi className="w-4 h-4 text-green-500" />;
      case "checking":
        return <RefreshCw className="w-4 h-4 text-yellow-500 animate-spin" />;
      case "disconnected":
      case "error":
        return <WifiOff className="w-4 h-4 text-red-500" />;
      default:
        return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case "connected":
      case "active":
        return "text-green-600 dark:text-green-400";
      case "checking":
        return "text-yellow-600 dark:text-yellow-400";
      case "disconnected":
      case "error":
        return "text-red-600 dark:text-red-400";
      default:
        return "text-gray-600 dark:text-gray-400";
    }
  };

  switch (step.id) {
    case "prerequisites":
      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 border border-blue-200 dark:border-blue-800 rounded-xl p-6">
            <div className="flex items-start space-x-4">
              <div className="p-2 bg-blue-500 rounded-lg">
                <AlertCircle className="w-5 h-5 text-white" />
              </div>
              <div>
                <h4 className="font-bold text-blue-800 dark:text-blue-200 text-lg">
                  Prerequisites Required
                </h4>
                <p className="text-blue-700 dark:text-blue-300 mt-1">
                  Install these tools before proceeding with the setup process.
                </p>
              </div>
            </div>
          </div>

          <div className="grid gap-6">
            {step.items.map((item, index) => (
              <div
                key={index}
                className="bg-light-muted-background dark:bg-dark-muted-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-xl p-6 hover:shadow-lg transition-shadow"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-start space-x-4 flex-1">
                    <div className="p-3 bg-light-highlight dark:bg-dark-highlight rounded-lg">
                      <div className="text-white">{item.icon}</div>
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <h4 className="font-bold text-lg text-light-text dark:text-dark-text">
                          {item.name}
                        </h4>
                        <span className="px-3 py-1 bg-light-highlight/10 dark:bg-dark-highlight/10 text-light-highlight dark:text-dark-highlight text-sm font-medium rounded-full">
                          {item.version}
                        </span>
                      </div>
                      <p className="text-light-muted-text dark:text-dark-muted-text mb-3">
                        {item.description}
                      </p>
                      <code className="bg-light-background dark:bg-dark-background px-3 py-2 rounded-lg text-sm font-mono border border-light-muted-text/20 dark:border-dark-muted-text/20 inline-block">
                        {item.command}
                      </code>
                    </div>
                  </div>
                  <a
                    href={item.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center space-x-2 px-6 py-3 bg-light-highlight dark:bg-dark-highlight text-white rounded-lg font-medium text-sm hover:opacity-90 transition-opacity shadow-lg"
                  >
                    <Download className="w-4 h-4" />
                    <span>Download</span>
                  </a>
                </div>
              </div>
            ))}
          </div>
        </div>
      );

    case "verification":
      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 border border-green-200 dark:border-green-800 rounded-xl p-6">
            <div className="flex items-start space-x-4">
              <div className="p-2 bg-green-500 rounded-lg">
                <CheckCircle className="w-5 h-5 text-white" />
              </div>
              <div>
                <h4 className="font-bold text-green-800 dark:text-green-200 text-lg">
                  Almost Complete!
                </h4>
                <p className="text-green-700 dark:text-green-300 mt-1">
                  Verify that all components are running correctly before using
                  the application.
                </p>
              </div>
            </div>
          </div>

          <div className="grid gap-4">
            {step.verifications.map((verification, index) => (
              <div
                key={index}
                className="bg-light-muted-background dark:bg-dark-muted-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-xl p-6 hover:shadow-lg transition-shadow"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4 flex-1">
                    <div className="flex items-center space-x-3">
                      {getStatusIcon(verification.status)}
                      <div>
                        <h4 className="font-bold text-light-text dark:text-dark-text">
                          {verification.name}
                        </h4>
                        <p className="text-sm text-light-muted-text dark:text-dark-muted-text">
                          {verification.description}
                        </p>
                        {verification.url && (
                          <code className="text-xs bg-light-background dark:bg-dark-background px-2 py-1 rounded mt-2 inline-block font-mono border border-light-muted-text/20 dark:border-dark-muted-text/20">
                            {verification.url}
                          </code>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <span
                      className={`text-sm font-medium ${getStatusColor(
                        verification.status
                      )}`}
                    >
                      {verification.status === "connected" ||
                      verification.status === "active"
                        ? "Online"
                        : verification.status === "checking"
                        ? "Checking..."
                        : verification.status === "pending"
                        ? "Pending"
                        : "Offline"}
                    </span>
                    {verification.url && (
                      <a
                        href={verification.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center space-x-2 px-4 py-2 bg-light-highlight dark:bg-dark-highlight text-white rounded-lg font-medium text-sm hover:opacity-90 transition-opacity"
                      >
                        <Play className="w-4 h-4" />
                        <span>Test</span>
                      </a>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      );

    default:
      return (
        <div className="space-y-6">
          {step.commands?.map((cmd, index) => (
            <div
              key={index}
              className="bg-gradient-to-r from-gray-900 to-gray-800 border border-gray-700 rounded-xl p-6 shadow-lg"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div
                    className={`p-2 rounded-lg ${getCategoryColor(
                      cmd.category
                    )}`}
                  >
                    {getCategoryIcon(cmd.category)}
                  </div>
                  <div>
                    <h4 className="text-white font-bold">{cmd.title}</h4>
                    <p className="text-gray-400 text-sm">{cmd.description}</p>
                  </div>
                </div>
                <button
                  onClick={() =>
                    copyToClipboard(cmd.command, `${step.id}-${index}`)
                  }
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
                    copiedCommands.has(`${step.id}-${index}`)
                      ? "bg-green-600 text-white"
                      : "bg-gray-700 hover:bg-gray-600 text-gray-300"
                  }`}
                >
                  {copiedCommands.has(`${step.id}-${index}`) ? (
                    <>
                      <Check className="w-4 h-4" />
                      <span>Copied!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-4 h-4" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              </div>
              <div className="bg-gray-950 rounded-lg p-4 border border-gray-800">
                <code className="text-green-400 font-mono text-sm">
                  $ {cmd.command}
                </code>
              </div>
            </div>
          ))}
        </div>
      );
  }
}

export default Setup;
