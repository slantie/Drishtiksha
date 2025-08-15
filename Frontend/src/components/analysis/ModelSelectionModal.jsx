// src/components/analysis/ModelSelectionModal.jsx

import React, { useState } from "react";
import {
    X,
    Brain,
    Play,
    Clock,
    Info,
    CheckCircle2,
    Loader2,
    Zap,
    Search,
    BarChart3,
    Layers,
    TrendingUp,
} from "lucide-react";
import { Button } from "../ui/Button";
import { Card } from "../ui/Card";
import {
    ANALYSIS_TYPES,
    MODEL_TYPES,
    ANALYSIS_TYPE_INFO,
    MODEL_INFO,
} from "../../constants/apiEndpoints.js";
import {
    useVideoAnalysis,
    useModelStatusQuery,
} from "../../hooks/useAnalysisQuery.js";

const ModelSelectionModal = ({ isOpen, onClose, videoId, onAnalysisStart }) => {
    const [selectedAnalysisType, setSelectedAnalysisType] = useState(
        ANALYSIS_TYPES.QUICK
    );
    const [selectedModels, setSelectedModels] = useState([
        MODEL_TYPES.SIGLIP_LSTM_V1,
    ]);
    const [isStarting, setIsStarting] = useState(false);

    const { modelStatus, isModelStatusLoading } = useModelStatusQuery();
    const { createAnalysis, createMultipleAnalyses } =
        useVideoAnalysis(videoId);

    if (!isOpen) return null;

    const handleModelToggle = (model) => {
        setSelectedModels((prev) => {
            if (prev.includes(model)) {
                return prev.filter((m) => m !== model);
            } else {
                return [...prev, model];
            }
        });
    };

    const handleStartAnalysis = async () => {
        if (selectedModels.length === 0) return;

        setIsStarting(true);
        try {
            if (selectedModels.length === 1) {
                await createAnalysis(selectedAnalysisType, selectedModels[0]);
            } else {
                const analysisConfigs = selectedModels.map((model) => ({
                    type: selectedAnalysisType,
                    model,
                }));
                await createMultipleAnalyses(analysisConfigs);
            }
            onAnalysisStart?.();
            onClose();
        } catch (error) {
            console.error("Failed to start analysis:", error);
        } finally {
            setIsStarting(false);
        }
    };

    const getAnalysisIcon = (type) => {
        switch (type) {
            case ANALYSIS_TYPES.QUICK:
                return <Zap className="h-5 w-5" />;
            case ANALYSIS_TYPES.DETAILED:
                return <Search className="h-5 w-5" />;
            case ANALYSIS_TYPES.FRAMES:
                return <Layers className="h-5 w-5" />;
            case ANALYSIS_TYPES.VISUALIZE:
                return <TrendingUp className="h-5 w-5" />;
            default:
                return <BarChart3 className="h-5 w-5" />;
        }
    };

    const isModelAvailable = (model) => {
        return modelStatus?.availableModels?.includes(model) !== false;
    };

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-white dark:bg-gray-900 rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
                    <div className="flex items-center gap-3">
                        <Brain className="h-6 w-6 text-blue-600" />
                        <h2 className="text-2xl font-bold">
                            Start New Analysis
                        </h2>
                    </div>
                    <Button
                        onClick={onClose}
                        variant="ghost"
                        size="sm"
                        className="p-2"
                    >
                        <X className="h-5 w-5" />
                    </Button>
                </div>

                <div className="p-6 space-y-6">
                    {/* Model Status */}
                    {isModelStatusLoading ? (
                        <Card>
                            <div className="flex items-center gap-3 p-4">
                                <Loader2 className="h-5 w-5 animate-spin" />
                                <span>Checking model availability...</span>
                            </div>
                        </Card>
                    ) : (
                        <Card
                            className={`${
                                modelStatus?.isConfigured
                                    ? "border-green-200 bg-green-50 dark:bg-green-900/20"
                                    : "border-yellow-200 bg-yellow-50 dark:bg-yellow-900/20"
                            }`}
                        >
                            <div className="flex items-center gap-3 p-4">
                                <CheckCircle2
                                    className={`h-5 w-5 ${
                                        modelStatus?.isConfigured
                                            ? "text-green-600"
                                            : "text-yellow-600"
                                    }`}
                                />
                                <div>
                                    <p className="font-medium">
                                        Model Service{" "}
                                        {modelStatus?.isConfigured
                                            ? "Available"
                                            : "Limited"}
                                    </p>
                                    <p className="text-sm text-gray-600 dark:text-gray-400">
                                        {modelStatus?.isConfigured
                                            ? `${
                                                  modelStatus.availableModels
                                                      ?.length || 0
                                              } models ready`
                                            : "Some models may not be available"}
                                    </p>
                                </div>
                            </div>
                        </Card>
                    )}

                    {/* Analysis Type Selection */}
                    <div>
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <BarChart3 className="h-5 w-5" />
                            Select Analysis Type
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {Object.values(ANALYSIS_TYPES).map((type) => {
                                const typeInfo = ANALYSIS_TYPE_INFO[type];
                                const isSelected =
                                    selectedAnalysisType === type;

                                return (
                                    <Card
                                        key={type}
                                        className={`cursor-pointer transition-all ${
                                            isSelected
                                                ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                                                : "hover:border-gray-300 dark:hover:border-gray-600"
                                        }`}
                                        onClick={() =>
                                            setSelectedAnalysisType(type)
                                        }
                                    >
                                        <div className="p-4">
                                            <div className="flex items-center gap-3 mb-2">
                                                {getAnalysisIcon(type)}
                                                <h4 className="font-semibold">
                                                    {typeInfo.label}
                                                </h4>
                                                {isSelected && (
                                                    <CheckCircle2 className="h-5 w-5 text-blue-600 ml-auto" />
                                                )}
                                            </div>
                                            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                                                {typeInfo.description}
                                            </p>
                                            <div className="flex items-center gap-2 text-xs text-gray-500">
                                                <Clock className="h-3 w-3" />
                                                {typeInfo.duration}
                                            </div>
                                        </div>
                                    </Card>
                                );
                            })}
                        </div>
                    </div>

                    {/* Model Selection */}
                    <div>
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <Brain className="h-5 w-5" />
                            Select Models
                            <span className="text-sm font-normal text-gray-500">
                                (Choose one or more)
                            </span>
                        </h3>
                        <div className="grid grid-cols-1 gap-4">
                            {Object.values(MODEL_TYPES).map((model) => {
                                const modelInfo = MODEL_INFO[model];
                                const isSelected =
                                    selectedModels.includes(model);
                                const isAvailable = isModelAvailable(model);

                                return (
                                    <Card
                                        key={model}
                                        className={`cursor-pointer transition-all ${
                                            !isAvailable
                                                ? "opacity-50 cursor-not-allowed"
                                                : isSelected
                                                ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                                                : "hover:border-gray-300 dark:hover:border-gray-600"
                                        }`}
                                        onClick={() =>
                                            isAvailable &&
                                            handleModelToggle(model)
                                        }
                                    >
                                        <div className="p-4">
                                            <div className="flex items-center gap-3 mb-2">
                                                <Brain className="h-5 w-5 text-gray-600" />
                                                <div className="flex-1">
                                                    <h4 className="font-semibold">
                                                        {modelInfo.label}
                                                    </h4>
                                                    <p className="text-sm text-gray-600 dark:text-gray-400">
                                                        {modelInfo.description}
                                                    </p>
                                                </div>
                                                {isSelected && (
                                                    <CheckCircle2 className="h-5 w-5 text-blue-600" />
                                                )}
                                                {!isAvailable && (
                                                    <div className="text-xs text-red-500 flex items-center gap-1">
                                                        <Info className="h-3 w-3" />
                                                        Unavailable
                                                    </div>
                                                )}
                                            </div>
                                            <div className="flex items-center justify-between text-xs text-gray-500">
                                                <span>
                                                    Specialty:{" "}
                                                    {modelInfo.specialty}
                                                </span>
                                                <span>
                                                    v{modelInfo.version}
                                                </span>
                                            </div>
                                        </div>
                                    </Card>
                                );
                            })}
                        </div>
                    </div>

                    {/* Analysis Summary */}
                    {selectedModels.length > 0 && (
                        <Card className="bg-gray-50 dark:bg-gray-800">
                            <div className="p-4">
                                <h4 className="font-semibold mb-2 flex items-center gap-2">
                                    <Info className="h-4 w-4" />
                                    Analysis Summary
                                </h4>
                                <div className="space-y-2 text-sm">
                                    <div className="flex justify-between">
                                        <span>Type:</span>
                                        <span className="font-medium">
                                            {
                                                ANALYSIS_TYPE_INFO[
                                                    selectedAnalysisType
                                                ].label
                                            }
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Models:</span>
                                        <span className="font-medium">
                                            {selectedModels.length}
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Estimated Duration:</span>
                                        <span className="font-medium">
                                            {
                                                ANALYSIS_TYPE_INFO[
                                                    selectedAnalysisType
                                                ].duration
                                            }
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </Card>
                    )}
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between p-6 border-t border-gray-200 dark:border-gray-700">
                    <div className="text-sm text-gray-500">
                        {selectedModels.length === 0
                            ? "Select at least one model to proceed"
                            : `Ready to analyze with ${
                                  selectedModels.length
                              } model${selectedModels.length > 1 ? "s" : ""}`}
                    </div>
                    <div className="flex gap-3">
                        <Button
                            onClick={onClose}
                            variant="outline"
                            disabled={isStarting}
                        >
                            Cancel
                        </Button>
                        <Button
                            onClick={handleStartAnalysis}
                            disabled={selectedModels.length === 0 || isStarting}
                        >
                            {isStarting ? (
                                <>
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                    Starting...
                                </>
                            ) : (
                                <>
                                    <Play className="mr-2 h-4 w-4" />
                                    Start Analysis
                                </>
                            )}
                        </Button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ModelSelectionModal;
