// src/components/analysis/ModelSelectionModal.jsx

import React, { useState, useEffect } from "react";
import {
    X,
    Brain,
    Play,
    Clock,
    Info,
    CheckCircle2,
    Loader2,
    BarChart3,
    Zap,
    Search,
    Layers,
    TrendingUp,
} from "lucide-react";
import { Button } from "../ui/Button";
import { Card } from "../ui/Card";
import {
    ANALYSIS_TYPES,
    MODEL_INFO,
    ANALYSIS_TYPE_INFO,
} from "../../constants/apiEndpoints.js";
import { useServerStatusQuery } from "../../hooks/useMonitoringQuery.js";
import { useCreateAnalysisMutation } from "../../hooks/useVideosQuery.jsx";

const ModelSelectionModal = ({ isOpen, onClose, videoId, onAnalysisStart }) => {
    const [selectedAnalysisType, setSelectedAnalysisType] = useState(
        ANALYSIS_TYPES.COMPREHENSIVE
    );
    const [selectedModels, setSelectedModels] = useState([]);

    const { data: serverStatus, isLoading: isModelsLoading } =
        useServerStatusQuery();
    const availableModels =
        serverStatus?.modelsInfo?.filter((m) => m.loaded).map((m) => m.name) ||
        [];

    const createAnalysisMutation = useCreateAnalysisMutation();

    useEffect(() => {
        if (availableModels.length > 0) {
            setSelectedModels([availableModels[0]]);
        }
    }, [availableModels.length]);

    if (!isOpen) return null;

    const handleModelToggle = (modelKey) => {
        setSelectedModels([modelKey]); // Only allow selecting one model for now
    };

    const handleStartAnalysis = async () => {
        if (selectedModels.length === 0 || !videoId) return;

        const analysisConfig = {
            type: selectedAnalysisType,
            model: selectedModels[0],
        };

        await createAnalysisMutation.mutateAsync({ videoId, analysisConfig });

        onAnalysisStart?.();
        onClose();
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

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-white dark:bg-dark-background rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
                <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
                    <div className="flex items-center gap-3">
                        <Brain className="h-6 w-6 text-primary-main" />
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
                    {isModelsLoading && (
                        <Card>
                            <div className="flex items-center gap-3 p-4">
                                <Loader2 className="h-5 w-5 animate-spin" />
                                <span>Checking model availability...</span>
                            </div>
                        </Card>
                    )}

                    <div>
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <Brain className="h-5 w-5" /> Select Model
                        </h3>
                        <div className="grid grid-cols-1 gap-4">
                            {Object.entries(MODEL_INFO).map(
                                ([key, modelInfo]) => {
                                    const isSelected =
                                        selectedModels.includes(key);
                                    const isAvailable =
                                        availableModels.includes(key);
                                    return (
                                        <Card
                                            key={key}
                                            className={`cursor-pointer transition-all ${
                                                !isAvailable
                                                    ? "opacity-50 cursor-not-allowed"
                                                    : isSelected
                                                    ? "border-primary-main bg-primary-main/10"
                                                    : "hover:border-gray-300 dark:hover:border-gray-600"
                                            }`}
                                            onClick={() =>
                                                isAvailable &&
                                                handleModelToggle(key)
                                            }
                                        >
                                            <div className="p-4">
                                                <div className="flex items-center gap-3 mb-2">
                                                    <div className="flex-1">
                                                        <h4 className="font-semibold">
                                                            {modelInfo.label}
                                                        </h4>
                                                        <p className="text-sm text-gray-600 dark:text-gray-400">
                                                            {
                                                                modelInfo.description
                                                            }
                                                        </p>
                                                    </div>
                                                    {isSelected && (
                                                        <CheckCircle2 className="h-5 w-5 text-primary-main" />
                                                    )}
                                                    {!isAvailable && (
                                                        <div className="text-xs text-red-500 flex items-center gap-1">
                                                            <Info className="h-3 w-3" />
                                                            Unavailable
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        </Card>
                                    );
                                }
                            )}
                        </div>
                    </div>
                </div>

                <div className="flex items-center justify-end p-6 border-t border-gray-200 dark:border-gray-700">
                    <div className="flex gap-3">
                        <Button
                            onClick={onClose}
                            variant="outline"
                            disabled={createAnalysisMutation.isPending}
                        >
                            Cancel
                        </Button>
                        <Button
                            onClick={handleStartAnalysis}
                            disabled={
                                selectedModels.length === 0 ||
                                createAnalysisMutation.isPending ||
                                isModelsLoading ||
                                availableModels.length === 0
                            }
                        >
                            {createAnalysisMutation.isPending ? (
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
