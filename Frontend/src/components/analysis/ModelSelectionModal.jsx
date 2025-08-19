// src/components/analysis/ModelSelectionModal.jsx

import React, { useState, useEffect, useMemo } from "react";
import {
    X,
    Brain,
    Play,
    Info,
    CheckCircle2,
    Loader2,
    AlertTriangle,
} from "lucide-react";
import { Button } from "../ui/Button";
import { Card } from "../ui/Card";
import { Modal } from "../ui/Modal";
import { Alert, AlertDescription, AlertTitle } from "../ui/Alert";
import { useServerStatusQuery } from "../../hooks/useMonitoringQuery.js";
import { useCreateAnalysisMutation } from "../../hooks/useVideosQuery.jsx";
import { cn } from "../../lib/utils";

const ModelCard = ({ model, isSelected, onSelect }) => (
    <Card
        className={cn(
            "cursor-pointer transition-all duration-200",
            isSelected && "border-primary-main ring-2 ring-primary-main/50",
            "hover:border-primary-main/50"
        )}
        onClick={() => onSelect(model.name)}
    >
        <div className="p-4">
            <div className="flex items-center gap-4">
                <div className="flex-shrink-0 w-12 h-12 bg-primary-main/10 rounded-lg flex items-center justify-center">
                    <Brain className="w-6 h-6 text-primary-main" />
                </div>
                <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-semibold">{model.name}</h4>
                        {model.isDetailed && (
                            <span className="px-2 py-1 text-xs bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 rounded-full">
                                Detailed Analysis
                            </span>
                        )}
                    </div>
                    <p className="text-sm text-light-muted-text dark:text-dark-muted-text">
                        {model.description}
                    </p>
                    {model.isDetailed && (
                        <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                            Supports frame-by-frame analysis and visualizations
                        </p>
                    )}
                </div>
                {isSelected && (
                    <CheckCircle2 className="h-6 w-6 text-primary-main flex-shrink-0" />
                )}
            </div>
        </div>
    </Card>
);

export const ModelSelectionModal = ({
    isOpen,
    onClose,
    videoId,
    onAnalysisStart,
}) => {
    const [selectedModel, setSelectedModel] = useState(null);
    const { data: serverStatus, isLoading: isModelsLoading } =
        useServerStatusQuery();

    const availableModels = useMemo(() => {
        return serverStatus?.modelsInfo?.filter((m) => m.loaded) || [];
    }, [serverStatus?.modelsInfo]);

    const createAnalysisMutation = useCreateAnalysisMutation();

    useEffect(() => {
        if (isOpen && availableModels.length > 0 && !selectedModel) {
            setSelectedModel(availableModels[0].name);
        }
    }, [isOpen, availableModels, selectedModel]);

    const handleStartAnalysis = async () => {
        if (!selectedModel || !videoId) return;
        await createAnalysisMutation.mutateAsync({
            videoId,
            analysisConfig: { model: selectedModel },
        });
        onAnalysisStart?.();
        onClose();
    };

    const modalFooter = (
        <>
            <Button variant="outline" onClick={onClose}>
                Cancel
            </Button>
            <Button
                onClick={handleStartAnalysis}
                isLoading={createAnalysisMutation.isPending}
                disabled={
                    !selectedModel ||
                    isModelsLoading ||
                    availableModels.length === 0
                }
            >
                {!createAnalysisMutation.isPending && (
                    <Play className="mr-2 h-4 w-4" />
                )}
                Start Analysis
            </Button>
        </>
    );

    return (
        <Modal
            isOpen={isOpen}
            onClose={onClose}
            title="Start New Analysis"
            description="Choose an available AI model to process your video."
            footer={modalFooter}
            className="max-w-5xl"
        >
            <div className="space-y-4">
                {isModelsLoading ? (
                    <div className="flex items-center justify-center h-48">
                        <Loader2 className="h-8 w-8 animate-spin text-primary-main" />
                    </div>
                ) : availableModels.length === 0 ? (
                    <Alert variant="destructive">
                        <AlertTriangle className="h-4 w-4" />
                        <AlertTitle>All Models Unavailable</AlertTitle>
                        <AlertDescription>
                            The analysis server is running, but no AI models are
                            currently loaded. Please contact support.
                        </AlertDescription>
                    </Alert>
                ) : (
                    <div className="grid grid-cols-1 gap-4">
                        {availableModels.map((model) => (
                            <ModelCard
                                key={model.name}
                                model={model}
                                isSelected={selectedModel === model.name}
                                onSelect={setSelectedModel}
                            />
                        ))}
                    </div>
                )}
            </div>
        </Modal>
    );
};

export default ModelSelectionModal;
