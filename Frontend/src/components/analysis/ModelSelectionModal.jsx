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
import { Modal } from "../ui/Modal"; // REFACTOR: Using our new base Modal component.
import { Alert, AlertDescription, AlertTitle } from "../ui/Alert"; // REFACTOR: Using our new Alert component.
import { MODEL_INFO } from "../../constants/apiEndpoints.js";
import { useServerStatusQuery } from "../../hooks/useMonitoringQuery.js";
import { useCreateAnalysisMutation } from "../../hooks/useVideosQuery.jsx";
import { cn } from "../../lib/utils";

// REFACTOR: Extracted the model selection card into a sub-component for clarity.
const ModelCard = ({
    modelKey,
    modelInfo,
    isSelected,
    isAvailable,
    onSelect,
}) => (
    <Card
        className={cn(
            "cursor-pointer transition-all duration-200",
            !isAvailable &&
                "opacity-50 cursor-not-allowed bg-light-hover dark:bg-dark-hover",
            isSelected && "border-primary-main ring-2 ring-primary-main/50",
            isAvailable && "hover:border-primary-main/50"
        )}
        onClick={() => isAvailable && onSelect(modelKey)}
    >
        <div className="p-4">
            <div className="flex items-center gap-4">
                <div className="flex-shrink-0 w-12 h-12 bg-primary-main/10 rounded-lg flex items-center justify-center">
                    <Brain className="w-6 h-6 text-primary-main" />
                </div>
                <div className="flex-1">
                    <h4 className="font-semibold">{modelInfo.label}</h4>
                    <p className="text-sm text-light-muted-text dark:text-dark-muted-text">
                        {modelInfo.description}
                    </p>
                </div>
                {isSelected && (
                    <CheckCircle2 className="h-6 w-6 text-primary-main flex-shrink-0" />
                )}
                {!isAvailable && (
                    <div className="text-xs text-red-500 flex items-center gap-1 font-semibold">
                        <Info className="h-3 w-3" /> Unavailable
                    </div>
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
    // REFACTOR: Logic is preserved.
    const [selectedModel, setSelectedModel] = useState(null);
    const { data: serverStatus, isLoading: isModelsLoading } =
        useServerStatusQuery();
    const availableModels = useMemo(() => {
        return (
            serverStatus?.modelsInfo?.filter((m) => m.loaded).map((m) => m.name) ||
            []
        );
    }, [serverStatus?.modelsInfo]);
    const createAnalysisMutation = useCreateAnalysisMutation();

    useEffect(() => {
        if (isOpen && availableModels.length > 0 && !selectedModel) {
            setSelectedModel(availableModels[0]);
        }
    }, [isOpen, availableModels, selectedModel]);

    const handleStartAnalysis = async () => {
        if (!selectedModel || !videoId) return;

        const analysisConfig = { model: selectedModel };
        await createAnalysisMutation.mutateAsync({ videoId, analysisConfig });

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
                    <div className="grid grid-cols-1 gap-3">
                        {Object.entries(MODEL_INFO).map(([key, modelInfo]) => (
                            <ModelCard
                                key={key}
                                modelKey={key}
                                modelInfo={modelInfo}
                                isSelected={selectedModel === key}
                                isAvailable={availableModels.includes(key)}
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
