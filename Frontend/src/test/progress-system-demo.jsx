// src/test/progress-system-demo.jsx
// Demo component to test the enhanced progress system

import React, { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "../components/ui/Card.jsx";
import { Button } from "../components/ui/Button.jsx";
import { ToastProgress } from "../components/ui/ToastProgress.jsx";
import { AnalysisProgress } from "../components/ui/AnalysisProgress.jsx";
import { useAnalysisProgress } from "../hooks/useAnalysisProgress.jsx";
import { useToastOrchestrator } from "../lib/toastOrchestrator.jsx";
import { Play, Square, RefreshCw } from "lucide-react";

// Mock progress data for testing
const createMockProgressEvent = (
  mediaId,
  modelName,
  phase,
  progress = 0,
  total = 100
) => ({
  mediaId,
  userId: "test-user",
  event: `${phase.toUpperCase()}_${modelName.toUpperCase().replace("-", "_")}`,
  message: `Processing ${modelName} - ${phase}`,
  data: {
    model_name: modelName,
    phase,
    progress,
    total,
    speed: Math.random() * 50 + 10, // Random speed between 10-60 items/sec
    eta: (total - progress) / (Math.random() * 50 + 10),
    details: {
      phase: phase,
      currentFrame: progress,
      totalFrames: total,
      windowsProcessed: Math.floor(progress / 10),
      totalWindows: Math.floor(total / 10),
    },
  },
  timestamp: new Date().toISOString(),
});

export const ProgressSystemDemo = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [mockEvents, setMockEvents] = useState([]);
  const testMediaId = "test-media-123";
  const testFilename = "demo-video.mp4";

  // Use our enhanced progress hooks
  const {
    isProgressVisible,
    modelProgress,
    showProgress,
    hideProgress,
    overallProgress,
    activeModels,
    analysisProgressProps,
  } = useAnalysisProgress(testMediaId, testFilename);

  // Initialize toast orchestrator
  useToastOrchestrator();

  // Models to simulate
  const models = [
    "EfficientNet-Detector",
    "SigLIP-LSTM",
    "ScatteringWave-Detector",
  ];
  const phases = [
    "started",
    "frame_analysis",
    "window_processing",
    "visualization",
    "complete",
  ];

  // Simulation steps
  const simulationSteps = [
    { step: 0, description: "Analysis Queued", events: [] },
    {
      step: 1,
      description: "EfficientNet Started",
      events: [
        createMockProgressEvent(
          testMediaId,
          "EfficientNet-Detector",
          "started"
        ),
      ],
    },
    {
      step: 2,
      description: "EfficientNet Frame Analysis",
      events: [
        createMockProgressEvent(
          testMediaId,
          "EfficientNet-Detector",
          "frame_analysis",
          25,
          100
        ),
        createMockProgressEvent(
          testMediaId,
          "EfficientNet-Detector",
          "frame_analysis",
          50,
          100
        ),
        createMockProgressEvent(
          testMediaId,
          "EfficientNet-Detector",
          "frame_analysis",
          75,
          100
        ),
        createMockProgressEvent(
          testMediaId,
          "EfficientNet-Detector",
          "frame_analysis",
          100,
          100
        ),
      ],
    },
    {
      step: 3,
      description: "EfficientNet Visualization & Complete",
      events: [
        createMockProgressEvent(
          testMediaId,
          "EfficientNet-Detector",
          "visualization",
          50,
          100
        ),
        createMockProgressEvent(
          testMediaId,
          "EfficientNet-Detector",
          "visualization",
          100,
          100
        ),
        createMockProgressEvent(
          testMediaId,
          "EfficientNet-Detector",
          "complete",
          100,
          100
        ),
      ],
    },
    {
      step: 4,
      description: "SigLIP-LSTM Started & Processing",
      events: [
        createMockProgressEvent(testMediaId, "SigLIP-LSTM", "started"),
        createMockProgressEvent(
          testMediaId,
          "SigLIP-LSTM",
          "window_processing",
          30,
          120
        ),
        createMockProgressEvent(
          testMediaId,
          "SigLIP-LSTM",
          "window_processing",
          60,
          120
        ),
        createMockProgressEvent(
          testMediaId,
          "SigLIP-LSTM",
          "window_processing",
          90,
          120
        ),
        createMockProgressEvent(
          testMediaId,
          "SigLIP-LSTM",
          "window_processing",
          120,
          120
        ),
      ],
    },
    {
      step: 5,
      description: "SigLIP-LSTM Complete & ScatteringWave Started",
      events: [
        createMockProgressEvent(
          testMediaId,
          "SigLIP-LSTM",
          "complete",
          120,
          120
        ),
        createMockProgressEvent(
          testMediaId,
          "ScatteringWave-Detector",
          "started"
        ),
        createMockProgressEvent(
          testMediaId,
          "ScatteringWave-Detector",
          "frame_analysis",
          40,
          80
        ),
      ],
    },
    {
      step: 6,
      description: "All Models Complete",
      events: [
        createMockProgressEvent(
          testMediaId,
          "ScatteringWave-Detector",
          "frame_analysis",
          80,
          80
        ),
        createMockProgressEvent(
          testMediaId,
          "ScatteringWave-Detector",
          "complete",
          80,
          80
        ),
      ],
    },
  ];

  // Simulate progress events
  const runSimulation = async () => {
    setIsRunning(true);
    setCurrentStep(0);
    setMockEvents([]);

    for (let i = 0; i < simulationSteps.length; i++) {
      const stepData = simulationSteps[i];
      setCurrentStep(i);

      // Emit events for this step
      for (const event of stepData.events) {
        setMockEvents((prev) => [...prev, event]);

        // Simulate the real event emission
        window.dispatchEvent(
          new CustomEvent("mock-progress-event", {
            detail: event,
          })
        );

        // Wait between events
        await new Promise((resolve) => setTimeout(resolve, 800));
      }

      // Wait between steps
      await new Promise((resolve) => setTimeout(resolve, 1500));
    }

    setIsRunning(false);
  };

  const stopSimulation = () => {
    setIsRunning(false);
  };

  const resetSimulation = () => {
    setIsRunning(false);
    setCurrentStep(0);
    setMockEvents([]);
    hideProgress();
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Enhanced Progress System Demo</CardTitle>
          <p className="text-sm text-gray-600">
            This demo simulates the enhanced TQDM-style progress tracking system
            from Server → Backend → Frontend with real-time updates.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-3">
            <Button
              onClick={runSimulation}
              disabled={isRunning}
              className="gap-2"
            >
              <Play className="h-4 w-4" />
              Start Simulation
            </Button>
            <Button
              onClick={stopSimulation}
              disabled={!isRunning}
              variant="outline"
              className="gap-2"
            >
              <Square className="h-4 w-4" />
              Stop
            </Button>
            <Button
              onClick={resetSimulation}
              variant="outline"
              className="gap-2"
            >
              <RefreshCw className="h-4 w-4" />
              Reset
            </Button>
            <Button onClick={showProgress} variant="outline" className="gap-2">
              Show Detailed Progress
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Simulation Status */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Simulation Status</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <p>
                    <strong>Current Step:</strong> {currentStep} /{" "}
                    {simulationSteps.length - 1}
                  </p>
                  <p>
                    <strong>Step Description:</strong>{" "}
                    {simulationSteps[currentStep]?.description || "Ready"}
                  </p>
                  <p>
                    <strong>Events Emitted:</strong> {mockEvents.length}
                  </p>
                  <p>
                    <strong>Overall Progress:</strong>{" "}
                    {overallProgress.toFixed(1)}%
                  </p>
                  <p>
                    <strong>Active Models:</strong> {activeModels}
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Model Progress Summary */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Model Progress</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {models.map((model) => {
                    const progress = modelProgress[model];
                    const percentage = progress
                      ? (progress.progress / progress.total) * 100
                      : 0;

                    return (
                      <div key={model} className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="font-medium">{model}</span>
                          <span>{percentage.toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${percentage}%` }}
                          />
                        </div>
                        <div className="text-xs text-gray-500">
                          Phase: {progress?.phase || "waiting"}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Recent Events Log */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Recent Progress Events</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="max-h-64 overflow-y-auto space-y-2">
                {mockEvents
                  .slice(-10)
                  .reverse()
                  .map((event, index) => (
                    <div key={index} className="p-2 bg-gray-50 rounded text-sm">
                      <div className="flex justify-between">
                        <span className="font-medium">
                          {event.data.model_name}
                        </span>
                        <span className="text-gray-500">
                          {event.data.phase}
                        </span>
                      </div>
                      <div className="text-gray-600">{event.message}</div>
                      {event.data.progress && (
                        <div className="text-xs text-gray-500">
                          Progress: {event.data.progress}/{event.data.total}
                          {event.data.speed &&
                            ` (${event.data.speed.toFixed(1)} fps)`}
                          {event.data.eta &&
                            ` ETA: ${event.data.eta.toFixed(1)}s`}
                        </div>
                      )}
                    </div>
                  ))}
                {mockEvents.length === 0 && (
                  <p className="text-gray-500 text-center py-4">
                    No events yet. Start simulation to see progress updates.
                  </p>
                )}
              </div>
            </CardContent>
          </Card>
        </CardContent>
      </Card>

      {/* Enhanced Analysis Progress Modal */}
      <AnalysisProgress {...analysisProgressProps} />
    </div>
  );
};

export default ProgressSystemDemo;
