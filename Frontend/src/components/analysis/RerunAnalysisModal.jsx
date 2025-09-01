// src/components/analysis/RerunAnalysisModal.jsx

import { Modal } from '../ui/Modal.jsx';
import { Button } from '../ui/Button.jsx';
import { BrainCircuit, Play } from 'lucide-react';
import { useRerunAnalysisMutation } from '../../hooks/useMediaQuery.jsx';

export const RerunAnalysisModal = ({ isOpen, onClose, media, onAnalysisStart }) => {
    // REFACTOR: Use the new, simpler mutation hook. It only needs the media ID.
    const rerunMutation = useRerunAnalysisMutation();

    const handleConfirm = async () => {
        if (!media?.id) return;
        
        await rerunMutation.mutateAsync(media.id);
        
        // Call the callback to trigger a data refetch on the page.
        onAnalysisStart?.();
        onClose();
    };

    // The modal's footer now has a clear primary and secondary action.
    const modalFooter = (
        <>
            <Button variant="outline" onClick={onClose} disabled={rerunMutation.isPending}>
                Cancel
            </Button>
            <Button
                onClick={handleConfirm}
                isLoading={rerunMutation.isPending}
            >
                {!rerunMutation.isPending && <Play className="mr-2 h-4 w-4" />}
                Start New Run
            </Button>
        </>
    );

    // The entire modal body is now a simple, clear confirmation message.
    return (
        <Modal
            isOpen={isOpen}
            onClose={onClose}
            title="Confirm New Analysis Run"
            description={`This will queue a new analysis for "${media?.filename}" using all currently available models.`}
            footer={modalFooter}
        >
            <div className="flex items-start space-x-4">
                <div className="flex-shrink-0 flex items-center justify-center h-12 w-12 rounded-full bg-blue-100 dark:bg-blue-900/30">
                    <BrainCircuit className="h-6 w-6 text-blue-600" />
                </div>
                <div className="text-sm">
                    <p className="font-semibold">Are you sure you want to proceed?</p>
                    <ul className="list-disc list-inside mt-2 text-light-muted-text dark:text-dark-muted-text space-y-1">
                        <li>A new, versioned analysis run will be created.</li>
                        <li>This action will consume processing resources.</li>
                        <li>Your existing analysis results will be preserved.</li>
                    </ul>
                </div>
            </div>
        </Modal>
    );
};