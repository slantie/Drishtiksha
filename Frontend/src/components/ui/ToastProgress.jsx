// src/components/ui/ToastProgress.jsx

export const ToastProgress = ({ message, modelName, progress, total }) => {
    const percentage = total > 0 ? Math.min(100, (progress / total) * 100) : 0;

    return (
        <div className="flex items-start space-x-3 text-white">
            <div className="flex-grow">
                {modelName && (
                    <p className="font-bold text-sm opacity-90">{modelName}</p>
                )}
                <p
                    className={`text-sm ${
                        modelName ? "opacity-80" : "font-semibold"
                    }`}
                >
                    {message}
                </p>

                {progress !== undefined && total !== undefined && (
                    <div className="w-full bg-white/20 rounded-full h-1.5 mt-2">
                        <div
                            className="bg-white h-1.5 rounded-full transition-all duration-300"
                            style={{ width: `${percentage}%` }}
                        ></div>
                    </div>
                )}
            </div>
        </div>
    );
};
