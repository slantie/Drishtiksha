import React, { useState, useEffect } from "react";
import { Toaster } from "react-hot-toast";

export function ToastProvider({ children }) {
    const [mounted, setMounted] = useState(false);

    // useEffect to set mounted to true only on the client-side after hydration
    useEffect(() => {
        setMounted(true);
    }, []);

    return (
        <>
            {children}
            {/* Conditionally render Toaster only on the client after mounting */}
            {mounted && (
                <Toaster
                    position="bottom-right"
                    reverseOrder={false}
                    gutter={8}
                    containerStyle={{
                        top: 20,
                    }}
                    toastOptions={{
                        duration: 5000,
                        style: {
                            borderRadius: "10px",
                            background: "#333",
                            color: "#fff",
                        },
                        success: {
                            duration: 4000,
                        },
                        error: {
                            duration: 10000,
                        },
                        loading: {
                            duration: Infinity,
                        },
                    }}
                />
            )}
        </>
    );
}

export default ToastProvider;
