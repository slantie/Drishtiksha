import React from 'react';
import { Toaster } from 'react-hot-toast';

function ToastProvider() {
  return (
    <Toaster
      position="top-center"
      reverseOrder={false}
      gutter={8}
      toastOptions={{
        duration: 4000,
        style: {
          background: 'var(--toast-bg)',
          color: 'var(--toast-text)',
          border: '1px solid var(--toast-border)',
          borderRadius: '12px',
          boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
          backdropFilter: 'blur(16px)',
          fontSize: '14px',
          fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace',
          fontWeight: '500',
          padding: '16px',
          maxWidth: '400px',
        },
        
        success: {
          duration: 3000,
          style: {
            border: '1px solid #10b981',
            background: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)',
            color: '#065f46',
          },
          iconTheme: {
            primary: '#10b981',
            secondary: '#ffffff',
          },
        },
        
        error: {
          duration: 5000,
          style: {
            border: '1px solid #ef4444',
            background: 'linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)',
            color: '#991b1b',
          },
          iconTheme: {
            primary: '#ef4444',
            secondary: '#ffffff',
          },
        },
        
        loading: {
          style: {
            border: '1px solid #6b7280',
            background: 'linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%)',
            color: '#374151',
          },
        },
      }}
    />
  );
}

export default ToastProvider;