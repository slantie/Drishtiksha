// src/services/pdf/MarkdownPDFService.js
// Simple PDF download service - backend handles everything

import { showToast } from "../../utils/toast.jsx";
import { authStorage } from "../../utils/authStorage.js";

/**
 * Simplified PDF Service
 * Just sends media ID to backend, backend does all the work
 */
export const MarkdownPDFService = {
  /**
   * Download PDF report for a specific analysis run
   * @param {string} analysisRunId - Analysis run ID
   * @param {Function} onProgress - Optional progress callback
   * @returns {Promise<void>}
   */
  async downloadPDF(analysisRunId, onProgress = null) {
    try {
      if (!analysisRunId) {
        throw new Error('Analysis run ID is required');
      }

      if (onProgress) onProgress({ stage: 'preparing', percent: 10 });

      // Get auth token from storage
      const { token } = authStorage.get();
      if (!token) {
        throw new Error('Authentication required. Please log in.');
      }

      // GET request to backend with analysis run ID
      const response = await fetch(`/api/v1/pdf/report/run/${analysisRunId}`, {
        method: 'GET',
        credentials: 'include', // Send auth cookie
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`PDF generation failed: ${errorText || response.statusText}`);
      }

      if (onProgress) onProgress({ stage: 'downloading', percent: 60 });

      // Get the PDF blob
      const blob = await response.blob();

      // Extract filename from Content-Disposition header if available
      const contentDisposition = response.headers.get('Content-Disposition');
      let downloadFilename = `Analysis_Report_${analysisRunId}.pdf`;
      
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?/);
        if (filenameMatch) {
          downloadFilename = filenameMatch[1];
        }
      }

      if (onProgress) onProgress({ stage: 'saving', percent: 90 });

      // Trigger download
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = downloadFilename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      if (onProgress) onProgress({ stage: 'complete', percent: 100 });

      showToast.success(`Report downloaded: ${downloadFilename}`);
    } catch (error) {
      console.error('[MarkdownPDFService] Error downloading PDF:', error);
      showToast.error(`Failed to download report: ${error.message}`);
      throw error;
    }
  },

  /**
   * Preview PDF in new tab
   * @param {string} analysisRunId - Analysis run ID
   * @returns {Promise<void>}
   */
  async previewPDF(analysisRunId) {
    try {
      if (!analysisRunId) {
        throw new Error('Analysis run ID is required');
      }

      // Get auth token from storage
      const { token } = authStorage.get();
      if (!token) {
        throw new Error('Authentication required. Please log in.');
      }

      // Fetch the PDF
      const response = await fetch(`/api/v1/pdf/report/run/${analysisRunId}`, {
        method: 'GET',
        credentials: 'include',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`PDF generation failed: ${errorText || response.statusText}`);
      }

      // Get blob and open in new tab
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      window.open(url, '_blank');

      // Clean up after a delay
      setTimeout(() => URL.revokeObjectURL(url), 60000);

      showToast.success('Opening report preview...');
    } catch (error) {
      console.error('[MarkdownPDFService] Error previewing PDF:', error);
      showToast.error(`Failed to preview report: ${error.message}`);
      throw error;
    }
  },
};

export default MarkdownPDFService;
