// src/services/pdf/pdfGeneratorService.js

import { pdf } from '@react-pdf/renderer';
import { PDFReportDocument } from './PDFReportDocument';
import {
  prepareReportData,
  generateTableOfContents,
  calculateTotalPages,
  generateReportId
} from './pdfUtils';

/**
 * PDFGenerator Service
 * Main service for generating and downloading PDF reports
 */
export const PDFGenerator = {
  /**
   * Prepares data for PDF generation
   * @param {Object} media - Media object from API
   * @param {Object} user - User object
   * @returns {Object} Prepared data with all necessary fields
   */
  prepareData(media, user) {
    // Enrich media data with calculated fields
    const enrichedData = prepareReportData(media, user);

    // Generate unique report ID
    enrichedData.reportId = generateReportId();

    // Calculate total pages
    enrichedData.totalPages = calculateTotalPages(enrichedData);

    // Generate table of contents
    enrichedData.toc = generateTableOfContents(enrichedData);

    return enrichedData;
  },

  /**
   * Generates PDF blob from media data
   * @param {Object} media - Media object
   * @param {Object} user - User object
   * @returns {Promise<Blob>} PDF blob
   */
  async generatePDFBlob(media, user) {
    try {
      // Prepare data
      const data = this.prepareData(media, user);

      // Generate PDF document
      const blob = await pdf(<PDFReportDocument data={data} />).toBlob();

      return blob;
    } catch (error) {
      console.error('[PDFGenerator] Error generating PDF blob:', error);
      throw new Error(`Failed to generate PDF: ${error.message}`);
    }
  },

  /**
   * Generates and downloads PDF report
   * @param {Object} media - Media object
   * @param {Object} user - User object
   * @param {Function} onProgress - Optional progress callback
   * @returns {Promise<void>}
   */
  async downloadPDF(media, user, onProgress = null) {
    try {
      if (onProgress) onProgress({ stage: 'preparing', percent: 10 });

      // Prepare data
      const data = this.prepareData(media, user);

      if (onProgress) onProgress({ stage: 'generating', percent: 30 });

      // Generate PDF
      const blob = await pdf(<PDFReportDocument data={data} />).toBlob();

      if (onProgress) onProgress({ stage: 'downloading', percent: 90 });

      // Create download link
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${media.filename.replace(/\.[^/.]+$/, '')}_analysis_report.pdf`;

      // Trigger download
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      // Clean up
      setTimeout(() => URL.revokeObjectURL(url), 100);

      if (onProgress) onProgress({ stage: 'complete', percent: 100 });

      return { success: true, filename: link.download };
    } catch (error) {
      console.error('[PDFGenerator] Error downloading PDF:', error);
      throw error;
    }
  },

  /**
   * Opens PDF in a new tab/window for preview
   * @param {Object} media - Media object
   * @param {Object} user - User object
   * @returns {Promise<void>}
   */
  async previewPDF(media, user) {
    try {
      // Prepare data
      const data = this.prepareData(media, user);

      // Generate PDF
      const blob = await pdf(<PDFReportDocument data={data} />).toBlob();

      // Create blob URL and open in new tab
      const url = URL.createObjectURL(blob);
      const newWindow = window.open(url, '_blank');

      if (!newWindow) {
        throw new Error('Popup blocked. Please allow popups to preview the PDF.');
      }

      // Clean up after a delay
      setTimeout(() => URL.revokeObjectURL(url), 10000);

      return { success: true };
    } catch (error) {
      console.error('[PDFGenerator] Error previewing PDF:', error);
      throw error;
    }
  },

  /**
   * Gets PDF metadata without generating the full document
   * Useful for showing file size estimates, page count, etc.
   * @param {Object} media - Media object
   * @param {Object} user - User object
   * @returns {Object} Metadata
   */
  getMetadata(media, user) {
    const data = this.prepareData(media, user);

    return {
      reportId: data.reportId,
      totalPages: data.totalPages,
      filename: `${media.filename.replace(/\.[^/.]+$/, '')}_analysis_report.pdf`,
      mediaId: data.id,
      mediaName: data.filename,
      overallAssessment: data.overallAssessment,
      totalModels: data.totalModels,
      generatedAt: new Date().toISOString(),
    };
  }
};

export default PDFGenerator;