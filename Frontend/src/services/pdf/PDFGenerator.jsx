// src/services/pdf/PDFGenerator.jsx
// Main PDF generation service using react-pdf

import React from 'react';
import { Document, pdf } from '@react-pdf/renderer';
import { PDFCoverPage } from '../../components/pdf-legacy/PDFCoverPage';
import { PDFTableOfContents } from '../../components/pdf-legacy/PDFTableOfContents';
import { PDFExecutiveSummary } from '../../components/pdf-legacy/PDFExecutiveSummary';
import { PDFMediaInfo } from '../../components/pdf-legacy/PDFMediaInfo';
import PDFModelComparison from '../../components/pdf-legacy/PDFModelComparison';
import PDFModelDetail from '../../components/pdf-legacy/PDFModelDetail';
import PDFTemporalAnalysis from '../../components/pdf-legacy/PDFTemporalAnalysis';
import PDFTechnicalMetadata from '../../components/pdf-legacy/PDFTechnicalMetadata';
import PDFMethodology from '../../components/pdf-legacy/PDFMethodology';
import { 
  prepareReportData, 
  generateTableOfContents, 
  calculateTotalPages,
  generateReportId 
} from './pdfUtils';

/**
 * Complete PDF Document Component
 * Combines all pages into a single PDF document
 */
const PDFReportDocument = ({ data }) => {
  const { completedAnalyses = [], failedAnalyses = [] } = data;
  const allAnalyses = [...completedAnalyses, ...failedAnalyses];
  
  // Calculate page numbers dynamically
  let currentPage = 1;
  const coverPage = currentPage++;
  const tocPage = currentPage++;
  const summaryPage = currentPage++;
  const mediaPage = currentPage++;
  const comparisonPage = currentPage++;
  
  // Add a page for each model
  const modelPages = allAnalyses.map((analysis, index) => ({
    page: currentPage++,
    analysis,
    index,
  }));
  
  // Phase 4 pages
  const temporalPage = currentPage++;
  const metadataPage = currentPage++;
  const methodologyPage = currentPage++;
  
  const totalPages = currentPage - 1;
  
  return (
    <Document
      title={`Deepfake Analysis Report - ${data.filename}`}
      author="Drishtiksha"
      subject="Media Analysis Report"
      keywords="deepfake, analysis, ai, detection"
      creator="Drishtiksha Analysis System"
      producer="Drishtiksha"
    >
      {/* Page 1: Cover */}
      <PDFCoverPage 
        data={data} 
        pageNumber={coverPage} 
        totalPages={totalPages} 
      />
      
      {/* Page 2: Table of Contents */}
      <PDFTableOfContents 
        toc={data.toc} 
        data={data} 
        pageNumber={tocPage} 
        totalPages={totalPages} 
      />
      
      {/* Page 3: Executive Summary */}
      <PDFExecutiveSummary 
        data={data} 
        pageNumber={summaryPage} 
        totalPages={totalPages} 
      />
      
      {/* Page 4: Media Information */}
      <PDFMediaInfo 
        data={data} 
        pageNumber={mediaPage} 
        totalPages={totalPages} 
      />
      
      {/* Page 5: Model Comparison Matrix */}
      <PDFModelComparison 
        data={data} 
        pageNumber={comparisonPage} 
        totalPages={totalPages} 
      />
      
      {/* Pages 6+: Detailed Model Analysis (one per model) */}
      {modelPages.map((modelPage) => (
        <PDFModelDetail
          key={modelPage.analysis.id}
          analysis={modelPage.analysis}
          data={data}
          pageNumber={modelPage.page}
          totalPages={totalPages}
          modelIndex={modelPage.index + 1}
          totalModels={allAnalyses.length}
        />
      ))}
      
      {/* Phase 4: Advanced Features */}
      
      {/* Temporal Analysis (if video with frame data) */}
      <PDFTemporalAnalysis 
        data={data} 
        pageNumber={temporalPage} 
        totalPages={totalPages} 
      />
      
      {/* Technical Metadata */}
      <PDFTechnicalMetadata 
        data={data} 
        pageNumber={metadataPage} 
        totalPages={totalPages} 
      />
      
      {/* Methodology & Disclaimer */}
      <PDFMethodology 
        data={data} 
        pageNumber={methodologyPage} 
        totalPages={totalPages} 
      />
    </Document>
  );
};

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
