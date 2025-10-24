// src/services/pdf/PDFReportDocument.jsx

import React from "react";
import { Document } from "@react-pdf/renderer";
import { PDFCoverPage } from "../../components/pdf-legacy/PDFCoverPage";
import { PDFTableOfContents } from "../../components/pdf-legacy/PDFTableOfContents";
import { PDFExecutiveSummary } from "../../components/pdf-legacy/PDFExecutiveSummary";
import { PDFMediaInfo } from "../../components/pdf-legacy/PDFMediaInfo";
import PDFModelComparison from "../../components/pdf-legacy/PDFModelComparison";
import PDFModelDetail from "../../components/pdf-legacy/PDFModelDetail";
import PDFTemporalAnalysis from "../../components/pdf-legacy/PDFTemporalAnalysis";
import PDFTechnicalMetadata from "../../components/pdf-legacy/PDFTechnicalMetadata";
import PDFMethodology from "../../components/pdf-legacy/PDFMethodology";

/**
 * Complete PDF Document Component
 * Combines all pages into a single PDF document
 */
export const PDFReportDocument = ({ data }) => {
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
