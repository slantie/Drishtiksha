// src/services/DownloadReport.js

import html2pdf from "html2pdf.js";

/**
 * @file Service for generating and downloading video analysis reports.
 * This version focuses on creating a high-fidelity, A4-formatted report
 * that looks professional both on-screen and when printed.
 */

//================================================================================================
// CONFIGURATION
//================================================================================================

const ReportConfig = {
    brandName: "Drishtiksha",
    mainTitle: "Video Analysis Report",
    // Icons
    icons: {
        real: "✅",
        fake: "⚠️",
        info: "ℹ️",
        summary: "📊",
        details: "🔍",
        error: "⚠️",
    },
    // Colors
    colors: {
        primary: "#1d4ed8",
        authentic: "#16a34a",
        deepfake: "#dc2626",
        textPrimary: "#111827",
        textSecondary: "#374151",
        textMuted: "#6b7280",
        background: "#f9fafb",
        border: "#e5e7eb",
        pageBackground: "#f0f2f5",
    },
};

//================================================================================================
// PRIVATE HELPERS & DATA PROCESSING
//================================================================================================

const formatters = {
    bytes: (bytes) =>
        bytes ? `${(bytes / 1024 / 1024).toFixed(2)} MB` : "N/A",
    date: (dateString) => {
        if (!dateString) return "N/A";
        return new Date(dateString).toLocaleString("en-US", {
            year: "numeric",
            month: "long",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
        });
    },
    processingTime: (timeInSeconds) => {
        if (timeInSeconds === null || typeof timeInSeconds === "undefined")
            return "N/A";
        if (timeInSeconds < 60) return `${timeInSeconds.toFixed(1)}s`;
        const minutes = Math.floor(timeInSeconds / 60);
        const seconds = (timeInSeconds % 60).toFixed(1);
        return `${minutes}m ${seconds}s`;
    },
};

const prepareReportData = (video) => {
    const completedAnalyses =
        video.analyses?.filter((a) => a.status === "COMPLETED") || [];
    const realDetections = completedAnalyses.filter(
        (a) => a.prediction === "REAL"
    ).length;
    const fakeDetections = completedAnalyses.filter(
        (a) => a.prediction === "FAKE"
    ).length;

    let overallAssessment = "Inconclusive";
    let assessmentColor = ReportConfig.colors.textMuted;
    if (fakeDetections > realDetections) {
        overallAssessment = "Likely Deepfake";
        assessmentColor = ReportConfig.colors.deepfake;
    } else if (realDetections > fakeDetections) {
        overallAssessment = "Likely Authentic";
        assessmentColor = ReportConfig.colors.authentic;
    }

    return {
        ...video,
        completedAnalyses,
        realDetections,
        fakeDetections,
        overallAssessment,
        assessmentColor,
    };
};

//================================================================================================
// HTML STYLING (CSS)
//================================================================================================

const getReportStyles = (data) => `
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        @page { size: A4; margin: 0; }

        body {
            -webkit-print-color-adjust: exact;
            color-adjust: exact;
            background-color: ${ReportConfig.colors.pageBackground};
            font-family: 'Inter', sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: ${ReportConfig.colors.textPrimary};
            margin: 0;
            padding: 2rem 0;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .page {
            width: 210mm;
            min-height: 297mm;
            padding: 20mm 18mm;
            background: white;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
            page-break-after: always;
        }
        .page:last-of-type {
            page-break-after: auto;
            margin-bottom: 0;
        }

        h1, h2, h3 { font-weight: 700; color: ${ReportConfig.colors.textPrimary}; line-height: 1.2; }
        h1 { font-size: 24pt; margin-bottom: 0.5em; }
        h2 {
            font-size: 18pt;
            border-bottom: 2px solid ${ReportConfig.colors.border};
            padding-bottom: 0.5rem;
            /* Use padding-top instead of margin-top to avoid page-break issues */
            padding-top: 2.5rem;
            margin-bottom: 1.5rem;
        }
        h3 { font-size: 14pt; }
        .section { break-inside: avoid; }

        .report-header {
            display: flex; justify-content: space-between; align-items: flex-start;
            padding-bottom: 1.5rem; border-bottom: 3px solid ${ReportConfig.colors.primary}; margin-bottom: 2rem;
        }
        .report-header .title-block { max-width: 65%; }
        .report-header .brand { font-weight: 600; color: ${ReportConfig.colors.primary}; font-size: 12pt; }
        .report-header .meta-block { text-align: right; font-size: 10pt; color: ${ReportConfig.colors.textMuted}; }
        .meta-block strong { color: ${ReportConfig.colors.textSecondary}; }
        .meta-block > div { margin-bottom: 0.25rem; }

        .summary-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem; }
        .summary-card {
            background-color: ${ReportConfig.colors.background}; border: 1px solid ${ReportConfig.colors.border};
            border-top: 4px solid; border-radius: 8px; padding: 1.25rem;
        }
        .summary-card .label { font-size: 11pt; font-weight: 500; color: ${ReportConfig.colors.textSecondary}; margin-bottom: 0.5rem; }
        .summary-card .value { font-size: 24pt; font-weight: 700; }
        .summary-card.overall { grid-column: 1 / -1; border-top-color: ${data.assessmentColor}; }
        .summary-card.overall .value { color: ${data.assessmentColor}; font-size: 20pt; }
        .summary-card.authentic { border-top-color: ${ReportConfig.colors.authentic}; }
        .summary-card.authentic .value { color: ${ReportConfig.colors.authentic}; }
        .summary-card.deepfake { border-top-color: ${ReportConfig.colors.deepfake}; }
        .summary-card.deepfake .value { color: ${ReportConfig.colors.deepfake}; }
        
        .details-table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        .details-table td { padding: 0.75rem 0; border-bottom: 1px solid ${ReportConfig.colors.border}; vertical-align: top; }
        .details-table tr:last-child td { border-bottom: none; }
        .details-table .property-cell { font-weight: 600; color: ${ReportConfig.colors.textSecondary}; width: 30%; }

        .analysis-card {
            border: 1px solid ${ReportConfig.colors.border}; border-radius: 8px;
            margin-bottom: 1.5rem; break-inside: avoid; overflow: hidden;
        }
        .analysis-card-header {
            display: flex; justify-content: space-between; align-items: center;
            padding: 1rem 1.25rem; background-color: ${ReportConfig.colors.background};
            border-bottom: 1px solid ${ReportConfig.colors.border};
        }
        .analysis-card-header .model-name { font-size: 14pt; font-weight: 600; }
        .prediction-badge { padding: 0.4rem 1rem; border-radius: 999px; font-weight: 600; color: white; font-size: 10pt; }
        .prediction-badge.real { background-color: ${ReportConfig.colors.authentic}; }
        .prediction-badge.fake { background-color: ${ReportConfig.colors.deepfake}; }

        .analysis-card-body { display: grid; grid-template-columns: 1fr 1fr; padding: 1.5rem 1.25rem; align-items: center; }
        .confidence-display { text-align: center; }
        .confidence-display .label { font-size: 11pt; color: ${ReportConfig.colors.textMuted}; margin-bottom: 0.25rem; }
        .confidence-display .value { font-size: 40pt; font-weight: 700; line-height: 1; }
        .confidence-display .value.real { color: ${ReportConfig.colors.authentic}; }
        .confidence-display .value.fake { color: ${ReportConfig.colors.deepfake}; }

        .no-analysis-placeholder {
            text-align: center; padding: 4rem 2rem; background-color: ${ReportConfig.colors.background};
            border: 2px dashed ${ReportConfig.colors.border}; border-radius: 8px; color: ${ReportConfig.colors.textMuted};
        }

        .report-footer {
            margin-top: auto; /* Push footer to the bottom of the page */
            padding-top: 2rem;
            border-top: 1px solid ${ReportConfig.colors.border};
            text-align: center; font-size: 9pt; color: ${ReportConfig.colors.textMuted};
        }

        @media print {
            body { background-color: white; padding: 0; display: block; }
            .page { box-shadow: none; margin: 0; padding: 15mm 18mm; min-height: 0; }
        }
    </style>
`;

//================================================================================================
// HTML COMPONENT RENDERERS
//================================================================================================

const renderHeader = (data, user) => `
    <div class="report-header">
        <div class="title-block">
            <div class="brand">${ReportConfig.brandName}</div>
            <h1>${ReportConfig.mainTitle}</h1>
            <div>File: ${data.filename} - <a href="${data.url.replace(
    "/upload/",
    "/upload/f_auto,q_auto/"
)}">View</a></div>
        </div>
        <div class="meta-block">
            <div><strong>Report Date:</strong> ${new Date().toLocaleDateString(
                "en-US"
            )}</div>
            <div style="margin-top: 1rem;"><strong>File:</strong> ${
                data.filename
            }</div>
            ${
                user
                    ? `
                <div style="margin-top: 1rem;"><strong>Generated By:</strong></div>
                <div>${user.firstName || "N/A"} ${user.lastName || ""} (${
                          user.email || "N/A"
                      })</div>
            `
                    : ""
            }
        </div>
    </div>
`;

const renderSummary = (data) => `
    <div class="section">
        <h2>${ReportConfig.icons.summary} Executive Summary</h2>
        <div class="summary-grid">
            <div class="summary-card overall">
                <div class="label">Overall Assessment</div>
                <div class="value">${data.overallAssessment}</div>
            </div>
            <div class="summary-card authentic">
                <div class="label">Authentic Detections</div>
                <div class="value">${data.realDetections}</div>
            </div>
            <div class="summary-card deepfake">
                <div class="label">Deepfake Detections</div>
                <div class="value">${data.fakeDetections}</div>
            </div>
        </div>
    </div>
`;

const renderVideoInfo = (data) => `
    <div class="section">
        <h2>${ReportConfig.icons.info} Video Information</h2>
        <table class="details-table">
            <tbody>
                <tr><td class="property-cell">Filename</td><td>${
                    data.filename
                }</td></tr>
                <tr><td class="property-cell">File Size</td><td>${formatters.bytes(
                    data.size
                )}</td></tr>
                <tr><td class="property-cell">Format</td><td>${
                    data.mimetype?.split("/")[1]?.toUpperCase() || "N/A"
                }</td></tr>
                <tr><td class="property-cell">Upload Date</td><td>${formatters.date(
                    data.createdAt
                )}</td></tr>
                <tr><td class="property-cell">Video Status</td><td>${
                    data.status
                }</td></tr>
                ${
                    data.description
                        ? `<tr><td class="property-cell">Description</td><td>${data.description}</td></tr>`
                        : ""
                }
            </tbody>
        </table>
    </div>
`;

const renderAnalysisCard = (analysis) => {
    const isReal = analysis.prediction === "REAL";
    const confidence = (analysis.confidence * 100).toFixed(1);
    const badgeClass = isReal ? "real" : "fake";

    return `
        <div class="analysis-card">
            <div class="analysis-card-header">
                <div class="model-name">${analysis.model} Model</div>
                <div class="prediction-badge ${badgeClass}">${
        isReal ? "Authentic" : "Deepfake"
    }</div>
            </div>
            <div class="analysis-card-body">
                <div class="confidence-display">
                    <div class="label">Confidence Score</div>
                    <div class="value ${badgeClass}">${confidence}%</div>
                </div>
                <table class="details-table">
                    <tr><td class="property-cell">Prediction</td><td>${
                        analysis.prediction
                    }</td></tr>
                    <tr><td class="property-cell">Analysis Date</td><td>${formatters.date(
                        analysis.createdAt
                    )}</td></tr>
                    <tr><td class="property-cell">Processing Time</td><td>${formatters.processingTime(
                        analysis.processingTime
                    )}</td></tr>
                </table>
            </div>
        </div>
    `;
};

const renderDetailedAnalyses = (data) => `
    <div class="section">
        <h2>${ReportConfig.icons.details} Detailed Analysis</h2>
        ${data.analyses.map(renderAnalysisCard).join("")}
    </div>
`;

const renderNoAnalysesPlaceholder = () => `
    <div class="section">
        <h2>${ReportConfig.icons.details} Analysis Results</h2>
        <div class="no-analysis-placeholder">
            <h3>No Analysis Completed</h3>
            <p>This video is pending analysis or could not be processed.</p>
        </div>
    </div>
`;

const renderFooter = (pageNumber, VideoID) => `
    <div class="report-footer">
        ${ReportConfig.brandName} | Page ${pageNumber} | ${VideoID}
    </div>
`;

//================================================================================================
// MAIN HTML GENERATOR
//================================================================================================

const generateReportHTML = (video, user) => {
    const data = prepareReportData(video);

    const mainContent = `
        ${renderHeader(data, user)}
        ${renderSummary(data)}
        ${renderVideoInfo(data)}
    `;

    const detailContent =
        data.analyses?.length > 0
            ? renderDetailedAnalyses(data)
            : renderNoAnalysesPlaceholder();

    return `
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Video Analysis Report - ${data.filename}</title>
            ${getReportStyles(data)}
        </head>
        <body>
            <div class="page" style="display: flex; flex-direction: column;">
                ${mainContent}
                ${renderFooter(1, data.id)}
            </div>
            ${
                data.analyses?.length > 0
                    ? `
            <div class="page" style="display: flex; flex-direction: column;">
                ${detailContent}
                ${renderFooter(2, data.id)}
            </div>
            `
                    : ""
            }
        </body>
        </html>
    `;
};

//================================================================================================
// PUBLIC API (EXPORTED SERVICE)
//================================================================================================

export const DownloadService = {
    async generateAndDownloadPDF(video, user) {
        try {
            const htmlContent = generateReportHTML(video, user);
            const element = document.createElement("div");
            element.innerHTML = htmlContent;
            const body = element.querySelector("body");

            const options = {
                margin: 0,
                filename: `${video.filename.replace(
                    /\.[^/.]+$/,
                    ""
                )}_report.pdf`,
                image: { type: "jpeg", quality: 0.98 },
                html2canvas: {
                    scale: 2,
                    useCORS: true,
                    letterRendering: true,
                    backgroundColor: null,
                },
                jsPDF: { unit: "mm", format: "a4", orientation: "portrait" },
            };

            await html2pdf().from(body).set(options).save();
        } catch (error) {
            console.error("Error generating PDF with html2pdf:", error);
            await this.generateAndDownloadPDFPrint(video, user);
        }
    },

    async generateAndDownloadPDFPrint(video, user) {
        try {
            const htmlContent = generateReportHTML(video, user);
            const printWindow = window.open("", "_blank");
            if (!printWindow) {
                alert(
                    "Popup blocked. Please allow popups to generate the report."
                );
                throw new Error("Popup blocked by browser.");
            }
            printWindow.document.write(htmlContent);
            printWindow.document.close();
            setTimeout(() => {
                printWindow.focus();
                printWindow.print();
                setTimeout(() => {
                    if (!printWindow.closed) printWindow.close();
                }, 2000);
            }, 750);
        } catch (error) {
            console.error("Error generating PDF via print dialog:", error);
            alert(`Failed to generate PDF report: ${error.message}`);
        }
    },

    async downloadVideo(videoUrl, filename) {
        try {
            const response = await fetch(videoUrl);
            if (!response.ok)
                throw new Error(`HTTP error! Status: ${response.status}`);
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error("Error downloading video:", error);
            alert(`Failed to download video: ${error.message}`);
        }
    },

    async downloadHTMLReport(video, user) {
        try {
            const htmlContent = generateReportHTML(video, user);
            const blob = new Blob([htmlContent], {
                type: "text/html;charset=utf-8",
            });
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = `${video.filename.replace(
                /\.[^/.]+$/,
                ""
            )}_analysis_report.html`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        } catch (error) {
            console.error("Error downloading HTML report:", error);
            alert(`Failed to generate HTML report: ${error.message}`);
        }
    },
};
