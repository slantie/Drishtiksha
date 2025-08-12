// src/services/DownloadReport.js

import html2pdf from "html2pdf.js";

// Helper functions remain the same
const formatBytes = (bytes) =>
    bytes ? `${(bytes / 1024 / 1024).toFixed(2)} MB` : "N/A";

const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
    });
};

const formatProcessingTime = (timeInSeconds) => {
    if (!timeInSeconds) return "N/A";
    if (timeInSeconds < 60) return `${timeInSeconds.toFixed(1)}s`;
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = (timeInSeconds % 60).toFixed(1);
    return `${minutes}m ${seconds}s`;
};

// Generate professional HTML template for PDF conversion
const generateReportHTML = (video) => {
    const completedAnalyses =
        video.analyses?.filter((a) => a.status === "COMPLETED") || [];
    const realDetections = completedAnalyses.filter(
        (a) => a.prediction === "REAL"
    ).length;
    const fakeDetections = completedAnalyses.filter(
        (a) => a.prediction === "FAKE"
    ).length;

    // Determine overall assessment
    let overallAssessment = "INCONCLUSIVE";
    let assessmentColor = "#6b7280";
    if (fakeDetections > realDetections) {
        overallAssessment = "LIKELY DEEPFAKE";
        assessmentColor = "#ef4444";
    } else if (realDetections > fakeDetections) {
        overallAssessment = "LIKELY AUTHENTIC";
        assessmentColor = "#22c55e";
    }

    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Analysis Report - ${video.filename}</title>
    <style>
        @page {
            size: A4;
            margin: 0.5in;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            line-height: 1.6;
            color: #1f2937;
            font-size: 11pt;
            background: white;
        }
        
        .report-container {
            max-width: 100%;
            margin: 0 auto;
            background: white;
        }
        
        /* Header Styles */
        .header {
            background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
            color: white;
            padding: 30px 25px;
            text-align: center;
            margin-bottom: 25px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .header h1 {
            font-size: 26pt;
            font-weight: 700;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }
        
        .header .subtitle {
            font-size: 14pt;
            opacity: 0.95;
            margin-bottom: 5px;
        }
        
        .header .timestamp {
            font-size: 11pt;
            opacity: 0.8;
        }
        
        /* Video Title */
        .video-title {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border-left: 6px solid #2563eb;
            padding: 20px 25px;
            margin-bottom: 25px;
            border-radius: 0 8px 8px 0;
        }
        
        .video-title h2 {
            font-size: 18pt;
            font-weight: 600;
            color: #1e40af;
            word-break: break-all;
        }
        
        /* Section Styles */
        .section {
            margin-bottom: 30px;
            break-inside: avoid;
        }
        
        .section-title {
            font-size: 16pt;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 3px solid #e5e7eb;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .section-icon {
            font-size: 18pt;
        }
        
        /* Table Styles */
        .info-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .info-table th {
            background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
            color: white;
            padding: 15px 20px;
            text-align: left;
            font-weight: 600;
            font-size: 12pt;
        }
        
        .info-table td {
            padding: 12px 20px;
            border-bottom: 1px solid #e5e7eb;
            font-size: 11pt;
        }
        
        .info-table tr:nth-child(even) {
            background: #f8fafc;
        }
        
        .info-table tr:last-child td {
            border-bottom: none;
        }
        
        .info-table .property-cell {
            font-weight: 600;
            color: #374151;
            width: 35%;
        }
        
        /* Summary Grid */
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .summary-card {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            padding: 25px 20px;
            text-align: center;
            transition: transform 0.2s;
        }
        
        .summary-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        .summary-card .number {
            font-size: 32pt;
            font-weight: 700;
            color: #2563eb;
            display: block;
            margin-bottom: 8px;
        }
        
        .summary-card .label {
            font-size: 11pt;
            color: #6b7280;
            font-weight: 500;
        }
        
        .summary-card.overall {
            grid-column: span 2;
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        }
        
        .summary-card.overall .number {
            font-size: 16pt;
            color: ${assessmentColor};
            font-weight: 700;
        }
        
        /* Analysis Cards */
        .analysis-card {
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            margin: 20px 0;
            overflow: hidden;
            break-inside: avoid;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        .analysis-card.real {
            border-left: 6px solid #22c55e;
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        }
        
        .analysis-card.fake {
            border-left: 6px solid #ef4444;
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        }
        
        .analysis-header {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            padding: 20px 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        }
        
        .model-info {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .model-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 16pt;
        }
        
        .model-name {
            font-size: 16pt;
            font-weight: 700;
            color: #1f2937;
        }
        
        .model-subtitle {
            font-size: 10pt;
            color: #6b7280;
            margin-top: 2px;
        }
        
        .prediction-badge {
            padding: 10px 20px;
            border-radius: 25px;
            color: white;
            font-weight: 700;
            font-size: 11pt;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .prediction-badge.real {
            background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        }
        
        .prediction-badge.fake {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        }
        
        .confidence-section {
            text-align: center;
            padding: 30px 25px;
        }
        
        .confidence-number {
            font-size: 48pt;
            font-weight: 700;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .confidence-number.real {
            background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .confidence-number.fake {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .confidence-label {
            font-size: 14pt;
            color: #6b7280;
            font-weight: 500;
            margin-bottom: 5px;
        }
        
        .confidence-description {
            font-size: 18pt;
            font-weight: 600;
            color: #374151;
        }
        
        /* Details Table */
        .details-table {
            width: 100%;
            border-collapse: collapse;
            margin: 0;
        }
        
        .details-table td {
            padding: 12px 25px;
            border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        }
        
        .details-table td:first-child {
            font-weight: 600;
            color: #374151;
            width: 40%;
            background: rgba(255, 255, 255, 0.5);
        }
        
        .details-table tr:last-child td {
            border-bottom: none;
        }
        
        /* Error Box */
        .error-box {
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border: 2px solid #fecaca;
            color: #dc2626;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 15px 25px;
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }
        
        .error-icon {
            color: #dc2626;
            font-size: 16pt;
            margin-top: 2px;
        }
        
        .error-content {
            flex: 1;
        }
        
        .error-title {
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        /* No Analysis State */
        .no-analysis {
            text-align: center;
            padding: 50px 30px;
            background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
            border: 2px dashed #d1d5db;
            border-radius: 12px;
            color: #6b7280;
        }
        
        .no-analysis-icon {
            font-size: 48pt;
            margin-bottom: 20px;
            opacity: 0.5;
        }
        
        .no-analysis h3 {
            font-size: 18pt;
            margin-bottom: 10px;
            color: #374151;
        }
        
        .no-analysis p {
            font-size: 12pt;
        }
        
        /* Footer */
        .footer {
            margin-top: 40px;
            padding: 25px 0;
            border-top: 3px solid #e5e7eb;
            text-align: center;
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border-radius: 8px;
        }
        
        .footer .brand {
            font-size: 14pt;
            font-weight: 700;
            color: #1e40af;
            margin-bottom: 5px;
        }
        
        .footer .confidential {
            font-size: 10pt;
            color: #6b7280;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        
        /* Print Optimizations */
        @media print {
            body { 
                font-size: 10pt;
                -webkit-print-color-adjust: exact;
                color-adjust: exact;
            }
            
            .header h1 { font-size: 22pt; }
            .confidence-number { font-size: 36pt; }
            
            .analysis-card {
                break-inside: avoid;
                page-break-inside: avoid;
            }
            
            .section {
                break-inside: avoid;
                page-break-inside: avoid;
            }
        }
        
        /* Utility Classes */
        .text-center { text-align: center; }
        .font-bold { font-weight: 700; }
        .text-sm { font-size: 10pt; }
        .text-lg { font-size: 14pt; }
        .mb-2 { margin-bottom: 8px; }
        .mb-4 { margin-bottom: 16px; }
    </style>
</head>
<body>
    <div class="report-container">
        <!-- Header -->
        <div class="header">
            <h1>🛡️ VIDEO ANALYSIS REPORT</h1>
            <div class="subtitle">VidVigilante Deepfake Detection System</div>
            <div class="timestamp">Generated: ${new Date().toLocaleString()}</div>
        </div>

        <!-- Video Title -->
        <div class="video-title">
            <h2>📁 ${video.filename}</h2>
        </div>

        <!-- Video Information Section -->
        <div class="section">
            <div class="section-title">
                <span class="section-icon">📊</span>
                VIDEO INFORMATION
            </div>
            <table class="info-table">
                <thead>
                    <tr>
                        <th>Property</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="property-cell">Filename</td>
                        <td>${video.filename}</td>
                    </tr>
                    <tr>
                        <td class="property-cell">File Size</td>
                        <td>${formatBytes(video.size)}</td>
                    </tr>
                    <tr>
                        <td class="property-cell">Format</td>
                        <td>${
                            video.mimetype?.split("/")[1]?.toUpperCase() ||
                            "Unknown"
                        }</td>
                    </tr>
                    <tr>
                        <td class="property-cell">Upload Date</td>
                        <td>${formatDate(video.createdAt)}</td>
                    </tr>
                    <tr>
                        <td class="property-cell">Status</td>
                        <td><strong>${video.status}</strong></td>
                    </tr>
                    ${
                        video.description
                            ? `
                    <tr>
                        <td class="property-cell">Description</td>
                        <td>${video.description}</td>
                    </tr>`
                            : ""
                    }
                </tbody>
            </table>
        </div>

        ${
            video.analyses?.length > 0
                ? `
        <!-- Analysis Summary -->
        <div class="section">
            <div class="section-title">
                <span class="section-icon">📈</span>
                ANALYSIS SUMMARY
            </div>
            <div class="summary-grid">
                <div class="summary-card">
                    <span class="number">${completedAnalyses.length}</span>
                    <div class="label">Completed Analyses</div>
                </div>
                <div class="summary-card">
                    <span class="number" style="color: #22c55e;">${realDetections}</span>
                    <div class="label">Authentic Detections</div>
                </div>
                <div class="summary-card">
                    <span class="number" style="color: #ef4444;">${fakeDetections}</span>
                    <div class="label">Deepfake Detections</div>
                </div>
                <div class="summary-card overall">
                    <span class="number">${overallAssessment}</span>
                    <div class="label">Overall Assessment</div>
                </div>
            </div>
        </div>

        <!-- Detailed Analysis Results -->
        <div class="section">
            <div class="section-title">
                <span class="section-icon">🔍</span>
                DETAILED ANALYSIS RESULTS
            </div>
            
            ${video.analyses
                .map((analysis) => {
                    const isReal = analysis.prediction === "REAL";
                    const confidence = (analysis.confidence * 100).toFixed(1);

                    return `
                <div class="analysis-card ${isReal ? "real" : "fake"}">
                    <div class="analysis-header">
                        <div class="model-info">
                            <div class="model-icon">🤖</div>
                            <div>
                                <div class="model-name">${
                                    analysis.model
                                } MODEL</div>
                                <div class="model-subtitle">AI Detection System</div>
                            </div>
                        </div>
                        <div class="prediction-badge ${
                            isReal ? "real" : "fake"
                        }">
                            <span>${isReal ? "✅" : "⚠️"}</span>
                            ${isReal ? "AUTHENTIC" : "DEEPFAKE"}
                        </div>
                    </div>
                    
                    <div class="confidence-section">
                        <div class="confidence-number ${
                            isReal ? "real" : "fake"
                        }">${confidence}%</div>
                        <div class="confidence-label">Confidence Level</div>
                        <div class="confidence-description">${
                            isReal
                                ? "Likely Authentic Video"
                                : "Likely Deepfake Content"
                        }</div>
                    </div>
                    
                    <table class="details-table">
                        <tr>
                            <td>Analysis Status</td>
                            <td><strong>${analysis.status}</strong></td>
                        </tr>
                        <tr>
                            <td>Analysis Date</td>
                            <td>${formatDate(analysis.createdAt)}</td>
                        </tr>
                        ${
                            analysis.processingTime
                                ? `
                        <tr>
                            <td>Processing Time</td>
                            <td>${formatProcessingTime(
                                analysis.processingTime
                            )}</td>
                        </tr>`
                                : ""
                        }
                    </table>
                    
                    ${
                        analysis.errorMessage
                            ? `
                    <div class="error-box">
                        <div class="error-icon">⚠️</div>
                        <div class="error-content">
                            <div class="error-title">Analysis Error</div>
                            <div>${analysis.errorMessage}</div>
                        </div>
                    </div>`
                            : ""
                    }
                </div>`;
                })
                .join("")}
        </div>
        `
                : `
        <!-- No Analysis Results -->
        <div class="section">
            <div class="section-title">
                <span class="section-icon">🔍</span>
                ANALYSIS RESULTS
            </div>
            <div class="no-analysis">
                <div class="no-analysis-icon">📋</div>
                <h3>No Analysis Results Available</h3>
                <p>This video has not been analyzed yet or analysis is still in progress.</p>
            </div>
        </div>
        `
        }

        <!-- Footer -->
        <div class="footer">
            <div class="brand">🛡️ VidVigilante - Advanced Deepfake Detection System</div>
            <div class="confidential">
                <span>🔒</span>
                <span>CONFIDENTIAL ANALYSIS REPORT</span>
            </div>
        </div>
    </div>
</body>
</html>`;
};

// Main PDF generation function using html2pdf.js
export const generateAndDownloadPDF = async (video) => {
    try {
        // Generate the HTML content
        const htmlContent = generateReportHTML(video);

        // Create a temporary container with the full HTML document
        const tempDiv = document.createElement("div");
        tempDiv.innerHTML = htmlContent;

        // Get the actual content (skip html, head tags)
        const reportContainer =
            tempDiv.querySelector(".report-container") ||
            tempDiv.firstElementChild;

        if (!reportContainer) {
            throw new Error("Could not find report content");
        }

        // Configure html2pdf options for better compatibility
        const options = {
            margin: 0.5,
            filename: `${video.filename.replace(
                /\.[^/.]+$/,
                ""
            )}_analysis_report.pdf`,
            image: {
                type: "jpeg",
                quality: 0.98,
            },
            html2canvas: {
                scale: 1.5,
                useCORS: true,
                logging: false,
                letterRendering: true,
                allowTaint: true,
                backgroundColor: "#ffffff",
            },
            jsPDF: {
                unit: "in",
                format: "a4",
                orientation: "portrait",
            },
            pagebreak: {
                mode: ["avoid-all", "css"],
            },
        };

        // Generate and download the PDF
        await html2pdf().set(options).from(reportContainer).save();
    } catch (error) {
        console.error("Error generating PDF:", error);

        // Fallback to print method if html2pdf fails
        console.log("Falling back to print method...");
        await generateAndDownloadPDFPrint(video);
    }
};

// Alternative method using print dialog for better browser compatibility
export const generateAndDownloadPDFPrint = async (video) => {
    try {
        const htmlContent = generateReportHTML(video);

        // Create a new window for printing
        const printWindow = window.open("", "_blank", "width=800,height=600");

        if (!printWindow) {
            throw new Error(
                "Popup blocked. Please allow popups and try again."
            );
        }

        printWindow.document.write(htmlContent);
        printWindow.document.close();

        // Wait for content to load, then print
        printWindow.onload = () => {
            printWindow.focus();
            printWindow.print();

            // Close the window after a delay
            setTimeout(() => {
                printWindow.close();
            }, 1000);
        };

        // Handle cases where onload doesn't fire
        setTimeout(() => {
            if (printWindow && !printWindow.closed) {
                printWindow.focus();
                printWindow.print();
            }
        }, 500);
    } catch (error) {
        console.error("Error generating PDF via print:", error);
        throw new Error(`Failed to generate PDF report: ${error.message}`);
    }
};

// Keep existing functions unchanged
export const downloadVideo = async (videoUrl, filename) => {
    try {
        const response = await fetch(videoUrl);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

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
        throw new Error(`Failed to download video: ${error.message}`);
    }
};

export const downloadHTMLReport = async (video) => {
    try {
        const htmlContent = generateReportHTML(video);
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
        link.style.display = "none";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

        console.log("HTML report downloaded successfully");
    } catch (error) {
        console.error("Error generating HTML report:", error);
        throw new Error(`Failed to generate HTML report: ${error.message}`);
    }
};
