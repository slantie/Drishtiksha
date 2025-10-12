// src/api/pdf/pdf.controller.js

import { pdfService } from '../../services/pdf.service.js';
import { asyncHandler } from '../../utils/asyncHandler.js';
import logger from '../../utils/logger.js';

/**
 * Generate and download PDF report for a specific analysis run
 * Takes analysis run ID, fetches data, generates markdown, creates PDF, and sends for download
 */
const generateRunReport = asyncHandler(async (req, res) => {
    const { analysisRunId } = req.params;
    const userId = req.user.id;
    
    logger.info(`[PDF Controller] Generating PDF report for analysis run ${analysisRunId}, user ${userId}`);
    
    // Generate PDF buffer (service handles everything: fetch data, generate markdown, create PDF)
    const { pdfBuffer, filename } = await pdfService.generateRunReportPDF(analysisRunId, userId);
    
    // Verify buffer is valid
    if (!pdfBuffer || pdfBuffer.length === 0) {
        throw new Error('Generated PDF buffer is empty');
    }
    
    // Log buffer info for debugging
    logger.info(`[PDF Controller] PDF buffer generated - Size: ${pdfBuffer.length} bytes, Type: ${pdfBuffer.constructor.name}`);
    
    // Set headers for download
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
    res.setHeader('Content-Length', Buffer.byteLength(pdfBuffer));
    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');

    // Send PDF buffer as binary and end response
    // Use binary encoding to ensure no character encoding is applied
    res.status(200).end(pdfBuffer, 'binary');
    
    logger.info(`[PDF Controller] PDF report sent successfully: ${filename}`);
});

/**
 * Debug endpoint: Generate a simple test PDF
 * Helps diagnose if the issue is with data fetching or PDF generation itself
 */
const generateTestPDF = asyncHandler(async (req, res) => {
    logger.info('[PDF Controller] Generating test PDF');
    
    // Generate a minimal test markdown
    const testMarkdown = `# Test PDF Report

This is a simple test PDF to verify the generation pipeline.

## Test Section

- Item 1
- Item 2
- Item 3

**Generated at:** ${new Date().toISOString()}`;
    
    // Generate PDF from test markdown
    const pdfBuffer = await pdfService.generatePDFFromMarkdown(testMarkdown);
    
    // Verify buffer
    if (!pdfBuffer || pdfBuffer.length === 0) {
        throw new Error('Test PDF buffer is empty');
    }
    
    logger.info(`[PDF Controller] Test PDF generated - Size: ${pdfBuffer.length} bytes`);
    
    // Set headers
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', 'attachment; filename="test-report.pdf"');
    res.setHeader('Content-Length', Buffer.byteLength(pdfBuffer));
    res.setHeader('Cache-Control', 'no-cache');
    
    // Send binary data
    res.status(200).end(pdfBuffer, 'binary');
    
    logger.info('[PDF Controller] Test PDF sent successfully');
});

export const pdfController = {
    generateRunReport,
    generateTestPDF,
};
