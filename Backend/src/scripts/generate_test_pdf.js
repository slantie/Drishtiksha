import fs from 'fs';
import path from 'path';
import { pdfService } from '../services/pdf.service.js';

const outPath = path.resolve('pdf-assets', 'test-backend-output.pdf');

(async () => {
  try {
    const markdown = '# Test PDF\n\nThis is a backend PDF generation test.';
    const buf = await pdfService.generatePDFFromMarkdown(markdown);
    console.log('Type:', Object.prototype.toString.call(buf));
    console.log('Length:', buf.length);
    fs.writeFileSync(outPath, buf);
    const fd = fs.openSync(outPath, 'r');
    const header = Buffer.alloc(8);
    fs.readSync(fd, header, 0, 8, 0);
    fs.closeSync(fd);
    console.log('Header bytes:', header.toString('utf8'));
    console.log('Wrote:', outPath);
  } catch (err) {
    console.error('Error generating PDF:', err);
    process.exit(1);
  }
})();
