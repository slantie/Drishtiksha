// Quick script to check the structure of analysis data

import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function checkAnalysisData() {
    try {
        const analysis = await prisma.deepfakeAnalysis.findFirst({
            where: { status: 'COMPLETED' },
            orderBy: { createdAt: 'desc' }
        });

        if (analysis) {
            console.log('=== Latest Completed Analysis ===');
            console.log('ID:', analysis.id);
            console.log('Model:', analysis.modelName);
            console.log('Prediction:', analysis.prediction);
            console.log('Confidence:', analysis.confidence);
            console.log('Processing Time:', analysis.processingTime);
            console.log('\n=== Result Payload Structure ===');
            console.log(JSON.stringify(analysis.resultPayload, null, 2));
        } else {
            console.log('No completed analyses found');
        }
    } catch (error) {
        console.error('Error:', error.message);
    } finally {
        await prisma.$disconnect();
    }
}

checkAnalysisData();
