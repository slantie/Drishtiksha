// src/scripts/check-stuck-runs.js
// Utility script to check and finalize stuck analysis runs

import { prisma } from "../config/index.js";
import { mediaRepository } from "../repositories/media.repository.js";
import logger from "../utils/logger.js";

/**
 * Checks for analysis runs that are stuck in PROCESSING or QUEUED state
 * and finalizes them based on their actual analysis results.
 */
async function checkAndFinalizeStuckRuns() {
  try {
    logger.info("[StuckRuns] Checking for stuck analysis runs...");

    // Find all runs that are in PROCESSING or QUEUED state
    const stuckRuns = await prisma.analysisRun.findMany({
      where: {
        status: {
          in: ["PROCESSING", "QUEUED"],
        },
        // Check runs older than 30 seconds (reduced from 5 minutes for faster detection)
        createdAt: {
          lt: new Date(Date.now() - 30 * 1000),
        },
      },
      include: {
        analyses: { select: { status: true, modelName: true } },
        media: { select: { id: true, filename: true, userId: true } },
      },
    });

    if (stuckRuns.length === 0) {
      logger.info("[StuckRuns] No stuck runs found.");
      return { checked: 0, finalized: 0 };
    }

    logger.info(
      `[StuckRuns] Found ${stuckRuns.length} potentially stuck runs. Checking...`
    );

    let finalizedCount = 0;

    for (const run of stuckRuns) {
      const totalAnalyses = run.analyses.length;
      const completedAnalyses = run.analyses.filter(
        (a) => a.status === "COMPLETED"
      ).length;
      const failedAnalyses = run.analyses.filter(
        (a) => a.status === "FAILED"
      ).length;
      const pendingAnalyses = run.analyses.filter(
        (a) => a.status === "PENDING"
      ).length;

      // 🔧 NEW LOGIC: Check if ALL analyses are done (completed or failed)
      const allAnalysesDone = pendingAnalyses === 0 && (completedAnalyses + failedAnalyses) === totalAnalyses;
      
      // Skip if there are still pending analyses AND the run is recent (less than 2 minutes old)
      if (pendingAnalyses > 0) {
        const runAge = Date.now() - new Date(run.createdAt).getTime();
        if (runAge < 2 * 60 * 1000) {
          logger.info(
            `[StuckRuns] Run ${run.id} has ${pendingAnalyses} pending analyses but is only ${Math.round(runAge / 1000)}s old. Waiting...`
          );
          continue;
        } else if (pendingAnalyses === totalAnalyses) {
          logger.warn(
            `[StuckRuns] Run ${run.id} has all analyses still pending after ${Math.round(runAge / 1000)}s. Skipping.`
          );
          continue;
        }
      }

      // Determine final status
      let finalStatus;
      if (completedAnalyses === 0 && failedAnalyses === 0) {
        // No analyses have started - keep as is
        logger.warn(
          `[StuckRuns] Run ${run.id} has no completed or failed analyses. Keeping as ${run.status}.`
        );
        continue;
      } else if (completedAnalyses > 0) {
        // At least one analysis succeeded
        finalStatus = "ANALYZED";
      } else {
        // All analyses failed
        finalStatus = "FAILED";
      }

      // Update the run and media status
      await prisma.$transaction(async (tx) => {
        await tx.analysisRun.update({
          where: { id: run.id },
          data: { status: finalStatus },
        });

        await tx.media.update({
          where: { id: run.media.id },
          data: {
            status: finalStatus,
            latestAnalysisRunId: run.id,
          },
        });
      });

      finalizedCount++;
      logger.info(
        `[StuckRuns] ✅ Finalized stuck run ${run.id} (media: ${run.media.filename}) with status ${finalStatus} (${completedAnalyses}/${totalAnalyses} completed, ${failedAnalyses} failed, ${pendingAnalyses} pending)`
      );
    }

    logger.info(
      `[StuckRuns] ✅ Check complete. Finalized ${finalizedCount} out of ${stuckRuns.length} stuck runs.`
    );
    return { checked: stuckRuns.length, finalized: finalizedCount };
  } catch (error) {
    logger.error(
      `[StuckRuns] Error checking stuck runs: ${error.message}`,
      { stack: error.stack }
    );
    throw error;
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  checkAndFinalizeStuckRuns()
    .then((result) => {
      console.log(`✅ Checked ${result.checked} runs, finalized ${result.finalized}`);
      process.exit(0);
    })
    .catch((error) => {
      console.error("❌ Error:", error);
      process.exit(1);
    })
    .finally(() => {
      prisma.$disconnect();
    });
}

export { checkAndFinalizeStuckRuns };
