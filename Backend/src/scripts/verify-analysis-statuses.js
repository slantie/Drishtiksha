// src/scripts/verify-analysis-statuses.js
// Diagnostic script to verify all analysis statuses are correct

import { prisma } from "../config/index.js";
import logger from "../utils/logger.js";

/**
 * Verifies that all analysis-related statuses are in sync
 */
async function verifyAnalysisStatuses() {
  try {
    logger.info("[StatusVerify] Starting comprehensive status verification...");

    // Get all media items
    const allMedia = await prisma.media.findMany({
      include: {
        analysisRuns: {
          include: {
            analyses: {
              select: { id: true, modelName: true, status: true },
            },
          },
        },
      },
    });

    let totalMedia = 0;
    let correctMedia = 0;
    let incorrectMedia = 0;
    const issues = [];

    for (const media of allMedia) {
      totalMedia++;

      // Get the latest run
      const latestRun = media.analysisRuns.sort(
        (a, b) => new Date(b.createdAt) - new Date(a.createdAt)
      )[0];

      if (!latestRun) {
        // No runs yet - media should be QUEUED
        if (media.status !== "QUEUED") {
          incorrectMedia++;
          issues.push({
            mediaId: media.id,
            filename: media.filename,
            issue: `No analysis runs but status is ${media.status} (expected QUEUED)`,
            currentMediaStatus: media.status,
            expectedMediaStatus: "QUEUED",
          });
        } else {
          correctMedia++;
        }
        continue;
      }

      // Analyze the latest run
      const totalAnalyses = latestRun.analyses.length;
      const completedAnalyses = latestRun.analyses.filter(
        (a) => a.status === "COMPLETED"
      ).length;
      const failedAnalyses = latestRun.analyses.filter(
        (a) => a.status === "FAILED"
      ).length;
      const pendingAnalyses = latestRun.analyses.filter(
        (a) => a.status === "PENDING"
      ).length;

      // Determine what the statuses SHOULD be
      let expectedRunStatus;
      let expectedMediaStatus;

      if (totalAnalyses === 0) {
        expectedRunStatus = "QUEUED";
        expectedMediaStatus = "QUEUED";
      } else if (pendingAnalyses > 0 || (completedAnalyses === 0 && failedAnalyses === 0)) {
        expectedRunStatus = "PROCESSING";
        expectedMediaStatus = "PROCESSING";
      } else if (completedAnalyses > 0) {
        expectedRunStatus = "ANALYZED";
        expectedMediaStatus = "ANALYZED";
      } else {
        // All failed
        expectedRunStatus = "FAILED";
        expectedMediaStatus = "FAILED";
      }

      // Check if statuses match
      const runStatusCorrect = latestRun.status === expectedRunStatus;
      const mediaStatusCorrect = media.status === expectedMediaStatus;

      if (runStatusCorrect && mediaStatusCorrect) {
        correctMedia++;
      } else {
        incorrectMedia++;
        issues.push({
          mediaId: media.id,
          filename: media.filename,
          runId: latestRun.id,
          issue: "Status mismatch",
          currentRunStatus: latestRun.status,
          expectedRunStatus: expectedRunStatus,
          currentMediaStatus: media.status,
          expectedMediaStatus: expectedMediaStatus,
          analyses: {
            total: totalAnalyses,
            completed: completedAnalyses,
            failed: failedAnalyses,
            pending: pendingAnalyses,
          },
        });
      }
    }

    // Report results
    logger.info(
      `[StatusVerify] ✅ Verification complete: ${correctMedia}/${totalMedia} media items have correct statuses`
    );

    if (incorrectMedia > 0) {
      logger.warn(
        `[StatusVerify] ⚠️ Found ${incorrectMedia} media items with incorrect statuses:`
      );
      issues.forEach((issue, index) => {
        logger.warn(`[StatusVerify] Issue ${index + 1}:`, issue);
      });
    }

    return {
      totalMedia,
      correctMedia,
      incorrectMedia,
      issues,
    };
  } catch (error) {
    logger.error(
      `[StatusVerify] Error verifying statuses: ${error.message}`,
      { stack: error.stack }
    );
    throw error;
  }
}

/**
 * Fixes incorrect statuses found during verification
 */
async function fixIncorrectStatuses(issues) {
  logger.info(`[StatusFix] Fixing ${issues.length} status issues...`);

  let fixed = 0;
  let failed = 0;

  for (const issue of issues) {
    try {
      await prisma.$transaction(async (tx) => {
        // Update run status if needed
        if (issue.runId && issue.expectedRunStatus) {
          await tx.analysisRun.update({
            where: { id: issue.runId },
            data: { status: issue.expectedRunStatus },
          });
        }

        // Update media status
        await tx.media.update({
          where: { id: issue.mediaId },
          data: { 
            status: issue.expectedMediaStatus,
            latestAnalysisRunId: issue.runId || undefined,
          },
        });
      });

      fixed++;
      logger.info(
        `[StatusFix] ✅ Fixed status for media ${issue.filename} (${issue.mediaId})`
      );
    } catch (error) {
      failed++;
      logger.error(
        `[StatusFix] ❌ Failed to fix status for media ${issue.filename}: ${error.message}`
      );
    }
  }

  logger.info(
    `[StatusFix] ✅ Fix complete: ${fixed} fixed, ${failed} failed`
  );
  return { fixed, failed };
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  (async () => {
    try {
      const result = await verifyAnalysisStatuses();

      if (result.incorrectMedia > 0) {
        console.log("\n⚠️ Found status issues. Fix them? (y/n)");
        process.stdin.once("data", async (data) => {
          const answer = data.toString().trim().toLowerCase();
          if (answer === "y" || answer === "yes") {
            const fixResult = await fixIncorrectStatuses(result.issues);
            console.log(
              `✅ Fixed ${fixResult.fixed} issues, ${fixResult.failed} failures`
            );
          }
          await prisma.$disconnect();
          process.exit(0);
        });
      } else {
        console.log("✅ All statuses are correct!");
        await prisma.$disconnect();
        process.exit(0);
      }
    } catch (error) {
      console.error("❌ Error:", error);
      await prisma.$disconnect();
      process.exit(1);
    }
  })();
}

export { verifyAnalysisStatuses, fixIncorrectStatuses };
