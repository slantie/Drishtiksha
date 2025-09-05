/*
  Warnings:

  - Changed the type of `model` on the `deepfake_analyses` table. No cast exists, the column would be dropped and recreated, which cannot be done if there is data, since the column is required.

*/
-- AlterTable
ALTER TABLE "public"."deepfake_analyses" DROP COLUMN "model",
ADD COLUMN     "model" TEXT NOT NULL;

-- DropEnum
DROP TYPE "public"."AnalysisModel";

-- CreateIndex
CREATE INDEX "deepfake_analyses_videoId_model_analysisType_idx" ON "public"."deepfake_analyses"("videoId", "model", "analysisType");
