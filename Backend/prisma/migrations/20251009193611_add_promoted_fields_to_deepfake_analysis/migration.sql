/*
  Warnings:

  - The values [PROCESSING] on the enum `AnalysisStatus` will be removed. If these variants are still used in the database, this will fail.
  - The values [MODATOR] on the enum `Role` will be removed. If these variants are still used in the database, this will fail.
  - You are about to drop the column `analysisType` on the `deepfake_analyses` table. All the data in the column will be lost.
  - You are about to drop the column `createdAt` on the `deepfake_analyses` table. All the data in the column will be lost.
  - You are about to drop the column `model` on the `deepfake_analyses` table. All the data in the column will be lost.
  - You are about to drop the column `modelVersion` on the `deepfake_analyses` table. All the data in the column will be lost.
  - You are about to drop the column `processingTime` on the `deepfake_analyses` table. All the data in the column will be lost.
  - You are about to drop the column `timestamp` on the `deepfake_analyses` table. All the data in the column will be lost.
  - You are about to drop the column `updatedAt` on the `deepfake_analyses` table. All the data in the column will be lost.
  - You are about to drop the column `videoId` on the `deepfake_analyses` table. All the data in the column will be lost.
  - You are about to drop the column `visualizedUrl` on the `deepfake_analyses` table. All the data in the column will be lost.
  - You are about to drop the column `availableModels` on the `server_health` table. All the data in the column will be lost.
  - You are about to drop the column `avgProcessingTime` on the `server_health` table. All the data in the column will be lost.
  - You are about to drop the column `createdAt` on the `server_health` table. All the data in the column will be lost.
  - You are about to drop the column `errorMessage` on the `server_health` table. All the data in the column will be lost.
  - You are about to drop the column `gpuInfo` on the `server_health` table. All the data in the column will be lost.
  - You are about to drop the column `lastHealthCheck` on the `server_health` table. All the data in the column will be lost.
  - You are about to drop the column `loadMetrics` on the `server_health` table. All the data in the column will be lost.
  - You are about to drop the column `modelStates` on the `server_health` table. All the data in the column will be lost.
  - You are about to drop the column `requestCount` on the `server_health` table. All the data in the column will be lost.
  - You are about to drop the column `responseTime` on the `server_health` table. All the data in the column will be lost.
  - You are about to drop the column `serverUrl` on the `server_health` table. All the data in the column will be lost.
  - You are about to drop the column `systemResources` on the `server_health` table. All the data in the column will be lost.
  - You are about to drop the column `uptime` on the `server_health` table. All the data in the column will be lost.
  - You are about to drop the column `version` on the `server_health` table. All the data in the column will be lost.
  - You are about to drop the column `avatar` on the `users` table. All the data in the column will be lost.
  - You are about to drop the column `bio` on the `users` table. All the data in the column will be lost.
  - You are about to drop the column `phone` on the `users` table. All the data in the column will be lost.
  - You are about to drop the `analysis_details` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `analysis_errors` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `frame_analysis` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `model_info` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `refresh_tokens` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `system_info` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `temporal_analysis` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `videos` table. If the table is not empty, all the data it contains will be lost.
  - Added the required column `analysisRunId` to the `deepfake_analyses` table without a default value. This is not possible if the table is not empty.
  - Added the required column `modelName` to the `deepfake_analyses` table without a default value. This is not possible if the table is not empty.
  - Added the required column `result_payload` to the `deepfake_analyses` table without a default value. This is not possible if the table is not empty.
  - Added the required column `updated_at` to the `deepfake_analyses` table without a default value. This is not possible if the table is not empty.
  - Changed the type of `prediction` on the `deepfake_analyses` table. No cast exists, the column would be dropped and recreated, which cannot be done if there is data, since the column is required.
  - Added the required column `responseTimeMs` to the `server_health` table without a default value. This is not possible if the table is not empty.
  - Added the required column `statsPayload` to the `server_health` table without a default value. This is not possible if the table is not empty.
  - Changed the type of `status` on the `server_health` table. No cast exists, the column would be dropped and recreated, which cannot be done if there is data, since the column is required.

*/
-- CreateEnum
CREATE TYPE "public"."MediaStatus" AS ENUM ('QUEUED', 'PROCESSING', 'ANALYZED', 'FAILED');

-- CreateEnum
CREATE TYPE "public"."MediaType" AS ENUM ('VIDEO', 'IMAGE', 'AUDIO', 'UNKNOWN');

-- AlterEnum
BEGIN;
CREATE TYPE "public"."AnalysisStatus_new" AS ENUM ('PENDING', 'COMPLETED', 'FAILED');
ALTER TABLE "public"."deepfake_analyses" ALTER COLUMN "status" DROP DEFAULT;
ALTER TABLE "public"."deepfake_analyses" ALTER COLUMN "status" TYPE "public"."AnalysisStatus_new" USING ("status"::text::"public"."AnalysisStatus_new");
ALTER TYPE "public"."AnalysisStatus" RENAME TO "AnalysisStatus_old";
ALTER TYPE "public"."AnalysisStatus_new" RENAME TO "AnalysisStatus";
DROP TYPE "public"."AnalysisStatus_old";
ALTER TABLE "public"."deepfake_analyses" ALTER COLUMN "status" SET DEFAULT 'PENDING';
COMMIT;

-- AlterEnum
BEGIN;
CREATE TYPE "public"."Role_new" AS ENUM ('USER', 'ADMIN');
ALTER TABLE "public"."users" ALTER COLUMN "role" DROP DEFAULT;
ALTER TABLE "public"."users" ALTER COLUMN "role" TYPE "public"."Role_new" USING ("role"::text::"public"."Role_new");
ALTER TYPE "public"."Role" RENAME TO "Role_old";
ALTER TYPE "public"."Role_new" RENAME TO "Role";
DROP TYPE "public"."Role_old";
ALTER TABLE "public"."users" ALTER COLUMN "role" SET DEFAULT 'USER';
COMMIT;

-- DropForeignKey
ALTER TABLE "public"."analysis_details" DROP CONSTRAINT "analysis_details_analysisId_fkey";

-- DropForeignKey
ALTER TABLE "public"."analysis_errors" DROP CONSTRAINT "analysis_errors_analysisId_fkey";

-- DropForeignKey
ALTER TABLE "public"."deepfake_analyses" DROP CONSTRAINT "deepfake_analyses_videoId_fkey";

-- DropForeignKey
ALTER TABLE "public"."frame_analysis" DROP CONSTRAINT "frame_analysis_analysisId_fkey";

-- DropForeignKey
ALTER TABLE "public"."model_info" DROP CONSTRAINT "model_info_analysisId_fkey";

-- DropForeignKey
ALTER TABLE "public"."refresh_tokens" DROP CONSTRAINT "refresh_tokens_userId_fkey";

-- DropForeignKey
ALTER TABLE "public"."system_info" DROP CONSTRAINT "system_info_analysisId_fkey";

-- DropForeignKey
ALTER TABLE "public"."temporal_analysis" DROP CONSTRAINT "temporal_analysis_analysisId_fkey";

-- DropForeignKey
ALTER TABLE "public"."videos" DROP CONSTRAINT "videos_userId_fkey";

-- DropIndex
DROP INDEX "public"."deepfake_analyses_videoId_model_analysisType_idx";

-- AlterTable
ALTER TABLE "public"."deepfake_analyses" DROP COLUMN "analysisType",
DROP COLUMN "createdAt",
DROP COLUMN "model",
DROP COLUMN "modelVersion",
DROP COLUMN "processingTime",
DROP COLUMN "timestamp",
DROP COLUMN "updatedAt",
DROP COLUMN "videoId",
DROP COLUMN "visualizedUrl",
ADD COLUMN     "analysisRunId" TEXT NOT NULL,
ADD COLUMN     "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN     "media_type" TEXT,
ADD COLUMN     "modelName" TEXT NOT NULL,
ADD COLUMN     "processing_time" DOUBLE PRECISION,
ADD COLUMN     "result_payload" JSONB NOT NULL,
ADD COLUMN     "updated_at" TIMESTAMP(3) NOT NULL,
DROP COLUMN "prediction",
ADD COLUMN     "prediction" TEXT NOT NULL;

-- AlterTable
ALTER TABLE "public"."server_health" DROP COLUMN "availableModels",
DROP COLUMN "avgProcessingTime",
DROP COLUMN "createdAt",
DROP COLUMN "errorMessage",
DROP COLUMN "gpuInfo",
DROP COLUMN "lastHealthCheck",
DROP COLUMN "loadMetrics",
DROP COLUMN "modelStates",
DROP COLUMN "requestCount",
DROP COLUMN "responseTime",
DROP COLUMN "serverUrl",
DROP COLUMN "systemResources",
DROP COLUMN "uptime",
DROP COLUMN "version",
ADD COLUMN     "checkedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN     "responseTimeMs" INTEGER NOT NULL,
ADD COLUMN     "statsPayload" JSONB NOT NULL,
DROP COLUMN "status",
ADD COLUMN     "status" TEXT NOT NULL;

-- AlterTable
ALTER TABLE "public"."users" DROP COLUMN "avatar",
DROP COLUMN "bio",
DROP COLUMN "phone";

-- DropTable
DROP TABLE "public"."analysis_details";

-- DropTable
DROP TABLE "public"."analysis_errors";

-- DropTable
DROP TABLE "public"."frame_analysis";

-- DropTable
DROP TABLE "public"."model_info";

-- DropTable
DROP TABLE "public"."refresh_tokens";

-- DropTable
DROP TABLE "public"."system_info";

-- DropTable
DROP TABLE "public"."temporal_analysis";

-- DropTable
DROP TABLE "public"."videos";

-- DropEnum
DROP TYPE "public"."AnalysisType";

-- DropEnum
DROP TYPE "public"."PredictionType";

-- DropEnum
DROP TYPE "public"."ServerStatus";

-- DropEnum
DROP TYPE "public"."VideoStatus";

-- CreateTable
CREATE TABLE "public"."media" (
    "id" TEXT NOT NULL,
    "filename" TEXT NOT NULL,
    "url" TEXT NOT NULL,
    "publicId" TEXT NOT NULL,
    "mimetype" TEXT NOT NULL,
    "size" INTEGER NOT NULL,
    "description" TEXT,
    "status" "public"."MediaStatus" NOT NULL DEFAULT 'QUEUED',
    "mediaType" "public"."MediaType" NOT NULL,
    "userId" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "latestAnalysisRunId" TEXT,

    CONSTRAINT "media_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."analysis_runs" (
    "id" TEXT NOT NULL,
    "mediaId" TEXT NOT NULL,
    "runNumber" INTEGER NOT NULL,
    "status" "public"."MediaStatus" NOT NULL DEFAULT 'QUEUED',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "analysis_runs_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "media_userId_createdAt_idx" ON "public"."media"("userId", "createdAt" DESC);

-- CreateIndex
CREATE UNIQUE INDEX "analysis_runs_mediaId_runNumber_key" ON "public"."analysis_runs"("mediaId", "runNumber");

-- CreateIndex
CREATE INDEX "deepfake_analyses_analysisRunId_idx" ON "public"."deepfake_analyses"("analysisRunId");

-- CreateIndex
CREATE INDEX "deepfake_analyses_modelName_idx" ON "public"."deepfake_analyses"("modelName");

-- CreateIndex
CREATE INDEX "deepfake_analyses_processing_time_idx" ON "public"."deepfake_analyses"("processing_time");

-- CreateIndex
CREATE INDEX "deepfake_analyses_media_type_idx" ON "public"."deepfake_analyses"("media_type");

-- CreateIndex
CREATE INDEX "deepfake_analyses_confidence_idx" ON "public"."deepfake_analyses"("confidence");

-- AddForeignKey
ALTER TABLE "public"."media" ADD CONSTRAINT "media_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."analysis_runs" ADD CONSTRAINT "analysis_runs_mediaId_fkey" FOREIGN KEY ("mediaId") REFERENCES "public"."media"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."deepfake_analyses" ADD CONSTRAINT "deepfake_analyses_analysisRunId_fkey" FOREIGN KEY ("analysisRunId") REFERENCES "public"."analysis_runs"("id") ON DELETE CASCADE ON UPDATE CASCADE;
