-- CreateEnum
CREATE TYPE "public"."Role" AS ENUM ('USER', 'ADMIN', 'MODERATOR');

-- CreateEnum
CREATE TYPE "public"."VideoStatus" AS ENUM ('UPLOADED', 'QUEUED', 'PROCESSING', 'ANALYZED', 'FAILED');

-- CreateEnum
CREATE TYPE "public"."PredictionType" AS ENUM ('REAL', 'FAKE');

-- CreateEnum
CREATE TYPE "public"."AnalysisStatus" AS ENUM ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED');

-- CreateEnum
CREATE TYPE "public"."AnalysisType" AS ENUM ('QUICK', 'DETAILED', 'FRAMES', 'VISUALIZE', 'COMPREHENSIVE');

-- CreateEnum
CREATE TYPE "public"."AnalysisModel" AS ENUM ('SIGLIP_LSTM_V1', 'SIGLIP_LSTM_V3', 'COLOR_CUES_LSTM_V1');

-- CreateEnum
CREATE TYPE "public"."ServerStatus" AS ENUM ('HEALTHY', 'DEGRADED', 'UNHEALTHY', 'MAINTENANCE', 'UNKNOWN');

-- CreateTable
CREATE TABLE "public"."users" (
    "id" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "firstName" TEXT NOT NULL,
    "lastName" TEXT NOT NULL,
    "password" TEXT NOT NULL,
    "bio" TEXT,
    "phone" TEXT,
    "avatar" TEXT,
    "role" "public"."Role" NOT NULL DEFAULT 'USER',
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "users_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."refresh_tokens" (
    "id" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "refresh_tokens_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."videos" (
    "id" TEXT NOT NULL,
    "filename" TEXT NOT NULL,
    "url" TEXT NOT NULL,
    "visualizedUrl" TEXT,
    "publicId" TEXT NOT NULL,
    "mimetype" TEXT NOT NULL,
    "size" INTEGER NOT NULL,
    "description" TEXT,
    "status" "public"."VideoStatus" NOT NULL DEFAULT 'UPLOADED',
    "userId" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "videos_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."deepfake_analyses" (
    "id" TEXT NOT NULL,
    "videoId" TEXT NOT NULL,
    "prediction" "public"."PredictionType" NOT NULL,
    "confidence" DOUBLE PRECISION NOT NULL,
    "processingTime" DOUBLE PRECISION,
    "model" "public"."AnalysisModel" NOT NULL,
    "modelVersion" TEXT,
    "analysisType" "public"."AnalysisType" NOT NULL DEFAULT 'QUICK',
    "status" "public"."AnalysisStatus" NOT NULL DEFAULT 'PENDING',
    "errorMessage" TEXT,
    "timestamp" TIMESTAMP(3),
    "visualizedUrl" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "deepfake_analyses_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."analysis_details" (
    "id" TEXT NOT NULL,
    "analysisId" TEXT NOT NULL,
    "frameCount" INTEGER NOT NULL,
    "avgConfidence" DOUBLE PRECISION NOT NULL,
    "confidenceStd" DOUBLE PRECISION NOT NULL,
    "temporalConsistency" DOUBLE PRECISION,
    "rollingAverage" DOUBLE PRECISION,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "analysis_details_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."frame_analysis" (
    "id" TEXT NOT NULL,
    "analysisId" TEXT NOT NULL,
    "frameNumber" INTEGER NOT NULL,
    "confidence" DOUBLE PRECISION NOT NULL,
    "prediction" "public"."PredictionType" NOT NULL,
    "timestamp" DOUBLE PRECISION,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "frame_analysis_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."temporal_analysis" (
    "id" TEXT NOT NULL,
    "analysisId" TEXT NOT NULL,
    "consistencyScore" DOUBLE PRECISION NOT NULL,
    "patternDetection" TEXT,
    "anomalyFrames" INTEGER[],
    "confidenceTrend" TEXT,
    "totalFrames" INTEGER NOT NULL,
    "fakeFrames" INTEGER NOT NULL,
    "realFrames" INTEGER NOT NULL,
    "avgConfidence" DOUBLE PRECISION NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "temporal_analysis_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."model_info" (
    "id" TEXT NOT NULL,
    "analysisId" TEXT NOT NULL,
    "version" TEXT NOT NULL,
    "architecture" TEXT NOT NULL,
    "device" TEXT NOT NULL,
    "batchSize" INTEGER,
    "numFrames" INTEGER,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "model_info_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."system_info" (
    "id" TEXT NOT NULL,
    "analysisId" TEXT NOT NULL,
    "gpuMemoryUsed" TEXT,
    "processingDevice" TEXT,
    "cudaAvailable" BOOLEAN,
    "systemMemoryUsed" TEXT,
    "loadBalancingInfo" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "system_info_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."analysis_errors" (
    "id" TEXT NOT NULL,
    "analysisId" TEXT NOT NULL,
    "errorType" TEXT NOT NULL,
    "errorMessage" TEXT NOT NULL,
    "availableModels" TEXT[],
    "suggestions" TEXT[],
    "stackTrace" TEXT,
    "serverResponse" JSONB,
    "retryAttempt" INTEGER NOT NULL DEFAULT 0,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "analysis_errors_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."server_health" (
    "id" TEXT NOT NULL,
    "serverUrl" TEXT NOT NULL,
    "status" "public"."ServerStatus" NOT NULL DEFAULT 'UNKNOWN',
    "availableModels" TEXT[],
    "loadMetrics" JSONB,
    "gpuInfo" JSONB,
    "lastHealthCheck" TIMESTAMP(3) NOT NULL,
    "responseTime" DOUBLE PRECISION,
    "errorMessage" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "server_health_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "users_email_key" ON "public"."users"("email");

-- CreateIndex
CREATE UNIQUE INDEX "refresh_tokens_token_key" ON "public"."refresh_tokens"("token");

-- CreateIndex
CREATE UNIQUE INDEX "analysis_details_analysisId_key" ON "public"."analysis_details"("analysisId");

-- CreateIndex
CREATE UNIQUE INDEX "frame_analysis_analysisId_frameNumber_key" ON "public"."frame_analysis"("analysisId", "frameNumber");

-- CreateIndex
CREATE UNIQUE INDEX "temporal_analysis_analysisId_key" ON "public"."temporal_analysis"("analysisId");

-- CreateIndex
CREATE UNIQUE INDEX "model_info_analysisId_key" ON "public"."model_info"("analysisId");

-- CreateIndex
CREATE UNIQUE INDEX "system_info_analysisId_key" ON "public"."system_info"("analysisId");

-- AddForeignKey
ALTER TABLE "public"."refresh_tokens" ADD CONSTRAINT "refresh_tokens_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."videos" ADD CONSTRAINT "videos_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."deepfake_analyses" ADD CONSTRAINT "deepfake_analyses_videoId_fkey" FOREIGN KEY ("videoId") REFERENCES "public"."videos"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."analysis_details" ADD CONSTRAINT "analysis_details_analysisId_fkey" FOREIGN KEY ("analysisId") REFERENCES "public"."deepfake_analyses"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."frame_analysis" ADD CONSTRAINT "frame_analysis_analysisId_fkey" FOREIGN KEY ("analysisId") REFERENCES "public"."deepfake_analyses"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."temporal_analysis" ADD CONSTRAINT "temporal_analysis_analysisId_fkey" FOREIGN KEY ("analysisId") REFERENCES "public"."deepfake_analyses"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."model_info" ADD CONSTRAINT "model_info_analysisId_fkey" FOREIGN KEY ("analysisId") REFERENCES "public"."deepfake_analyses"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."system_info" ADD CONSTRAINT "system_info_analysisId_fkey" FOREIGN KEY ("analysisId") REFERENCES "public"."deepfake_analyses"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."analysis_errors" ADD CONSTRAINT "analysis_errors_analysisId_fkey" FOREIGN KEY ("analysisId") REFERENCES "public"."deepfake_analyses"("id") ON DELETE CASCADE ON UPDATE CASCADE;
