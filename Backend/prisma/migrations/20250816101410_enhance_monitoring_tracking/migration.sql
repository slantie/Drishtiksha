/*
  Warnings:

  - Added the required column `modelName` to the `model_info` table without a default value. This is not possible if the table is not empty.

*/
-- AlterTable
ALTER TABLE "public"."model_info" ADD COLUMN     "loadTime" DOUBLE PRECISION,
ADD COLUMN     "memoryUsage" TEXT,
ADD COLUMN     "modelName" TEXT NOT NULL,
ADD COLUMN     "modelSize" TEXT;

-- AlterTable
ALTER TABLE "public"."server_health" ADD COLUMN     "avgProcessingTime" DOUBLE PRECISION,
ADD COLUMN     "modelStates" JSONB,
ADD COLUMN     "requestCount" INTEGER,
ADD COLUMN     "systemResources" JSONB,
ADD COLUMN     "uptime" TEXT,
ADD COLUMN     "version" TEXT;

-- AlterTable
ALTER TABLE "public"."system_info" ADD COLUMN     "cpuUsage" DOUBLE PRECISION,
ADD COLUMN     "cudaVersion" TEXT,
ADD COLUMN     "gpuMemoryTotal" TEXT,
ADD COLUMN     "pythonVersion" TEXT,
ADD COLUMN     "requestId" TEXT,
ADD COLUMN     "serverVersion" TEXT,
ADD COLUMN     "systemMemoryTotal" TEXT,
ADD COLUMN     "torchVersion" TEXT;
