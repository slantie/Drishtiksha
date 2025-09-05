-- CreateIndex
CREATE INDEX "deepfake_analyses_videoId_model_analysisType_idx" ON "public"."deepfake_analyses"("videoId", "model", "analysisType");
