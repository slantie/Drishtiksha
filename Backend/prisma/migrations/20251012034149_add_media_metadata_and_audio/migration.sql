-- AlterTable
ALTER TABLE "public"."media" ADD COLUMN     "has_audio" BOOLEAN,
ADD COLUMN     "metadata" JSONB;
