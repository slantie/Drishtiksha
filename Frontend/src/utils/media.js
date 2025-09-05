// src/utils/media.js

export const getMediaType = (mimetype) => {
  if (typeof mimetype !== "string") {
    return "UNKNOWN";
  }
  if (mimetype.startsWith("video/")) return "VIDEO";
  if (mimetype.startsWith("image/")) return "IMAGE";
  if (mimetype.startsWith("audio/")) return "AUDIO";
  return "UNKNOWN";
};
