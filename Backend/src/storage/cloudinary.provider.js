// src/storage/cloudinary.provider.js

import {
  uploadOnCloudinary,
  uploadStreamToCloudinary,
  deleteFromCloudinary,
} from "../utils/cloudinary.js";

const cloudinaryProvider = {
  async uploadFile(localFilePath, subfolder) {
    const options = {
      resource_type: "auto",
      folder: `drishtiksha/${subfolder || "uploads"}`,
    };
    const response = await uploadOnCloudinary(localFilePath, options);
    return { url: response.secure_url, publicId: response.public_id };
  },
  async uploadStream(stream, options) {
    const response = await uploadStreamToCloudinary(stream, options);
    return { url: response.secure_url, publicId: response.public_id };
  },
  async deleteFile(publicId, resourceType = "video") {
    await deleteFromCloudinary(publicId, resourceType);
  },
};

export default cloudinaryProvider;
