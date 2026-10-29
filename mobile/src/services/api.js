import axios from "axios";
import { Platform } from "react-native";

// In Android Emulator, host machine is 10.0.2.2; on iOS simulator it is localhost
const DEFAULT_HOST = Platform.OS === "android" ? "http://10.0.2.2:8000" : "http://localhost:8000";

// Replace with your computer's Wi-Fi IP (e.g., http://192.168.1.15:8000) when running on a physical phone
export const API_BASE_URL = DEFAULT_HOST;

export const predictLeafFromMobile = async (imageUri) => {
  const filename = imageUri.split("/").pop() || "leaf_specimen.jpg";
  const match = /\.(\w+)$/.exec(filename);
  const type = match ? `image/${match[1].toLowerCase()}` : "image/jpeg";

  const formData = new FormData();
  formData.append("file", {
    uri: imageUri,
    name: filename,
    type: type,
  });

  const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
    timeout: 15000,
  });

  return response.data;
};
