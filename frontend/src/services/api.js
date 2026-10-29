import axios from "axios";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
});

export const checkHealth = async () => {
  const response = await apiClient.get("/health");
  return response.data;
};

export const getSupportedClasses = async () => {
  const response = await apiClient.get("/classes");
  return response.data;
};

export const predictPotatoDisease = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await apiClient.post("/predict", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
};

export default apiClient;
