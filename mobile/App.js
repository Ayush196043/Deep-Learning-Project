import React, { useState } from "react";
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Image,
  ScrollView,
  ActivityIndicator,
  Alert,
  SafeAreaView,
  StatusBar,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { predictLeafFromMobile } from "./src/services/api";

export default function App() {
  const [imageUri, setImageUri] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  // 1. Pick from Photo Gallery
  const pickImageFromGallery = async () => {
    setErrorMsg(null);
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== "granted") {
      Alert.alert(
        "Permission Denied",
        "Gallery access permission is required to select leaf photographs."
      );
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.9,
    });

    if (!result.canceled && result.assets && result.assets.length > 0) {
      setImageUri(result.assets[0].uri);
      setPrediction(null);
    }
  };

  // 2. Capture Directly with Camera in Field
  const captureWithCamera = async () => {
    setErrorMsg(null);
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== "granted") {
      Alert.alert(
        "Permission Denied",
        "Camera access permission is required to photograph leaves in the field."
      );
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.9,
    });

    if (!result.canceled && result.assets && result.assets.length > 0) {
      setImageUri(result.assets[0].uri);
      setPrediction(null);
    }
  };

  // 3. Trigger FastAPI Prediction
  const handleAnalyze = async () => {
    if (!imageUri) return;
    setIsLoading(true);
    setErrorMsg(null);

    try {
      const data = await predictLeafFromMobile(imageUri);
      setPrediction(data);
    } catch (err) {
      console.error(err);
      setErrorMsg(
        err.response?.data?.detail ||
        "Failed to reach FastAPI server. Check your connection to http://127.0.0.1:8000."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setImageUri(null);
    setPrediction(null);
    setErrorMsg(null);
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" backgroundColor="#1b5e20" />
      <View style={styles.header}>
        <Text style={styles.headerIcon}>🥔</Text>
        <View>
          <Text style={styles.headerTitle}>Potato Leaf Doctor</Text>
          <Text style={styles.headerSubtitle}>Field Pathology AI Scanner</Text>
        </View>
      </View>

      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Specimen View */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Leaf Specimen</Text>

          {imageUri ? (
            <View style={styles.previewBox}>
              <Image source={{ uri: imageUri }} style={styles.imagePreview} />
              <TouchableOpacity
                style={styles.btnSecondary}
                onPress={handleReset}
                disabled={isLoading}
              >
                <Text style={styles.btnSecondaryText}>🔄 Retake / Choose Other</Text>
              </TouchableOpacity>
            </View>
          ) : (
            <View style={styles.pickerActions}>
              <TouchableOpacity
                style={[styles.btnAction, styles.btnCamera]}
                onPress={captureWithCamera}
              >
                <Text style={styles.btnActionText}>📸 Snap Photo in Field</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.btnAction, styles.btnGallery]}
                onPress={pickImageFromGallery}
              >
                <Text style={styles.btnActionText}>🖼️ Pick from Gallery</Text>
              </TouchableOpacity>
            </View>
          )}

          {imageUri && !prediction && (
            <TouchableOpacity
              style={[styles.btnPrimary, isLoading && styles.btnDisabled]}
              onPress={handleAnalyze}
              disabled={isLoading}
            >
              {isLoading ? (
                <View style={styles.spinnerContainer}>
                  <ActivityIndicator color="#ffffff" />
                  <Text style={styles.btnPrimaryText}> Analyzing Specimen...</Text>
                </View>
              ) : (
                <Text style={styles.btnPrimaryText}>🔬 Run Disease Diagnosis</Text>
              )}
            </TouchableOpacity>
          )}

          {errorMsg && (
            <View style={styles.errorBox}>
              <Text style={styles.errorText}>⚠️ {errorMsg}</Text>
            </View>
          )}
        </View>

        {/* Diagnostic Results */}
        {prediction && (
          <View style={styles.card}>
            <View style={styles.badgeRow}>
              <Text
                style={[
                  styles.statusBadge,
                  prediction.is_healthy
                    ? styles.badgeGreen
                    : prediction.predicted_class.includes("Early")
                    ? styles.badgeOrange
                    : styles.badgeRed,
                ]}
              >
                {prediction.is_healthy
                  ? "✅ Healthy"
                  : prediction.predicted_class.includes("Early")
                  ? "⚠️ Early Blight"
                  : "🚨 Late Blight"}
              </Text>
              <Text style={styles.confidenceBadge}>
                {prediction.confidence_percentage} Certainty
              </Text>
            </View>

            <Text style={styles.diagnosisTitle}>
              {prediction.class_name_clean}
            </Text>

            {/* Educational Advisory */}
            <View style={styles.advisoryBlock}>
              <Text style={styles.advisoryHeader}>🔍 Clinical Symptoms:</Text>
              <Text style={styles.advisoryBody}>
                {prediction.educational_info?.symptoms}
              </Text>

              <Text style={[styles.advisoryHeader, { marginTop: 14 }]}>
                🧪 Farmer Treatment Actions:
              </Text>
              <Text style={styles.advisoryBody}>
                {prediction.educational_info?.treatment_advice}
              </Text>
            </View>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#f4f7f4",
  },
  header: {
    backgroundColor: "#1b5e20",
    paddingVertical: 18,
    paddingHorizontal: 20,
    flexDirection: "row",
    alignItems: "center",
  },
  headerIcon: {
    fontSize: 32,
    marginRight: 12,
  },
  headerTitle: {
    color: "#ffffff",
    fontSize: 20,
    fontWeight: "800",
  },
  headerSubtitle: {
    color: "#a7f3d0",
    fontSize: 12,
    fontWeight: "600",
  },
  scrollContent: {
    padding: 16,
  },
  card: {
    backgroundColor: "#ffffff",
    borderRadius: 14,
    padding: 18,
    marginBottom: 16,
    elevation: 3,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 6,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: "#1e293b",
    marginBottom: 14,
  },
  pickerActions: {
    gap: 12,
  },
  btnAction: {
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
  },
  btnCamera: {
    backgroundColor: "#2e7d32",
  },
  btnGallery: {
    backgroundColor: "#334155",
  },
  btnActionText: {
    color: "#ffffff",
    fontSize: 15,
    fontWeight: "700",
  },
  previewBox: {
    alignItems: "center",
  },
  imagePreview: {
    width: "100%",
    height: 240,
    borderRadius: 10,
    marginBottom: 12,
  },
  btnPrimary: {
    backgroundColor: "#1b5e20",
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
    marginTop: 12,
  },
  btnPrimaryText: {
    color: "#ffffff",
    fontSize: 16,
    fontWeight: "700",
  },
  btnSecondary: {
    paddingVertical: 8,
  },
  btnSecondaryText: {
    color: "#475569",
    fontSize: 14,
    fontWeight: "600",
  },
  btnDisabled: {
    opacity: 0.6,
  },
  spinnerContainer: {
    flexDirection: "row",
    alignItems: "center",
  },
  errorBox: {
    backgroundColor: "#fef2f2",
    padding: 12,
    borderRadius: 8,
    marginTop: 12,
  },
  errorText: {
    color: "#b91c1c",
    fontSize: 13,
  },
  badgeRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12,
  },
  statusBadge: {
    paddingVertical: 4,
    paddingHorizontal: 12,
    borderRadius: 9999,
    fontSize: 12,
    fontWeight: "800",
  },
  badgeGreen: {
    backgroundColor: "#dcfce7",
    color: "#15803d",
  },
  badgeOrange: {
    backgroundColor: "#fef3c7",
    color: "#b45309",
  },
  badgeRed: {
    backgroundColor: "#fee2e2",
    color: "#b91c1c",
  },
  confidenceBadge: {
    fontSize: 13,
    fontWeight: "700",
    color: "#2e7d32",
  },
  diagnosisTitle: {
    fontSize: 22,
    fontWeight: "800",
    color: "#0f172a",
    marginBottom: 14,
  },
  advisoryBlock: {
    backgroundColor: "#f8fafc",
    padding: 14,
    borderRadius: 10,
    borderLeftWidth: 4,
    borderLeftColor: "#2e7d32",
  },
  advisoryHeader: {
    fontSize: 13,
    fontWeight: "700",
    color: "#1e293b",
    marginBottom: 4,
  },
  advisoryBody: {
    fontSize: 13,
    color: "#475569",
    lineHeight: 18,
  },
});
