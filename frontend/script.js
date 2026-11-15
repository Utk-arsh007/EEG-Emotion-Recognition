/**
 * Frontend Interactive Controller for EEG Emotion Recognition System
 * Handles sample selection, file upload, inference API calls,
 * metric gauges, and 2D Russell's Circumplex canvas rendering.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const btnModeSample = document.getElementById("btn-mode-sample");
  const btnModeUpload = document.getElementById("btn-mode-upload");
  const sampleSection = document.getElementById("sample-input-section");
  const uploadSection = document.getElementById("upload-input-section");
  const sampleSelect = document.getElementById("sample-select");
  const trialIndexInput = document.getElementById("trial-index");
  
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const fileInfo = document.getElementById("file-info");
  const selectedFilename = document.getElementById("selected-filename");
  const btnRemoveFile = document.getElementById("btn-remove-file");
  
  const btnPredict = document.getElementById("btn-predict");
  const spinner = document.getElementById("spinner");
  const errorMessage = document.getElementById("error-message");

  // Results elements
  const badgeQuadrant = document.getElementById("badge-quadrant");
  const circumplexBadge = document.getElementById("circumplex-badge");
  const stateTitle = document.getElementById("state-title");
  const stateMeta = document.getElementById("state-meta");

  const valValence = document.getElementById("val-valence");
  const fillValence = document.getElementById("fill-valence");
  const valArousal = document.getElementById("val-arousal");
  const fillArousal = document.getElementById("fill-arousal");
  const valDominance = document.getElementById("val-dominance");
  const fillDominance = document.getElementById("fill-dominance");
  const valLiking = document.getElementById("val-liking");
  const fillLiking = document.getElementById("fill-liking");

  const canvas = document.getElementById("circumplex-canvas");
  const ctx = canvas.getContext("2d");

  let currentMode = "sample"; // 'sample' or 'upload'
  let uploadedFile = null;
  let currentValence = 5.0;
  let currentArousal = 5.0;

  // -----------------------------------------------------------------------
  // Initial Setup & Canvas Drawing
  // -----------------------------------------------------------------------
  drawCircumplex(currentValence, currentArousal, false);
  loadSampleList();

  // -----------------------------------------------------------------------
  // Mode Switching
  // -----------------------------------------------------------------------
  btnModeSample.addEventListener("click", () => {
    currentMode = "sample";
    btnModeSample.classList.add("active");
    btnModeUpload.classList.remove("active");
    sampleSection.classList.remove("hidden");
    uploadSection.classList.add("hidden");
    hideError();
  });

  btnModeUpload.addEventListener("click", () => {
    currentMode = "upload";
    btnModeUpload.classList.add("active");
    btnModeSample.classList.remove("active");
    uploadSection.classList.remove("hidden");
    sampleSection.classList.add("hidden");
    hideError();
  });

  // -----------------------------------------------------------------------
  // File Upload & Drag-and-Drop
  // -----------------------------------------------------------------------
  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  btnRemoveFile.addEventListener("click", () => {
    uploadedFile = null;
    fileInput.value = "";
    fileInfo.classList.add("hidden");
    dropZone.classList.remove("hidden");
  });

  function handleSelectedFile(file) {
    const ext = file.name.split(".").pop().toLowerCase();
    if (ext !== "dat" && ext !== "npy") {
      showError("Please upload a valid .dat or .npy EEG file conforming to DEAP specifications.");
      return;
    }
    uploadedFile = file;
    selectedFilename.textContent = file.name;
    fileInfo.classList.remove("hidden");
    dropZone.classList.add("hidden");
    hideError();
  }

  // -----------------------------------------------------------------------
  // Load Available Samples from Backend
  // -----------------------------------------------------------------------
  async function loadSampleList() {
    try {
      const res = await fetch("/api/samples");
      if (res.ok) {
        const data = await res.json();
        if (data.samples && data.samples.length > 0) {
          sampleSelect.innerHTML = "";
          data.samples.forEach((s) => {
            const opt = document.createElement("option");
            opt.value = s.filename;
            opt.textContent = `${s.filename} (${s.participant_id})`;
            sampleSelect.appendChild(opt);
          });
        }
      }
    } catch (err) {
      console.warn("Could not fetch samples dynamically. Using defaults.", err);
    }
  }

  // -----------------------------------------------------------------------
  // Inference Action
  // -----------------------------------------------------------------------
  btnPredict.addEventListener("click", async () => {
    hideError();
    setLoading(true);

    try {
      let data;
      if (currentMode === "sample") {
        const payload = {
          sample_filename: sampleSelect.value,
          trial_index: parseInt(trialIndexInput.value, 10) || 0
        };
        const res = await fetch("/api/predict/sample", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || "Sample inference failed");
        }
        data = await res.json();
      } else {
        if (!uploadedFile) {
          throw new Error("Please select or drop an EEG file (.dat or .npy) first.");
        }
        const formData = new FormData();
        formData.append("file", uploadedFile);
        const res = await fetch("/api/predict/upload", {
          method: "POST",
          body: formData
        });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || "Uploaded file processing failed");
        }
        data = await res.json();
      }

      displayResults(data);
    } catch (err) {
      showError(err.message);
    } finally {
      setLoading(false);
    }
  });

  // -----------------------------------------------------------------------
  // Render Results on UI
  // -----------------------------------------------------------------------
  function displayResults(data) {
    const p = data.predictions;
    const c = data.circumplex;

    // Numerical Ratings
    valValence.textContent = p.Valence.toFixed(2);
    fillValence.style.width = `${((p.Valence - 1.0) / 8.0) * 100}%`;

    valArousal.textContent = p.Arousal.toFixed(2);
    fillArousal.style.width = `${((p.Arousal - 1.0) / 8.0) * 100}%`;

    valDominance.textContent = p.Dominance.toFixed(2);
    fillDominance.style.width = `${((p.Dominance - 1.0) / 8.0) * 100}%`;

    valLiking.textContent = p.Liking.toFixed(2);
    fillLiking.style.width = `${((p.Liking - 1.0) / 8.0) * 100}%`;

    // Circumplex State & Badge
    badgeQuadrant.textContent = c.quadrant.split(" ")[0]; // e.g. HVHA
    badgeQuadrant.style.color = c.theme_color;
    circumplexBadge.style.borderColor = c.theme_color;

    stateTitle.textContent = c.emotional_state;
    stateTitle.style.color = c.theme_color;
    stateMeta.textContent = `Analyzed ${data.windows_analyzed} temporal segments (2s window) | In-band 4–45 Hz`;

    // Redraw 2D Circumplex Plot
    currentValence = p.Valence;
    currentArousal = p.Arousal;
    drawCircumplex(currentValence, currentArousal, true, c.theme_color);
  }

  // -----------------------------------------------------------------------
  // 2D Russell's Circumplex Canvas Renderer
  // -----------------------------------------------------------------------
  function drawCircumplex(valence, arousal, hasActivePoint = false, pointColor = "#3b82f6") {
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padX = 60;
    const padY = 40;
    const plotW = w - 2 * padX;
    const plotH = h - 2 * padY;
    const midX = padX + plotW / 2;
    const midY = padY + plotH / 2;

    // Background Quadrants
    // HVHA: Top-Right
    ctx.fillStyle = "rgba(16, 185, 129, 0.06)";
    ctx.fillRect(midX, padY, plotW / 2, plotH / 2);

    // LVHA: Top-Left
    ctx.fillStyle = "rgba(239, 68, 68, 0.06)";
    ctx.fillRect(padX, padY, plotW / 2, plotH / 2);

    // LVLA: Bottom-Left
    ctx.fillStyle = "rgba(71, 85, 105, 0.08)";
    ctx.fillRect(padX, midY, plotW / 2, plotH / 2);

    // HVLA: Bottom-Right
    ctx.fillStyle = "rgba(59, 130, 246, 0.06)";
    ctx.fillRect(midX, midY, plotW / 2, plotH / 2);

    // Quadrant Labels
    ctx.font = "bold 11px Inter, sans-serif";
    ctx.fillStyle = "#10b981";
    ctx.fillText("HVHA: Joy / Excitement / Alert", midX + 20, padY + 24);

    ctx.fillStyle = "#ef4444";
    ctx.fillText("LVHA: Anger / Anxiety / Distress", padX + 16, padY + 24);

    ctx.fillStyle = "#94a3b8";
    ctx.fillText("LVLA: Sadness / Depression / Boredom", padX + 16, h - padY - 14);

    ctx.fillStyle = "#60a5fa";
    ctx.fillText("HVLA: Calmness / Relaxation / Content", midX + 20, h - padY - 14);

    // Grid Axes
    ctx.strokeStyle = "#475569";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);

    // Horizontal Neutral Valence Line (Arousal = 5.0)
    ctx.beginPath();
    ctx.moveTo(padX, midY);
    ctx.lineTo(w - padX, midY);
    ctx.stroke();

    // Vertical Neutral Arousal Line (Valence = 5.0)
    ctx.beginPath();
    ctx.moveTo(midX, padY);
    ctx.lineTo(midX, h - padY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Outer Border
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1;
    ctx.strokeRect(padX, padY, plotW, plotH);

    // Axis End Labels
    ctx.font = "10px Inter, sans-serif";
    ctx.fillStyle = "#64748b";
    ctx.fillText("← Unpleasant (1.0)", padX, midY - 6);
    ctx.fillText("Pleasant (9.0) →", w - padX - 85, midY - 6);
    ctx.fillText("High Arousal (9.0) ↑", midX + 8, padY + 12);
    ctx.fillText("↓ Low Arousal (1.0)", midX + 8, h - padY - 6);

    // Plot Point
    if (hasActivePoint) {
      const ptX = padX + ((valence - 1.0) / 8.0) * plotW;
      const ptY = (padY + plotH) - ((arousal - 1.0) / 8.0) * plotH;

      // Outer Pulse Ring
      ctx.beginPath();
      ctx.arc(ptX, ptY, 14, 0, 2 * Math.PI);
      ctx.fillStyle = pointColor.replace(")", ", 0.2)").replace("rgb", "rgba");
      ctx.fill();

      // Solid Dot
      ctx.beginPath();
      ctx.arc(ptX, ptY, 6, 0, 2 * Math.PI);
      ctx.fillStyle = pointColor;
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Coordinate Tag
      ctx.font = "bold 11px 'JetBrains Mono', monospace";
      ctx.fillStyle = "#fff";
      ctx.fillText(`(V:${valence.toFixed(2)}, A:${arousal.toFixed(2)})`, ptX + 10, ptY - 8);
    }
  }

  // -----------------------------------------------------------------------
  // Helper UI Utilities
  // -----------------------------------------------------------------------
  function setLoading(isLoading) {
    if (isLoading) {
      btnPredict.disabled = true;
      spinner.classList.remove("hidden");
    } else {
      btnPredict.disabled = false;
      spinner.classList.add("hidden");
    }
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    errorMessage.classList.remove("hidden");
  }

  function hideError() {
    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
  }
});

