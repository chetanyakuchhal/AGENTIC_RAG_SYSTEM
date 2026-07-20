const state = {
  format: "json",
  lastDraft: "",
  lastQuery: "",
};

const elements = {
  systemStatus: document.querySelector("#systemStatus"),
  chunkCount: document.querySelector("#chunkCount"),
  geminiStatus: document.querySelector("#geminiStatus"),
  openaiStatus: document.querySelector("#openaiStatus"),
  groqStatus: document.querySelector("#groqStatus"),
  webStatus: document.querySelector("#webStatus"),
  sourceList: document.querySelector("#sourceList"),
  sourceChips: document.querySelector("#sourceChips"),
  fileInput: document.querySelector("#fileInput"),
  uploadButton: document.querySelector("#uploadButton"),
  ingestButton: document.querySelector("#ingestButton"),
  clearButton: document.querySelector("#clearButton"),
  refreshButton: document.querySelector("#refreshButton"),
  uploadMessage: document.querySelector("#uploadMessage"),
  questionForm: document.querySelector("#questionForm"),
  queryInput: document.querySelector("#queryInput"),
  modelProvider: document.querySelector("#modelProvider"),
  modelName: document.querySelector("#modelName"),
  temperatureInput: document.querySelector("#temperatureInput"),
  maxTokensInput: document.querySelector("#maxTokensInput"),
  topPInput: document.querySelector("#topPInput"),
  pptTemplateControl: document.querySelector("#pptTemplateControl"),
  pptTemplate: document.querySelector("#pptTemplate"),
  askButton: document.querySelector("#askButton"),
  answerTitle: document.querySelector("#answerTitle"),
  answerBox: document.querySelector("#answerBox"),
  pptWorkflow: document.querySelector("#pptWorkflow"),
  draftEditor: document.querySelector("#draftEditor"),
  refinePrompt: document.querySelector("#refinePrompt"),
  refineButton: document.querySelector("#refineButton"),
  generatePptButton: document.querySelector("#generatePptButton"),
  totalTiming: document.querySelector("#totalTiming"),
  timingGrid: document.querySelector("#timingGrid"),
  confidenceBadge: document.querySelector("#confidenceBadge"),
  formatButtons: document.querySelectorAll(".format-button"),
};

function setBusy(button, isBusy, label) {
  button.dataset.defaultLabel = button.dataset.defaultLabel || button.textContent;
  button.disabled = isBusy;
  button.textContent = isBusy ? label : button.dataset.defaultLabel;
}

function setMessage(message) {
  elements.uploadMessage.textContent = message;
}

function renderSources(sources) {
  elements.sourceList.innerHTML = "";
  elements.sourceChips.innerHTML = "";

  if (!sources || sources.length === 0) {
    elements.sourceList.innerHTML = "<li>No indexed sources yet.<span>Upload and index documents first.</span></li>";
    return;
  }

  for (const source of sources) {
    const item = document.createElement("li");
    const pageText = source.page_count ? `${source.page_count} page(s)` : "source metadata";
    item.innerHTML = `${source.source}<span>${source.chunks || 0} chunks • ${pageText}</span>`;
    elements.sourceList.appendChild(item);
  }
}

function renderAnswer(result) {
  elements.answerTitle.textContent = "Answer generated";
  elements.answerBox.textContent = result.answer || "No answer returned.";
  state.lastDraft = result.answer || "";
  state.lastQuery = result.query || state.lastQuery;

  const confidence = result.confidence || "unknown";
  elements.confidenceBadge.textContent = confidence;
  elements.confidenceBadge.className = `badge ${confidence}`;

  elements.sourceChips.innerHTML = "";
  for (const source of result.sources || []) {
    const chip = document.createElement("span");
    chip.className = "source-chip";
    chip.textContent = source.label || source.source || "Source";
    elements.sourceChips.appendChild(chip);
  }

  renderTimings(result.timings || {});
}

function showPptWorkflow(show) {
  elements.pptWorkflow.classList.toggle("hidden", !show);
}

function showPptTemplateControl(show) {
  elements.pptTemplateControl.classList.toggle("hidden", !show);
}

function renderTimings(timings) {
  const entries = Object.entries(timings || {});
  elements.timingGrid.innerHTML = "";

  const total = timings.request_total ?? timings.graph_total;
  elements.totalTiming.textContent = total ? `${total.toFixed(2)}s total` : "-";

  if (!entries.length) {
    const empty = document.createElement("div");
    empty.className = "timing-empty";
    empty.textContent = "Run a query to see per-step timings.";
    elements.timingGrid.appendChild(empty);
    return;
  }

  const labels = {
    retrieval: "Retrieval",
    router: "Router",
    web_search: "Web search",
    generator: "Generation",
    graph_total: "Graph total",
    format: "Export formatting",
    request_total: "Request total",
  };

  for (const [key, value] of entries) {
    const card = document.createElement("div");
    card.className = "timing-item";
    card.innerHTML = `<span>${labels[key] || key}</span><strong>${Number(value).toFixed(2)}s</strong>`;
    elements.timingGrid.appendChild(card);
  }
}

function timingsFromHeaders(response) {
  const raw = response.headers.get("X-Agent-Timings");
  if (!raw) {
    return {};
  }

  try {
    return JSON.parse(raw);
  } catch (error) {
    return {};
  }
}

function downloadBlob(blob, filename) {
  try {
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = filename;
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    link.remove();

    window.setTimeout(() => {
      URL.revokeObjectURL(objectUrl);
    }, 1000);
  } catch (error) {
    throw new Error("The report was generated, but the browser could not start the download. Please try again.");
  }
}

function currentModelSettings() {
  const provider = elements.modelProvider.value;
  const fallbackModel = provider === "openai"
    ? "gpt-4o-mini"
    : provider === "groq"
      ? "llama-3.1-8b-instant"
      : "gemini-2.5-flash";
  const maxTokenLimit = provider === "groq" ? 1800 : 8000;
  const requestedMaxTokens = Number(elements.maxTokensInput.value || 1400);
  const maxTokens = Math.min(requestedMaxTokens, maxTokenLimit);
  if (requestedMaxTokens !== maxTokens) {
    elements.maxTokensInput.value = String(maxTokens);
  }
  return {
    model_provider: provider,
    model_name: elements.modelName.value.trim() || fallbackModel,
    temperature: Number(elements.temperatureInput.value || 0.2),
    max_tokens: maxTokens,
    top_p: Number(elements.topPInput.value || 0.9),
  };
}

function applyProviderDefaults() {
  const provider = elements.modelProvider.value;
  if (provider === "openai") {
    elements.modelName.value = "gpt-4o-mini";
    elements.maxTokensInput.max = "8000";
  } else if (provider === "groq") {
    elements.modelName.value = "llama-3.1-8b-instant";
    elements.maxTokensInput.max = "1800";
    if (Number(elements.maxTokensInput.value) > 1800) {
      elements.maxTokensInput.value = "1800";
    }
  } else {
    elements.modelName.value = "gemini-2.5-flash";
    elements.maxTokensInput.max = "8000";
  }
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    elements.systemStatus.textContent = "Online";
    elements.chunkCount.textContent = status.indexed_chunks || 0;
    elements.geminiStatus.textContent = status.gemini_configured ? "Ready" : "Missing key";
    elements.openaiStatus.textContent = status.openai_configured ? "Ready" : "Missing key";
    elements.groqStatus.textContent = status.groq_configured ? "Ready" : "Missing key";
    elements.webStatus.textContent = status.web_search_configured ? "Ready" : "Optional";
    renderSources(status.sources || []);
  } catch (error) {
    elements.systemStatus.textContent = "Offline";
    setMessage("Could not reach the backend.");
  }
}

async function uploadFiles() {
  const files = elements.fileInput.files;
  if (!files.length) {
    setMessage("Choose at least one document first.");
    return;
  }

  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }

  setBusy(elements.uploadButton, true, "Uploading");
  try {
    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || "Upload failed.");
    }
    setMessage(`Uploaded ${result.files.length} file(s). Run Index to add them to memory.`);
    elements.fileInput.value = "";
    await refreshStatus();
  } catch (error) {
    setMessage(error.message);
  } finally {
    setBusy(elements.uploadButton, false, "Uploading");
  }
}

async function ingestDocuments() {
  setBusy(elements.ingestButton, true, "Indexing");
  setMessage("Indexing documents into the vector database...");
  try {
    const response = await fetch("/api/ingest", { method: "POST" });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || "Indexing failed.");
    }
    setMessage(`${result.message} ${result.chunks || 0} chunk(s) processed.`);
    await refreshStatus();
  } catch (error) {
    setMessage(error.message);
  } finally {
    setBusy(elements.ingestButton, false, "Indexing");
  }
}

async function clearIndexedMemory() {
  const confirmed = window.confirm(
    "Reset the knowledge base? This clears indexed memory and removes uploaded documents."
  );

  if (!confirmed) return;

  setBusy(elements.clearButton, true, "Clearing");

  try {
    const response = await fetch("/api/clear", {
      method: "POST",
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Clear failed.");
    }

    const deletedFiles = result.deleted_files?.length || 0;

    setMessage(
      `${result.message} ${result.deleted_chunks || 0} chunk(s) and ${deletedFiles} file(s) removed.`
    );

    elements.fileInput.value = "";
    elements.answerTitle.textContent = "Knowledge base reset";
    elements.answerBox.textContent =
      "The vector database and uploaded documents have been cleared.";

    elements.confidenceBadge.textContent = "cleared";
    elements.confidenceBadge.className = "badge";

    await refreshStatus();
  } catch (error) {
    console.error(error);
    setMessage(error.message);
  } finally {
    setBusy(elements.clearButton, false, "Clearing");
  }
}
async function askQuestion(event) {
  event.preventDefault();
  const query = elements.queryInput.value.trim();
  if (!query) {
    elements.answerTitle.textContent = "Question required";
    elements.answerBox.textContent = "Enter a question before running the agent.";
    return;
  }

  setBusy(elements.askButton, true, "Running");
  elements.answerTitle.textContent = "Agent is working";
  elements.answerBox.textContent = "Retrieving evidence and generating the final response...";
  if (state.format !== "pptx") {
    showPptWorkflow(false);
  }

  try {
    const requestFormat = state.format === "pptx" ? "json" : state.format;
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        output_format: requestFormat,
        presentation_mode: state.format === "pptx",
        ppt_template: elements.pptTemplate.value,
        ...currentModelSettings(),
      }),
    });

    if (requestFormat === "json") {
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || "Question failed.");
      }
      renderAnswer(result);
      if (state.format === "pptx") {
        elements.answerTitle.textContent = "Draft ready for PPT";
        elements.answerBox.textContent = "Review the generated draft below. Refine it if needed, then generate the PPT from the edited draft.";
        elements.draftEditor.value = result.answer || "";
        elements.refinePrompt.value = "";
        showPptWorkflow(true);
      }
      return;
    }

    if (!response.ok) {
      const result = await response.json();
      throw new Error(result.detail || "Report generation failed.");
    }

    const blob = await response.blob();
    const timings = timingsFromHeaders(response);
    const downloadNames = {
      pdf: "agentic_report.pdf",
      pptx: "agentic_report.pptx",
      xlsx: "agentic_report.xlsx",
    };
    downloadBlob(blob, downloadNames[state.format] || "agentic_report");

    elements.answerTitle.textContent = "Report downloaded";
    elements.answerBox.textContent = `Your ${state.format.toUpperCase()} report has been generated from the latest agent response.`;
    elements.confidenceBadge.textContent = "export";
    elements.confidenceBadge.className = "badge high";
    renderTimings(timings);
    await refreshStatus();
  } catch (error) {
    elements.answerTitle.textContent = "Something needs attention";
    elements.answerBox.textContent = error.message;
    elements.confidenceBadge.textContent = "error";
    elements.confidenceBadge.className = "badge medium";
  } finally {
    setBusy(elements.askButton, false, "Running");
  }
}

async function refineDraft() {
  const current_answer = elements.draftEditor.value.trim();
  const refinement_prompt = elements.refinePrompt.value.trim();
  const query = elements.queryInput.value.trim() || state.lastQuery;

  if (!query || !current_answer) {
    setMessage("Generate a draft first before refining it.");
    return;
  }

  if (!refinement_prompt) {
    setMessage("Add a refinement prompt so the model knows what to improve.");
    return;
  }

  setBusy(elements.refineButton, true, "Refining");
  elements.answerTitle.textContent = "Refining draft";
  elements.answerBox.textContent = "Improving the draft for PPT quality...";

  try {
    const response = await fetch("/api/refine", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        current_answer,
        refinement_prompt,
        ...currentModelSettings(),
      }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || "Refinement failed.");
    }
    renderAnswer(result);
    elements.answerTitle.textContent = "Draft refined";
    elements.answerBox.textContent = "The draft was refined. Review it below, refine again if needed, or generate the PPT.";
    elements.draftEditor.value = result.answer || "";
    showPptWorkflow(true);
  } catch (error) {
    elements.answerTitle.textContent = "Something needs attention";
    elements.answerBox.textContent = error.message;
  } finally {
    setBusy(elements.refineButton, false, "Refining");
  }
}

async function generatePptFromDraft() {
  const query = elements.queryInput.value.trim() || state.lastQuery;
  const custom_answer = elements.draftEditor.value.trim();

  if (!query || !custom_answer) {
    setMessage("Generate a draft first before exporting the PPT.");
    return;
  }

  setBusy(elements.generatePptButton, true, "Generating PPT");
  elements.answerTitle.textContent = "Generating PPT";
  elements.answerBox.textContent = "Building the PowerPoint from your reviewed draft...";

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        output_format: "pptx",
        custom_answer,
        ppt_template: elements.pptTemplate.value,
        ...currentModelSettings(),
      }),
    });

    if (!response.ok) {
      const result = await response.json();
      throw new Error(result.detail || "PPT generation failed.");
    }

    const blob = await response.blob();
    const timings = timingsFromHeaders(response);
    downloadBlob(blob, "agentic_report.pptx");

    elements.answerTitle.textContent = "PPT downloaded";
    elements.answerBox.textContent = "The PowerPoint was generated from your reviewed draft and downloaded successfully.";
    elements.confidenceBadge.textContent = "export";
    elements.confidenceBadge.className = "badge high";
    renderTimings(timings);
    await refreshStatus();
  } catch (error) {
    elements.answerTitle.textContent = "Something needs attention";
    elements.answerBox.textContent = error.message;
  } finally {
    setBusy(elements.generatePptButton, false, "Generating PPT");
  }
}

elements.formatButtons.forEach((button) => {
  button.addEventListener("click", () => {
    state.format = button.dataset.format;
    elements.formatButtons.forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    showPptTemplateControl(state.format === "pptx");
  });
});

elements.modelProvider.addEventListener("change", () => {
  applyProviderDefaults();
});

elements.uploadButton.addEventListener("click", uploadFiles);
elements.ingestButton.addEventListener("click", ingestDocuments);
elements.clearButton.addEventListener("click", clearIndexedMemory);
elements.refreshButton.addEventListener("click", refreshStatus);
elements.refineButton.addEventListener("click", refineDraft);
elements.generatePptButton.addEventListener("click", generatePptFromDraft);
elements.questionForm.addEventListener("submit", askQuestion);

applyProviderDefaults();
showPptTemplateControl(state.format === "pptx");
refreshStatus();
renderTimings({});
