import FONTS from "./fonts.js";

const SPECS_DIR = "../datasets/canva_specs";
const CANVA_DIR = "../datasets/canva";

let currentSpec = null;
let currentSpecName = null;
let hasChanges = false;

const API_URL = "http://localhost:5001";

// Simple toast notification
function showToast(message, isError = false) {
  const toast = document.createElement("div");
  toast.className = `toast ${isError ? "toast-error" : "toast-success"}`;
  toast.textContent = message;
  document.body.appendChild(toast);

  // Trigger animation
  setTimeout(() => toast.classList.add("show"), 10);

  // Remove after 2 seconds
  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.remove(), 300);
  }, 2000);
}

// Google Fonts mapping
const GOOGLE_FONTS = {
  Anton: "Anton",
  "Dancing Script": "Dancing Script",
  "Great Vibes": "Great Vibes",
  Montserrat: "Montserrat",
  Poppins: "Poppins",
};

async function loadSpecList() {
  try {
    const response = await fetch(SPECS_DIR);
    const text = await response.text();

    // Parse directory listing (this assumes a simple file server)
    // If you're using a different setup, you might need to provide a manifest
    const parser = new DOMParser();
    const doc = parser.parseFromString(text, "text/html");
    const links = Array.from(doc.querySelectorAll("a"))
      .map((a) => a.getAttribute("href"))
      .filter((href) => href && href !== "../" && !href.includes("."));

    const specList = document.getElementById("spec-list");

    links.forEach((dir, index) => {
      const specName = dir.replace("/", "");
      const item = document.createElement("div");
      item.className = "spec-item";
      item.dataset.specName = specName;

      const textSpan = document.createElement("span");
      textSpan.className = "spec-item-text";
      textSpan.textContent = `${index + 1}. ${specName}`;
      textSpan.onclick = () => loadSpec(specName);

      const copyBtn = document.createElement("button");
      copyBtn.className = "spec-copy-btn";
      copyBtn.innerHTML = "📋";
      copyBtn.title = "Copy spec ID";
      copyBtn.onclick = (e) => {
        e.stopPropagation();
        navigator.clipboard.writeText(specName).then(() => {
          copyBtn.innerHTML = "✓";
          setTimeout(() => {
            copyBtn.innerHTML = "📋";
          }, 1000);
        });
      };

      item.appendChild(textSpan);
      item.appendChild(copyBtn);
      specList.appendChild(item);
    });
  } catch (error) {
    document.getElementById("error").textContent =
      "Error loading spec list. Please ensure you're running a local server.";
    console.error("Error loading spec list:", error);
  }
}

async function loadSpec(specName) {
  try {
    // Update active state
    document.querySelectorAll(".spec-item").forEach((item) => {
      item.classList.toggle("active", item.dataset.specName === specName);
    });

    // Load spec JSON
    const specResponse = await fetch(`${SPECS_DIR}/${specName}/spec.json`);
    const spec = await specResponse.json();

    // Store current spec
    currentSpec = spec;
    currentSpecName = specName;
    hasChanges = false;
    document.getElementById("save-btn").disabled = true;
    document.getElementById("copy-btn").disabled = false;

    // Update URL without reloading page
    const newUrl = `${window.location.pathname}?design=${encodeURIComponent(specName)}`;
    history.pushState({ specName }, "", newUrl);

    // Render the spec
    renderSpec(spec, specName);

    document.getElementById("error").textContent = "";
  } catch (error) {
    document.getElementById(
      "error"
    ).textContent = `Error loading spec: ${error.message}`;
    console.error("Error loading spec:", error);
  }
}

function loadOriginalImage(specName) {
  const originalImage = document.getElementById("original-image");
  const imagePath = `${CANVA_DIR}/${specName}.webp`;

  originalImage.innerHTML = `<img src="${imagePath}" alt="${specName}" onerror="this.parentElement.innerHTML='<div style=\\'padding: 20px; color: #999;\\'>Original image not found</div>'" />`;
}

function renderSpec(spec, specName) {
  const canvasWidth = spec.canvas_width || 800;
  const canvasHeight = spec.canvas_height || 600;

  // Load the original image for comparison
  loadOriginalImage(specName);

  // Build nodes HTML
  let nodesHtml = "";

  for (let i = 0; i < (spec.nodes || []).length; i++) {
    const node = spec.nodes[i];

    if (node.type === "text") {
      const style = `
                position: absolute;
                left: ${node.x}px;
                top: ${node.y}px;
                width: ${node.width}px;
                height: ${node.height}px;
                transform: rotate(${node.rotation}deg);
                font-family: ${node["font-family"] || node.font_family};
                font-size: ${node["font-size"] || node.font_size}px;
                color: ${node.color};
                text-align: ${node["text-align"] || node.text_align};
                font-weight: ${node["font-weight"] || node.font_weight};
                font-style: ${node["font-style"] || node.font_style};
                text-decoration: ${
                  node["text-decoration"] || node.text_decoration
                };
                text-transform: ${
                  node["text-transform"] || node.text_transform
                };
                line-height: ${node["line-height"] || node.line_height || 1.2};
                opacity: ${node.opacity !== undefined ? node.opacity : 1};
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                display: flex;
                align-items: center;
                justify-content: ${
                  node["text-align"] === "right"
                    ? "flex-end"
                    : node["text-align"] === "center"
                    ? "center"
                    : "flex-start"
                };
            `;
      const textContent = escapeHtml(node.text);
      nodesHtml += `
                <div class="draggable" data-node-idx="${i}" style="${style}">
                    <div style="width: 100%;">${textContent}</div>
                    <div class="resize-handle nw" data-corner="nw"></div>
                    <div class="resize-handle ne" data-corner="ne"></div>
                    <div class="resize-handle sw" data-corner="sw"></div>
                    <div class="resize-handle se" data-corner="se"></div>
                </div>
            `;
    } else if (node.type === "image") {
      // Use explicit filename from spec, fallback to placeholder
      const assetPath = node.filename
        ? `${SPECS_DIR}/${specName}/${node.filename}`
        : "";

      const style = `
                position: absolute;
                left: ${node.x}px;
                top: ${node.y}px;
                width: ${node.width}px;
                height: ${node.height}px;
                transform: rotate(${node.rotation}deg);
                opacity: ${node.opacity !== undefined ? node.opacity : 1};
                object-fit: contain;
            `;

      // Try to load the asset, fallback to placeholder
      nodesHtml += `
                <div class="draggable" data-node-idx="${i}" style="${style}">
                    <img
                        src="${assetPath}"
                        style="width: 100%; height: 100%; object-fit: fill;"
                        alt="${node.asset_description}"
                        onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';"
                    />
                    <div style="width: 100%; height: 100%; background: #ddd; display: none; align-items: center; justify-content: center; font-size: 12px; color: #666; text-align: center; padding: 10px;">
                        [Image: ${
                          node.asset_description?.substring(0, 100) || ""
                        }]
                    </div>
                    <div class="resize-handle nw" data-corner="nw"></div>
                    <div class="resize-handle ne" data-corner="ne"></div>
                    <div class="resize-handle sw" data-corner="sw"></div>
                    <div class="resize-handle se" data-corner="se"></div>
                </div>
            `;
    } else if (node.type === "svg") {
      // Use explicit filename from spec, fallback to placeholder
      const svgPath = node.filename
        ? `${SPECS_DIR}/${specName}/${node.filename}`
        : "";

      const style = `
                position: absolute;
                left: ${node.x}px;
                top: ${node.y}px;
                width: ${node.width}px;
                height: ${node.height}px;
                transform: rotate(${node.rotation}deg);
                opacity: ${node.opacity !== undefined ? node.opacity : 1};
            `;

      nodesHtml += `
                <div class="draggable svg-container" data-node-idx="${i}" data-svg-path="${svgPath}" style="${style}">
                    <div class="svg-content" style="width: 100%; height: 100%;"></div>
                    <div class="svg-placeholder" style="width: 100%; height: 100%; background: #e8f5e9; border: 2px dashed #4caf50; display: none; align-items: center; justify-content: center; font-size: 12px; color: #2e7d32; text-align: center; padding: 10px; box-sizing: border-box;">
                        [SVG: ${
                          node.svg_description?.substring(0, 100) || ""
                        }]
                    </div>
                    <div class="resize-handle nw" data-corner="nw"></div>
                    <div class="resize-handle ne" data-corner="ne"></div>
                    <div class="resize-handle sw" data-corner="sw"></div>
                    <div class="resize-handle se" data-corner="se"></div>
                </div>
            `;
    }
  }

  // Build background style
  let bgStyle = `background-color: ${spec.background_color};`;
  if (spec.has_background_image) {
    const bgPath = `${SPECS_DIR}/${specName}/background.png`;
    bgStyle = `background-image: url('${bgPath}'); background-size: cover; background-position: center;`;
  }

  // Generate HTML
  const html = `
        <div id="canvas" style="
            position: relative;
            width: ${canvasWidth}px;
            height: ${canvasHeight}px;
            ${bgStyle}
            overflow: hidden;
        ">
            ${nodesHtml}
        </div>
    `;

  document.getElementById("canvas-container").innerHTML = html;

  // Load SVG files for SVG nodes
  loadSVGs();

  // Add drag-and-drop functionality
  setupDraggable();

  // Setup image drop zone
  setupImageDropZone();
}

async function loadSVGs() {
  const svgContainers = document.querySelectorAll(".svg-container");

  for (const container of svgContainers) {
    const svgPath = container.getAttribute("data-svg-path");
    const svgContent = container.querySelector(".svg-content");
    const svgPlaceholder = container.querySelector(".svg-placeholder");

    try {
      const response = await fetch(svgPath);
      if (response.ok) {
        const svgText = await response.text();
        // Inject SVG with proper sizing
        let svgMarkup = svgText;
        if (
          svgMarkup.includes("<svg") &&
          !svgMarkup.includes("width=") &&
          !svgMarkup.includes("height=")
        ) {
          svgMarkup = svgMarkup.replace(
            "<svg",
            '<svg width="100%" height="100%"'
          );
        }
        svgContent.innerHTML = svgMarkup;
        svgContent.style.display = "block";
        svgPlaceholder.style.display = "none";
      } else {
        // Show placeholder if SVG not found
        svgContent.style.display = "none";
        svgPlaceholder.style.display = "flex";
      }
    } catch (error) {
      // Show placeholder on error
      svgContent.style.display = "none";
      svgPlaceholder.style.display = "flex";
    }
  }
}

function setupDraggable() {
  const draggables = document.querySelectorAll(".draggable");
  let draggedElement = null;
  let resizingElement = null;
  let resizeCorner = null;
  let startX, startY, offsetX, offsetY;
  let startWidth, startHeight, startLeft, startTop;

  draggables.forEach((el) => {
    // Click to select
    el.addEventListener("click", (e) => {
      if (e.target.classList.contains("resize-handle")) return;

      document
        .querySelectorAll(".draggable")
        .forEach((d) => d.classList.remove("selected"));
      el.classList.add("selected");

      const nodeIdx = parseInt(el.getAttribute("data-node-idx"));
      showProperties(nodeIdx);

      e.stopPropagation();
    });

    // Drag to move
    el.addEventListener("mousedown", (e) => {
      if (e.target.classList.contains("resize-handle")) {
        // Start resizing
        resizingElement = el;
        resizeCorner = e.target.getAttribute("data-corner");

        const rect = el.getBoundingClientRect();
        const canvas = document
          .getElementById("canvas")
          .getBoundingClientRect();

        startX = e.clientX;
        startY = e.clientY;
        startWidth = parseInt(el.style.width);
        startHeight = parseInt(el.style.height);
        startLeft = parseInt(el.style.left);
        startTop = parseInt(el.style.top);

        e.preventDefault();
        e.stopPropagation();
        return;
      }

      draggedElement = el;
      draggedElement.classList.add("dragging");

      const rect = el.getBoundingClientRect();
      const canvas = document.getElementById("canvas").getBoundingClientRect();

      offsetX = e.clientX - rect.left;
      offsetY = e.clientY - rect.top;

      e.preventDefault();
    });
  });

  // Click canvas to deselect
  document.getElementById("canvas").addEventListener("click", (e) => {
    if (e.target.id === "canvas") {
      document
        .querySelectorAll(".draggable")
        .forEach((d) => d.classList.remove("selected"));
      hideProperties();
    }
  });

  document.addEventListener("mousemove", (e) => {
    if (resizingElement) {
      const deltaX = e.clientX - startX;
      const deltaY = e.clientY - startY;

      let newWidth = startWidth;
      let newHeight = startHeight;
      let newLeft = startLeft;
      let newTop = startTop;

      if (resizeCorner.includes("e")) {
        newWidth = startWidth + deltaX;
      }
      if (resizeCorner.includes("w")) {
        newWidth = startWidth - deltaX;
        newLeft = startLeft + deltaX;
      }
      if (resizeCorner.includes("s")) {
        newHeight = startHeight + deltaY;
      }
      if (resizeCorner.includes("n")) {
        newHeight = startHeight - deltaY;
        newTop = startTop + deltaY;
      }

      resizingElement.style.width = `${Math.max(10, newWidth)}px`;
      resizingElement.style.height = `${Math.max(10, newHeight)}px`;
      resizingElement.style.left = `${newLeft}px`;
      resizingElement.style.top = `${newTop}px`;

      return;
    }

    if (!draggedElement) return;

    const canvas = document.getElementById("canvas").getBoundingClientRect();
    const newX = e.clientX - canvas.left - offsetX;
    const newY = e.clientY - canvas.top - offsetY;

    draggedElement.style.left = `${newX}px`;
    draggedElement.style.top = `${newY}px`;
  });

  document.addEventListener("mouseup", (e) => {
    if (resizingElement) {
      const nodeIdx = parseInt(resizingElement.getAttribute("data-node-idx"));
      currentSpec.nodes[nodeIdx].width = Math.round(
        parseInt(resizingElement.style.width)
      );
      currentSpec.nodes[nodeIdx].height = Math.round(
        parseInt(resizingElement.style.height)
      );
      currentSpec.nodes[nodeIdx].x = Math.round(
        parseInt(resizingElement.style.left)
      );
      currentSpec.nodes[nodeIdx].y = Math.round(
        parseInt(resizingElement.style.top)
      );

      hasChanges = true;
      document.getElementById("save-btn").disabled = false;
      document.getElementById("copy-btn").disabled = false;

      resizingElement = null;
      resizeCorner = null;
      return;
    }

    if (!draggedElement) return;

    draggedElement.classList.remove("dragging");

    // Update the spec
    const nodeIdx = parseInt(draggedElement.getAttribute("data-node-idx"));
    const canvas = document.getElementById("canvas").getBoundingClientRect();
    const newX = e.clientX - canvas.left - offsetX;
    const newY = e.clientY - canvas.top - offsetY;

    currentSpec.nodes[nodeIdx].x = Math.round(newX);
    currentSpec.nodes[nodeIdx].y = Math.round(newY);

    hasChanges = true;
    document.getElementById("save-btn").disabled = false;

    draggedElement = null;
  });
}

function setupImageDropZone() {
  const canvas = document.getElementById("canvas");
  if (!canvas) return;

  // Prevent default drag behaviors
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    canvas.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  // Highlight drop zone when item is dragged over it
  ["dragenter", "dragover"].forEach((eventName) => {
    canvas.addEventListener(eventName, () => {
      canvas.style.outline = "3px dashed #2196f3";
      canvas.style.outlineOffset = "5px";
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    canvas.addEventListener(eventName, () => {
      canvas.style.outline = "";
      canvas.style.outlineOffset = "";
    });
  });

  // Handle dropped files
  canvas.addEventListener("drop", handleDrop);

  async function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;

    if (files.length === 0) return;

    // Only handle image files
    const imageFile = Array.from(files).find((file) =>
      file.type.startsWith("image/")
    );

    if (!imageFile) {
      showToast("Please drop an image file", true);
      return;
    }

    await uploadAndAddImage(imageFile, e.offsetX, e.offsetY);
  }
}

async function uploadAndAddImage(file, x, y) {
  try {
    // Upload the image
    const formData = new FormData();
    formData.append("file", file);
    formData.append("specName", currentSpecName);

    const uploadResponse = await fetch(`${API_URL}/api/upload-image`, {
      method: "POST",
      body: formData,
    });

    const uploadResult = await uploadResponse.json();

    if (!uploadResponse.ok) {
      throw new Error(uploadResult.error || "Failed to upload image");
    }

    // Get image dimensions
    const img = new Image();
    const imageLoaded = new Promise((resolve) => {
      img.onload = resolve;
    });
    img.src = URL.createObjectURL(file);
    await imageLoaded;

    // Add new image node to spec
    const newNode = {
      type: "image",
      asset_description: `Uploaded image: ${file.name}`,
      filename: uploadResult.filename,
      x: Math.round(x),
      y: Math.round(y),
      width: img.width,
      height: img.height,
      rotation: 0,
      opacity: 1,
    };

    currentSpec.nodes.push(newNode);

    // Save the updated spec
    hasChanges = true;
    await saveSpec();

    // Re-render to show the new image
    renderSpec(currentSpec, currentSpecName);

    showToast("✓ Image added successfully!");
  } catch (error) {
    showToast("Error adding image: " + error.message, true);
    console.error("Upload error:", error);
  }
}

async function copySpec() {
  if (!currentSpec || !currentSpecName) return;

  try {
    const specJson = JSON.stringify(currentSpec, null, 2);
    await navigator.clipboard.writeText(specJson);
    showToast("✓ Spec copied to clipboard!");
  } catch (error) {
    showToast("Error copying spec: " + error.message, true);
  }
}

async function saveSpec() {
  if (!currentSpec || !currentSpecName) return;

  try {
    // Save via API endpoint
    const response = await fetch(`${API_URL}/api/save-spec`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        specName: currentSpecName,
        spec: currentSpec,
      }),
    });

    const result = await response.json();

    if (response.ok) {
      showToast("✓ Spec saved successfully!");
      hasChanges = false;
      document.getElementById("save-btn").disabled = true;
    } else {
      throw new Error(result.error || "Failed to save spec");
    }
  } catch (error) {
    showToast("Error saving spec: " + error.message, true);
    console.error("Save error:", error);
  }
}

function showProperties(nodeIdx) {
  const node = currentSpec.nodes[nodeIdx];
  const propertiesPanel = document.getElementById("properties");
  const propertiesContent = document.getElementById("properties-content");

  propertiesPanel.classList.remove("hidden");

  let html = `
        <div class="property-group">
            <h3>Transform</h3>
            <div class="property-row">
                <label>X</label>
                <input type="number" data-prop="x" data-node-idx="${nodeIdx}" value="${
    node.x
  }" />
            </div>
            <div class="property-row">
                <label>Y</label>
                <input type="number" data-prop="y" data-node-idx="${nodeIdx}" value="${
    node.y
  }" />
            </div>
            <div class="property-row">
                <label>Width</label>
                <input type="number" data-prop="width" data-node-idx="${nodeIdx}" value="${
    node.width
  }" />
            </div>
            <div class="property-row">
                <label>Height</label>
                <input type="number" data-prop="height" data-node-idx="${nodeIdx}" value="${
    node.height
  }" />
            </div>
            <div class="property-row">
                <label>Rotation</label>
                <input type="number" data-prop="rotation" data-node-idx="${nodeIdx}" value="${
    node.rotation
  }" />
            </div>
            <div class="property-row">
                <label>Opacity</label>
                <input type="number" step="0.1" min="0" max="1" data-prop="opacity" data-node-idx="${nodeIdx}" value="${
    node.opacity !== undefined ? node.opacity : 1
  }" />
            </div>
        </div>
    `;

  if (node.type === "text") {
    html += `
            <div class="property-group">
                <h3>Text</h3>
                <div class="property-row">
                    <label>Content</label>
                    <textarea data-prop="text" data-node-idx="${nodeIdx}" rows="3">${escapeHtml(
      node.text
    )}</textarea>
                </div>
                <div class="property-row">
                    <label>Font Family</label>
                    <div class="font-picker" data-node-idx="${nodeIdx}">
                        <input type="text"
                               class="font-picker-input"
                               data-prop="font-family"
                               data-node-idx="${nodeIdx}"
                               value="${node["font-family"] || node.font_family}"
                               readonly
                               placeholder="Select font..."/>
                        <div class="font-picker-dropdown" style="display: none;">
                            <input type="text"
                                   class="font-picker-search"
                                   placeholder="Search fonts..." />
                            <div class="font-picker-list"></div>
                        </div>
                    </div>
                </div>
                <div class="property-row">
                    <label>Font Size</label>
                    <input type="number" data-prop="font-size" data-node-idx="${nodeIdx}" value="${
      node["font-size"] || node.font_size
    }" />
                </div>
                <div class="property-row">
                    <label>Color</label>
                    <input type="text" data-prop="color" data-node-idx="${nodeIdx}" value="${
      node.color
    }" />
                </div>
                <div class="property-row">
                    <label>Text Align</label>
                    <select data-prop="text-align" data-node-idx="${nodeIdx}">
                        <option value="left" ${
                          (node["text-align"] || node.text_align) === "left"
                            ? "selected"
                            : ""
                        }>Left</option>
                        <option value="center" ${
                          (node["text-align"] || node.text_align) === "center"
                            ? "selected"
                            : ""
                        }>Center</option>
                        <option value="right" ${
                          (node["text-align"] || node.text_align) === "right"
                            ? "selected"
                            : ""
                        }>Right</option>
                    </select>
                </div>
                <div class="property-row">
                    <label>Font Weight</label>
                    <select data-prop="font-weight" data-node-idx="${nodeIdx}">
                        <option value="normal" ${
                          (node["font-weight"] || node.font_weight) === "normal"
                            ? "selected"
                            : ""
                        }>Normal</option>
                        <option value="bold" ${
                          (node["font-weight"] || node.font_weight) === "bold"
                            ? "selected"
                            : ""
                        }>Bold</option>
                        <option value="100" ${
                          (node["font-weight"] || node.font_weight) === "100"
                            ? "selected"
                            : ""
                        }>100</option>
                        <option value="200" ${
                          (node["font-weight"] || node.font_weight) === "200"
                            ? "selected"
                            : ""
                        }>200</option>
                        <option value="300" ${
                          (node["font-weight"] || node.font_weight) === "300"
                            ? "selected"
                            : ""
                        }>300</option>
                        <option value="400" ${
                          (node["font-weight"] || node.font_weight) === "400"
                            ? "selected"
                            : ""
                        }>400</option>
                        <option value="500" ${
                          (node["font-weight"] || node.font_weight) === "500"
                            ? "selected"
                            : ""
                        }>500</option>
                        <option value="600" ${
                          (node["font-weight"] || node.font_weight) === "600"
                            ? "selected"
                            : ""
                        }>600</option>
                        <option value="700" ${
                          (node["font-weight"] || node.font_weight) === "700"
                            ? "selected"
                            : ""
                        }>700</option>
                        <option value="800" ${
                          (node["font-weight"] || node.font_weight) === "800"
                            ? "selected"
                            : ""
                        }>800</option>
                        <option value="900" ${
                          (node["font-weight"] || node.font_weight) === "900"
                            ? "selected"
                            : ""
                        }>900</option>
                    </select>
                </div>
                <div class="property-row">
                    <label>Font Style</label>
                    <select data-prop="font-style" data-node-idx="${nodeIdx}">
                        <option value="normal" ${
                          (node["font-style"] || node.font_style) === "normal"
                            ? "selected"
                            : ""
                        }>Normal</option>
                        <option value="italic" ${
                          (node["font-style"] || node.font_style) === "italic"
                            ? "selected"
                            : ""
                        }>Italic</option>
                    </select>
                </div>
                <div class="property-row">
                    <label>Text Decoration</label>
                    <select data-prop="text-decoration" data-node-idx="${nodeIdx}">
                        <option value="none" ${
                          (node["text-decoration"] || node.text_decoration) ===
                          "none"
                            ? "selected"
                            : ""
                        }>None</option>
                        <option value="underline" ${
                          (node["text-decoration"] || node.text_decoration) ===
                          "underline"
                            ? "selected"
                            : ""
                        }>Underline</option>
                    </select>
                </div>
                <div class="property-row">
                    <label>Text Transform</label>
                    <select data-prop="text-transform" data-node-idx="${nodeIdx}">
                        <option value="none" ${
                          (node["text-transform"] || node.text_transform) ===
                          "none"
                            ? "selected"
                            : ""
                        }>None</option>
                        <option value="uppercase" ${
                          (node["text-transform"] || node.text_transform) ===
                          "uppercase"
                            ? "selected"
                            : ""
                        }>Uppercase</option>
                        <option value="lowercase" ${
                          (node["text-transform"] || node.text_transform) ===
                          "lowercase"
                            ? "selected"
                            : ""
                        }>Lowercase</option>
                        <option value="capitalize" ${
                          (node["text-transform"] || node.text_transform) ===
                          "capitalize"
                            ? "selected"
                            : ""
                        }>Capitalize</option>
                    </select>
                </div>
                <div class="property-row">
                    <label>Line Height</label>
                    <input type="number" step="0.1" data-prop="line-height" data-node-idx="${nodeIdx}" value="${
      node["line-height"] || node.line_height || 1.2
    }" />
                </div>
            </div>
        `;
  } else if (node.type === "svg") {
    html += `
            <div class="property-group">
                <h3>SVG</h3>
                <div class="property-row">
                    <label>Description</label>
                    <textarea data-prop="svg_description" data-node-idx="${nodeIdx}" rows="3">${escapeHtml(
      node.svg_description || ""
    )}</textarea>
                </div>
            </div>
            <div class="property-group">
                <h3>SVG Content</h3>
                <div class="property-row" style="display: block;">
                    <label>Edit SVG Markup</label>
                    <textarea id="svg-editor" rows="15" style="font-family: monospace; font-size: 24px; width: 100%; margin-top: 8px;" placeholder="Loading SVG content..."></textarea>
                    <button id="save-svg-btn" style="margin-top: 10px; padding: 10px 20px; background: #4caf50; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; width: 100%;">Save SVG</button>
                </div>
            </div>
        `;
  } else if (node.type === "image") {
    html += `
            <div class="property-group">
                <h3>Image</h3>
                <div class="property-row">
                    <label>Description</label>
                    <textarea data-prop="asset_description" data-node-idx="${nodeIdx}" rows="3">${escapeHtml(
      node.asset_description || ""
    )}</textarea>
                </div>
                ${
                  node.filename
                    ? `<div class="property-row" style="display: block;">
                    <button id="download-image-btn" style="margin-top: 10px; padding: 10px 20px; background: #2196f3; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; width: 100%;">Download Image</button>
                </div>`
                    : ""
                }
            </div>
        `;
  }

  propertiesContent.innerHTML = html;

  // Add event listeners to update spec on change
  propertiesContent.querySelectorAll("input, select, textarea").forEach((input) => {
    // Skip the SVG editor textarea - it has its own save button
    if (input.id === "svg-editor") return;
    // Skip the font picker inputs - they have their own handlers
    if (input.classList.contains("font-picker-input") || input.classList.contains("font-picker-search")) return;

    input.addEventListener("input", (e) => {
      const nodeIdx = parseInt(e.target.getAttribute("data-node-idx"));
      const prop = e.target.getAttribute("data-prop");
      const value =
        e.target.type === "number"
          ? parseFloat(e.target.value)
          : e.target.value;

      currentSpec.nodes[nodeIdx][prop] = value;

      hasChanges = true;
      document.getElementById("save-btn").disabled = false;
      document.getElementById("copy-btn").disabled = false;

      // Re-render to see changes
      renderSpec(currentSpec, currentSpecName);
    });
  });

  // Handle SVG editing
  if (node.type === "svg" && node.filename) {
    loadSvgContent(nodeIdx, node.filename);
  }

  // Handle image download
  if (node.type === "image" && node.filename) {
    const downloadBtn = document.getElementById("download-image-btn");
    if (downloadBtn) {
      downloadBtn.onclick = () => {
        downloadImage(node.filename);
      };
    }
  }

  // Setup font picker for text nodes
  if (node.type === "text") {
    setupFontPicker(nodeIdx);
  }
}

function setupFontPicker(nodeIdx) {
  const fontPicker = document.querySelector(`.font-picker[data-node-idx="${nodeIdx}"]`);
  if (!fontPicker) return;

  const pickerInput = fontPicker.querySelector(".font-picker-input");
  const dropdown = fontPicker.querySelector(".font-picker-dropdown");
  const searchInput = fontPicker.querySelector(".font-picker-search");
  const fontList = fontPicker.querySelector(".font-picker-list");

  // Populate font list
  function renderFontList(filter = "") {
    const filteredFonts = FONTS.filter((font) =>
      font.toLowerCase().includes(filter.toLowerCase())
    );

    fontList.innerHTML = filteredFonts
      .map(
        (font) => `
      <div class="font-option" data-font="${font}" style="font-family: '${font}';">
        ${font}
      </div>
    `
      )
      .join("");

    // Add click handlers
    fontList.querySelectorAll(".font-option").forEach((option) => {
      option.addEventListener("click", () => {
        const selectedFont = option.getAttribute("data-font");
        pickerInput.value = selectedFont;

        // Update spec
        currentSpec.nodes[nodeIdx]["font-family"] = selectedFont;
        hasChanges = true;
        document.getElementById("save-btn").disabled = false;
        document.getElementById("copy-btn").disabled = false;

        // Re-render to see changes
        renderSpec(currentSpec, currentSpecName);

        // Close dropdown
        dropdown.style.display = "none";
      });
    });
  }

  // Initial render
  renderFontList();

  // Toggle dropdown
  pickerInput.addEventListener("click", (e) => {
    e.stopPropagation();
    const isVisible = dropdown.style.display === "block";
    dropdown.style.display = isVisible ? "none" : "block";
    if (!isVisible) {
      searchInput.value = "";
      renderFontList();
      searchInput.focus();
    }
  });

  // Filter fonts on search
  searchInput.addEventListener("input", (e) => {
    renderFontList(e.target.value);
  });

  // Close dropdown when clicking outside
  document.addEventListener("click", (e) => {
    if (!fontPicker.contains(e.target)) {
      dropdown.style.display = "none";
    }
  });
}

function downloadImage(filename) {
  const imagePath = `${SPECS_DIR}/${currentSpecName}/${filename}`;
  const link = document.createElement("a");
  link.href = imagePath;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  showToast("✓ Downloading image...");
}

async function loadSvgContent(nodeIdx, filename) {
  const svgEditor = document.getElementById("svg-editor");
  if (!svgEditor) return;

  try {
    const response = await fetch(
      `${API_URL}/api/read-svg?specName=${encodeURIComponent(
        currentSpecName
      )}&filename=${encodeURIComponent(filename)}`
    );
    const result = await response.json();

    if (response.ok && result.success) {
      svgEditor.value = result.content;
    } else {
      svgEditor.value = `<!-- Error loading SVG: ${result.error} -->`;
    }
  } catch (error) {
    svgEditor.value = `<!-- Error loading SVG: ${error.message} -->`;
  }

  // Add save button handler
  const saveSvgBtn = document.getElementById("save-svg-btn");
  if (saveSvgBtn) {
    saveSvgBtn.onclick = async () => {
      await saveSvgContent(nodeIdx, filename, svgEditor.value);
    };
  }
}

async function saveSvgContent(nodeIdx, filename, content) {
  try {
    const response = await fetch(`${API_URL}/api/save-svg`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        specName: currentSpecName,
        filename: filename,
        content: content,
      }),
    });

    const result = await response.json();

    if (response.ok) {
      showToast("✓ SVG saved successfully!");

      // Reload the SVG in the canvas
      renderSpec(currentSpec, currentSpecName);
    } else {
      throw new Error(result.error || "Failed to save SVG");
    }
  } catch (error) {
    showToast("Error saving SVG: " + error.message, true);
    console.error("Save SVG error:", error);
  }
}

function hideProperties() {
  document.getElementById("properties").classList.add("hidden");
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Delete key handler
document.addEventListener("keydown", (e) => {
  if (e.key === "Delete" || e.key === "Backspace") {
    // Don't delete if user is typing in an input field
    const activeElement = document.activeElement;
    if (
      activeElement &&
      (activeElement.tagName === "INPUT" ||
        activeElement.tagName === "TEXTAREA" ||
        activeElement.tagName === "SELECT")
    ) {
      return;
    }

    const selected = document.querySelector(".draggable.selected");
    if (selected) {
      const nodeIdx = parseInt(selected.getAttribute("data-node-idx"));

      // Remove from spec
      currentSpec.nodes.splice(nodeIdx, 1);

      hasChanges = true;
      document.getElementById("save-btn").disabled = false;

      // Hide properties panel
      hideProperties();

      // Re-render
      renderSpec(currentSpec, currentSpecName);

      e.preventDefault();
    }
  }
});

// Handle browser back/forward buttons
window.addEventListener("popstate", (event) => {
  if (event.state && event.state.specName) {
    loadSpec(event.state.specName);
  }
});

// Check URL for design parameter and auto-load
function loadFromUrl() {
  const urlParams = new URLSearchParams(window.location.search);
  const designId = urlParams.get("design");
  if (designId) {
    loadSpec(designId);
  }
}

// Initialize
document.getElementById("copy-btn").addEventListener("click", copySpec);
document.getElementById("save-btn").addEventListener("click", saveSpec);
loadSpecList().then(() => {
  // After spec list loads, check if URL has a design parameter
  loadFromUrl();
});
