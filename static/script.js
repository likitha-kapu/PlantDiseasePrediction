const fileInput = document.getElementById("fileInput");
const dropArea = document.getElementById("dropArea");
const preview = document.getElementById("preview");
const resultBox = document.getElementById("result");
const form = document.getElementById("uploadForm");

// Drag & Drop
dropArea.addEventListener("click", () => fileInput.click());
dropArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropArea.style.borderColor = "#ffeb3b";
});
dropArea.addEventListener("dragleave", () => {
  dropArea.style.borderColor = "#aaa";
});
dropArea.addEventListener("drop", (e) => {
  e.preventDefault();
  fileInput.files = e.dataTransfer.files;
  showPreview();
});

fileInput.addEventListener("change", showPreview);

function showPreview() {
  preview.innerHTML = "";
  const file = fileInput.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = () => {
      const img = document.createElement("img");
      img.src = reader.result;
      preview.appendChild(img);
    };
    reader.readAsDataURL(file);
  }
}

// Submit Form
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const formData = new FormData(form);
  resultBox.style.display = "block";
  resultBox.innerHTML = "⏳ Detecting...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    resultBox.innerHTML = `
      <div class="prediction-card">
        <h3>✅ Prediction</h3>
        <p><strong>Disease:</strong> ${data.class_name}</p>
        <p><strong>Confidence:</strong> ${data.confidence}</p>
        <h3>📝 Description</h3>
        <p>${data.description}</p>
        <h3>💊 Treatment</h3>
        <p>${data.treatment}</p>
      </div>
    `;
  } catch (err) {
    resultBox.innerHTML = `<p style="color:red;">❌ Error: Could not fetch prediction.</p>`;
    console.error(err);
  }
});
