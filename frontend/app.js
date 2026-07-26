// --- LeafGuard Frontend Application Logic ---

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const dropZonePrompt = document.getElementById('drop-zone-prompt');

    const analyseBtn = document.getElementById('analyse-btn');
    const clearBtn = document.getElementById('clear-btn');
    const btnSpinner = document.getElementById('btn-spinner');

    const resultsCard = document.getElementById('results-card');
    const diagnosticTime = document.getElementById('diagnostic-time');
    const diagnosisBadge = document.getElementById('diagnosis-badge');
    const confidenceValue = document.getElementById('confidence-value');
    const primaryProgressFill = document.getElementById('primary-progress-fill');
    const alternativesList = document.getElementById('alternatives-list');

    const recCard = document.getElementById('rec-card');
    const recTitle = document.getElementById('rec-title');
    const recDesc = document.getElementById('rec-desc');

    // Selected file placeholder
    let selectedFile = null;

    // API URL configuration - defaults to local host where FastAPI is running
    const API_URL = `${window.location.origin}/predict`;

    // Diagnostic references / details for each disease state
    const DIAGNOSTIC_REFS = {
        'potato___early_blight': {
            title: 'Early Blight (Alternaria solani) Detected',
            desc: 'Early blight is caused by the fungus Alternaria solani. It manifests as concentric rings or "target spots" on older leaves. To manage early blight, apply copper-based fungicides, prune infected lower leaves to increase air circulation, and ensure crop rotation with non-solanaceous crops for future seasons.',
            status: 'warning'
        },
        'potato___late_blight': {
            title: 'Late Blight (Phytophthora infestans) Detected',
            desc: 'Late blight is caused by the oomycete Phytophthora infestans. This is a highly destructive disease that causes dark, water-soaked lesions on leaves and stems. Under humid conditions, a white fuzzy growth appears. Immediately destroy infected plants. Apply targeted fungicides (e.g. chlorothalonil) and avoid overhead watering.',
            status: 'danger'
        },
        'potato___healthy': {
            title: 'Healthy Potato Foliage',
            desc: 'No visual symptoms of common pathogens detected. Maintain good cultural practices: ensure proper fertilization, drip irrigation to prevent wet foliage, and inspect plants weekly for early detection of pests or blight.',
            status: 'healthy'
        }
    };

    // --- Drag and Drop Handlers ---

    // Prevent default browser behaviors for drag actions
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Visual highlights for drag interactions
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('drag-over'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('drag-over'), false);
    });

    // Handle dropped files
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // Trigger file input dialog on click
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle file input changes
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Process selected file and render visual preview
    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (PNG, JPG, or JPEG).');
            return;
        }

        selectedFile = file;

        // Read file contents for local preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            dropZonePrompt.style.display = 'none';

            // Enable analysis button and show clear button
            analyseBtn.removeAttribute('disabled');
            clearBtn.style.display = 'inline-flex';
        };
        reader.readAsDataURL(file);
    }

    // --- Reset UI state ---
    clearBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent opening file dialog
        resetUI();
    });

    function resetUI() {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        imagePreview.style.display = 'none';
        dropZonePrompt.style.display = 'flex';

        analyseBtn.setAttribute('disabled', 'true');
        clearBtn.style.display = 'none';
        btnSpinner.style.display = 'none';

        resultsCard.style.display = 'none';
    }

    // --- API Request & Prediction Handling ---
    analyseBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Visual loading state
        analyseBtn.setAttribute('disabled', 'true');
        btnSpinner.style.display = 'inline-block';
        analyseBtn.querySelector('.btn-text').textContent = 'Analyzing...';

        // Hide previous results during new request
        resultsCard.style.display = 'none';

        const formData = new FormData();
        formData.append('file', selectedFile);

        const startTime = performance.now();
        try {
            console.log(`Sending image to FastAPI API endpoint: ${API_URL}`);
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to analyze the leaf image.');
            }

            const data = await response.json();
            const duration = Math.round(performance.now() - startTime);
            renderResults(data, duration);

        } catch (error) {
            console.error('Error contacting FastAPI server:', error);
            alert(`An error occurred: ${error.message}`);
        } finally {
            // Restore button state
            btnSpinner.style.display = 'none';
            analyseBtn.removeAttribute('disabled');
            analyseBtn.querySelector('.btn-text').textContent = 'Analyze Image';
        }
    });

    // Helper to format raw class strings for display (e.g. Potato___Early_blight -> Potato - Early Blight)
    function formatClassName(name) {
        return name
            .replace(/___/g, ' - ')
            .replace(/_/g, ' ')
            .replace(/\b\w/g, c => c.toUpperCase());
    }

    // Render server prediction payload to UI cards
    function renderResults(data, duration) {
        // Display results panel
        resultsCard.style.display = 'flex';

        // Update diagnostic completion timestamp
        const now = new Date();
        diagnosticTime.textContent = `Analysis completed at ${now.toLocaleTimeString()} | Duration: ${duration}ms`;

        const predClass = data.predicted_class;
        const confidencePct = (data.confidence * 100).toFixed(1);

        // Update Primary Diagnosis Badge & Value
        diagnosisBadge.textContent = formatClassName(predClass);
        confidenceValue.textContent = `${confidencePct}%`;

        // Retrieve advisory reference metadata matching the predicted class
        const classKey = predClass.toLowerCase();
        const ref = DIAGNOSTIC_REFS[classKey] || {
            title: 'Unknown Leaf Condition Detected',
            desc: 'The model has identified a class label but did not map a clinical advisory reference. Maintain standard agricultural crop hygiene.',
            status: 'warning'
        };

        // Reset badge status classes and assign the correct one
        diagnosisBadge.className = 'diagnosis-badge';
        primaryProgressFill.className = 'progress-bar-fill';

        if (ref.status === 'healthy') {
            diagnosisBadge.classList.add('status-healthy');
            primaryProgressFill.classList.add('bar-healthy');
            recCard.style.borderLeftColor = 'var(--accent-emerald)';
            recCard.style.background = 'rgba(16, 185, 129, 0.05)';
        } else if (ref.status === 'warning') {
            diagnosisBadge.classList.add('status-warning');
            primaryProgressFill.classList.add('bar-warning');
            recCard.style.borderLeftColor = 'var(--accent-yellow)';
            recCard.style.background = 'rgba(245, 158, 11, 0.05)';
        } else {
            diagnosisBadge.classList.add('status-danger');
            primaryProgressFill.classList.add('bar-danger');
            recCard.style.borderLeftColor = 'var(--accent-red)';
            recCard.style.background = 'rgba(239, 68, 68, 0.05)';
        }

        // Animate primary progress bar fill
        setTimeout(() => {
            primaryProgressFill.style.width = `${confidencePct}%`;
        }, 50);

        // Populate alternative classifications list
        alternativesList.innerHTML = '';
        data.predictions.forEach(pred => {
            const pct = (pred.confidence * 100).toFixed(1);

            const altRow = document.createElement('div');
            altRow.className = 'alt-row';
            altRow.innerHTML = `
                <div class="alt-info">
                    <span class="alt-name">${formatClassName(pred.class_name)}</span>
                    <span class="alt-pct">${pct}%</span>
                </div>
                <div class="alt-bar-container">
                    <div class="alt-bar-fill" style="width: 0%"></div>
                </div>
            `;

            alternativesList.appendChild(altRow);

            // Animate alternative progress bars
            setTimeout(() => {
                altRow.querySelector('.alt-bar-fill').style.width = `${pct}%`;
            }, 100);
        });

        // Update Recommendations / Advisory text
        recTitle.textContent = ref.title;
        recDesc.textContent = ref.desc;

        // Scroll results card into view on small screens
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});
