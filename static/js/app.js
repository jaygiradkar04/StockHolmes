// Application State
let selectedSymbol = null;
let modelStatus = { loaded: false, model_key: null, device: null };
let isModelLoading = false;
let isPredicting = false;

// Global Data Storage for instant chart toggles/zooming
let currentHistory = [];
let currentForecast = [];
let currentForecastEnhanced = [];

// DOM Elements
const elements = {
    // Model status
    statusDot: document.getElementById('status-dot'),
    statusText: document.getElementById('status-text'),
    modelStatusBtn: document.getElementById('model-status-btn'),
    
    // Portfolio summary
    portTotalValue: document.getElementById('port-total-value'),
    portTotalCost: document.getElementById('port-total-cost'),
    portTotalPnl: document.getElementById('port-total-pnl'),
    portTodayPnl: document.getElementById('port-today-pnl'),
    
    // Add holding form
    addHoldingForm: document.getElementById('add-holding-form'),
    stockSearchInput: document.getElementById('stock-search-input'),
    holdingShares: document.getElementById('holding-shares'),
    holdingPrice: document.getElementById('holding-price'),
    addHoldingBtn: document.getElementById('add-holding-btn'),
    searchSuggestions: document.getElementById('search-suggestions'),
    
    // Holdings list
    holdingsCount: document.getElementById('holdings-count'),
    holdingsListContainer: document.getElementById('holdings-list-container'),
    
    // Global search
    globalSearchInput: document.getElementById('global-search-input'),
    globalSearchBtn: document.getElementById('global-search-btn'),
    globalSuggestions: document.getElementById('global-suggestions'),
    
    // Stock details
    stockDetailsSection: document.getElementById('stock-details-section'),
    stockDetailName: document.getElementById('stock-detail-name'),
    stockDetailSymbol: document.getElementById('stock-detail-symbol'),
    stockDetailExchange: document.getElementById('stock-detail-exchange'),
    stockDetailPrice: document.getElementById('stock-detail-price'),
    stockDetailChange: document.getElementById('stock-detail-change'),
    
    // Stock detailed metrics
    metricOpen: document.getElementById('metric-open'),
    metricRange: document.getElementById('metric-range'),
    metric52w: document.getElementById('metric-52w'),
    metricMcap: document.getElementById('metric-mcap'),
    metricPe: document.getElementById('metric-pe'),
    metricVolume: document.getElementById('metric-volume'),
    metricDiv: document.getElementById('metric-div'),
    metricSector: document.getElementById('metric-sector'),
    
    // Chart Controls & Elements
    chartWrapper: document.getElementById('chart-wrapper'),
    chartSubLabel: document.getElementById('chart-sub-label'),
    chartActionSection: document.getElementById('chart-action-section'),
    runPredictionBtn: document.getElementById('run-prediction-btn'),
    chartLoader: document.getElementById('chart-loader'),
    loaderStatus: document.getElementById('loader-status'),
    hyperparamsTriggerBtn: document.getElementById('hyperparams-trigger-btn'),
    hyperparametersPanel: document.getElementById('hyperparameters-panel'),
    closeHyperparamsBtn: document.getElementById('close-hyperparams-btn'),
    
    // Chart type & Date range elements
    chartTypeSelect: document.getElementById('chart-type-select'),
    chartPeriodSelect: document.getElementById('chart-period-select'),
    
    // Hyperparameters values
    paramPredLen: document.getElementById('param-pred-len'),
    paramTemp: document.getElementById('param-temp'),
    paramTopp: document.getElementById('param-topp'),
    valTemp: document.getElementById('val-temp'),
    valTopp: document.getElementById('val-topp'),
    
    // Model config modal
    settingsModal: document.getElementById('settings-modal'),
    closeSettingsModal: document.getElementById('close-settings-modal'),
    selectModel: document.getElementById('select-model'),
    selectDevice: document.getElementById('select-device'),
    modelDesc: document.getElementById('model-desc'),
    loadModelSubmitBtn: document.getElementById('load-model-submit-btn'),
    modalLoadingBar: document.getElementById('modal-loading-bar'),
    loadingPct: document.getElementById('loading-pct'),
    loadingPctBar: document.getElementById('loading-pct-bar'),
    inputSpreadsheetId: document.getElementById('input-spreadsheet-id'),
    
    // Forecast Analysis
    analysisSection: document.getElementById('analysis-section'),
    analysisSignal: document.getElementById('analysis-signal'),
    analysisTarget: document.getElementById('analysis-target'),
    analysisMax: document.getElementById('analysis-max'),
    analysisMin: document.getElementById('analysis-min'),
    analysisSummary: document.getElementById('analysis-summary'),
    
    // Sentiment & News Elements
    enhancedPredictionToggle: document.getElementById('enhanced-prediction-toggle'),
    enhancedProgressContainer: document.getElementById('enhanced-progress-container'),
    enhancedProgressBarFill: document.getElementById('enhanced-progress-fill'),
    enhancedProgressPct: document.getElementById('enhanced-progress-pct'),
    sentimentNewsSection: document.getElementById('sentiment-news-section'),
    netSentimentVal: document.getElementById('net-sentiment-val'),
    newsListContainer: document.getElementById('news-list-container'),
    syncSheetBtn: document.getElementById('sync-sheet-btn'),
    
    // Notification Toast
    toastMessage: document.getElementById('toast-message'),
    toastText: document.getElementById('toast-text'),
    toastIcon: document.getElementById('toast-icon')
};

// Model descriptions dictionary for dynamic info update
const modelDescriptions = {
    'kronos-mini': 'Mini model (4.1M parameters). Ideal for ultra-fast predictions and standard hardware constraints.',
    'kronos-small': 'Small model (24.7M parameters). Balanced accuracy, memory footprint, and CPU speed (Recommended).',
    'kronos-base': 'Base model (102.3M parameters). Maximum forecasting performance. Weights take longer to download initially.'
};

// --- Initializing App ---
document.addEventListener('DOMContentLoaded', () => {
    fetchModelStatus();
    fetchPortfolio();
    setupEventListeners();
});

// --- Event Listeners Setup ---
function setupEventListeners() {
    // Model Config Modal toggles
    elements.modelStatusBtn.addEventListener('click', openSettingsModal);
    elements.closeSettingsModal.addEventListener('click', closeSettingsModal);
    
    // Update modal description on select change
    elements.selectModel.addEventListener('change', (e) => {
        elements.modelDesc.textContent = modelDescriptions[e.target.value] || '';
    });
    
    // Load Model weights submit
    elements.loadModelSubmitBtn.addEventListener('click', handleLoadModel);
    
    // Form submission
    elements.addHoldingForm.addEventListener('submit', handleAddHolding);
    
    // Autocomplete events (Add holding input)
    setupAutoComplete(elements.stockSearchInput, elements.searchSuggestions, (selected) => {
        elements.stockSearchInput.value = selected.symbol;
        elements.holdingPrice.value = ''; // Let user fetch or type price
        // Query details to auto-populate buy price if available
        fetch(`/api/stock-details?symbol=${selected.symbol}`)
            .then(res => res.json())
            .then(data => {
                if (data.current_price) {
                    elements.holdingPrice.value = data.current_price.toFixed(2);
                }
            })
            .catch(err => console.error(err));
    });

    // Autocomplete events (Global search input)
    setupAutoComplete(elements.globalSearchInput, elements.globalSuggestions, (selected) => {
        elements.globalSearchInput.value = '';
        selectStock(selected.symbol);
    });

    // Search button click handler
    elements.globalSearchBtn.addEventListener('click', () => {
        const val = elements.globalSearchInput.value.trim();
        if (val) {
            selectStock(val);
        }
    });

    // Enter keypress handler on search input
    elements.globalSearchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const val = elements.globalSearchInput.value.trim();
            if (val) {
                selectStock(val);
            }
        }
    });
    
    // Hyperparameters popover controls
    elements.hyperparamsTriggerBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        elements.hyperparametersPanel.classList.toggle('hidden');
    });
    
    elements.closeHyperparamsBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        elements.hyperparametersPanel.classList.add('hidden');
    });
    
    elements.hyperparametersPanel.addEventListener('click', (e) => {
        e.stopPropagation();
    });
    
    // Dynamic sliders value display
    elements.paramTemp.addEventListener('input', (e) => {
        elements.valTemp.textContent = e.target.value;
    });
    elements.paramTopp.addEventListener('input', (e) => {
        elements.valTopp.textContent = e.target.value;
    });
    
    // Click outside popover to close it
    document.addEventListener('click', () => {
        elements.hyperparametersPanel.classList.add('hidden');
    });
    
    // Trigger Prediction Forecast button click
    elements.runPredictionBtn.addEventListener('click', () => {
        if (elements.enhancedPredictionToggle && elements.enhancedPredictionToggle.checked) {
            handleRunPredictionEnhanced();
        } else {
            handleRunPrediction();
        }
    });
    
    // Sync Sheet button click
    elements.syncSheetBtn.addEventListener('click', handleSyncGoogleSheet);

    // Interactive chart controls listeners (redrawing chart instantly on change)
    elements.chartTypeSelect.addEventListener('change', () => updateChart());
    elements.chartPeriodSelect.addEventListener('change', () => updateChart());
}

// --- Autocomplete helper ---
function setupAutoComplete(inputElement, suggestionsElement, onSelectCallback) {
    let debounceTimer;
    
    inputElement.addEventListener('input', (e) => {
        const query = e.target.value.trim();
        clearTimeout(debounceTimer);
        
        if (query.length < 2) {
            suggestionsElement.classList.add('hidden');
            return;
        }
        
        debounceTimer = setTimeout(() => {
            fetch(`/api/search?q=${encodeURIComponent(query)}`)
                .then(res => res.json())
                .then(data => {
                    const suggestions = data.suggestions || [];
                    if (suggestions.length === 0) {
                        suggestionsElement.classList.add('hidden');
                        return;
                    }
                    
                    suggestionsElement.innerHTML = '';
                    suggestions.forEach(item => {
                        const div = document.createElement('div');
                        div.className = 'suggestion-item';
                        div.innerHTML = `
                            <div class="item-details">
                                <span class="item-name">${item.name}</span>
                                <span class="item-symbol">${item.symbol}</span>
                            </div>
                            <span class="item-exchange">${item.exchange}</span>
                        `;
                        div.addEventListener('mousedown', (e) => {
                            // Prevent loss of focus or early closing of suggestions container
                            e.preventDefault();
                            onSelectCallback(item);
                            suggestionsElement.classList.add('hidden');
                        });
                        suggestionsElement.appendChild(div);
                    });
                    
                    suggestionsElement.classList.remove('hidden');
                })
                .catch(err => {
                    console.error('Autocomplete search failed:', err);
                });
        }, 250);
    });
    
    // Hide suggestions when clicking elsewhere
    document.addEventListener('click', (e) => {
        if (!inputElement.contains(e.target) && !suggestionsElement.contains(e.target)) {
            suggestionsElement.classList.add('hidden');
        }
    });
}

// --- Fetching Model Status ---
function fetchModelStatus() {
    fetch('/api/model-status')
        .then(res => res.json())
        .then(data => {
            modelStatus = data;
            updateModelStatusUI();
        })
        .catch(err => console.error('Failed to query model status:', err));
}

function updateModelStatusUI() {
    if (modelStatus.loaded) {
        elements.statusDot.className = 'status-indicator-dot positive';
        const modelName = elements.selectModel.querySelector(`option[value="${modelStatus.model_key}"]`)?.textContent.split('(')[0].trim() || modelStatus.model_key;
        elements.statusText.textContent = `Active: ${modelName} (${modelStatus.device.toUpperCase()})`;
    } else {
        elements.statusDot.className = 'status-indicator-dot negative';
        elements.statusText.textContent = 'Model: Unloaded';
    }
}

// --- Modal Settings Handlers ---
function openSettingsModal() {
    if (isModelLoading) return;
    elements.modalLoadingBar.classList.add('hidden');
    elements.settingsModal.classList.remove('hidden');
    
    // Load spreadsheet ID
    elements.inputSpreadsheetId.value = localStorage.getItem('google_spreadsheet_id') || '';
    
    // Prep values
    if (modelStatus.loaded) {
        elements.selectModel.value = modelStatus.model_key;
        elements.selectDevice.value = modelStatus.device;
    }
}

function closeSettingsModal() {
    if (isModelLoading) return;
    if (elements.inputSpreadsheetId) {
        let val = elements.inputSpreadsheetId.value.trim();
        const match = val.match(/\/d\/([a-zA-Z0-9-_]+)/);
        if (match) {
            val = match[1];
            elements.inputSpreadsheetId.value = val;
        }
        localStorage.setItem('google_spreadsheet_id', val);
    }
    elements.settingsModal.classList.add('hidden');
}

function handleLoadModel() {
    if (elements.inputSpreadsheetId) {
        let val = elements.inputSpreadsheetId.value.trim();
        const match = val.match(/\/d\/([a-zA-Z0-9-_]+)/);
        if (match) {
            val = match[1];
            elements.inputSpreadsheetId.value = val;
        }
        localStorage.setItem('google_spreadsheet_id', val);
    }
    const model_key = elements.selectModel.value;
    const device = elements.selectDevice.value;
    
    isModelLoading = true;
    elements.loadModelSubmitBtn.disabled = true;
    elements.modalLoadingBar.classList.remove('hidden');
    
    // Animate mock progress bar
    let pct = 0;
    elements.loadingPct.textContent = 'Initializing download...';
    elements.loadingPctBar.style.width = '0%';
    
    const progressInterval = setInterval(() => {
        if (pct < 85) {
            pct += Math.floor(Math.random() * 8) + 2;
            elements.loadingPct.textContent = `Downloading model weights... ${pct}%`;
            elements.loadingPctBar.style.width = `${pct}%`;
        }
    }, 400);

    fetch('/api/load-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key, device })
    })
    .then(res => res.json())
    .then(data => {
        clearInterval(progressInterval);
        if (data.error) {
            showToast(data.error, 'error');
            elements.modalLoadingBar.classList.add('hidden');
        } else {
            elements.loadingPct.textContent = 'Loading to device... 100%';
            elements.loadingPctBar.style.width = '100%';
            setTimeout(() => {
                showToast(data.message, 'success');
                modelStatus.loaded = true;
                modelStatus.model_key = model_key;
                modelStatus.device = device;
                updateModelStatusUI();
                closeSettingsModal();
            }, 300);
        }
    })
    .catch(err => {
        clearInterval(progressInterval);
        showToast('Connection error: Failed to download model weights.', 'error');
        elements.modalLoadingBar.classList.add('hidden');
        console.error(err);
    })
    .finally(() => {
        isModelLoading = false;
        elements.loadModelSubmitBtn.disabled = false;
    });
}

// --- Portfolio Management ---
function fetchPortfolio() {
    fetch('/api/portfolio')
        .then(res => res.json())
        .then(data => {
            renderPortfolioSummary(data.summary);
            renderHoldingsList(data.holdings);
        })
        .catch(err => console.error('Failed to load portfolio details:', err));
}

function renderPortfolioSummary(summary) {
    elements.portTotalValue.textContent = `₹${formatCurrency(summary.total_value)}`;
    elements.portTotalCost.textContent = `₹${formatCurrency(summary.total_cost)}`;
    
    // Total P&L
    const pnlSign = summary.total_pnl >= 0 ? '+' : '';
    elements.portTotalPnl.textContent = `₹${pnlSign}${formatCurrency(summary.total_pnl)} (${pnlSign}${summary.total_pnl_percent.toFixed(2)}%)`;
    elements.portTotalPnl.className = `metric-value ${summary.total_pnl >= 0 ? 'positive' : 'negative'}`;
    
    // Today P&L
    const todaySign = summary.today_pnl >= 0 ? '+' : '';
    elements.portTodayPnl.textContent = `₹${todaySign}${formatCurrency(summary.today_pnl)} (${todaySign}${summary.today_pnl_percent.toFixed(2)}%)`;
    elements.portTodayPnl.className = `metric-value ${summary.today_pnl >= 0 ? 'positive' : 'negative'}`;
}

function renderHoldingsList(holdings) {
    elements.holdingsCount.textContent = `${holdings.length} Stock${holdings.length === 1 ? '' : 's'}`;
    
    if (holdings.length === 0) {
        elements.holdingsListContainer.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-folder-open empty-icon"></i>
                <p>No holdings added yet.</p>
                <p class="sub">Search and add transaction above to begin tracking.</p>
            </div>
        `;
        return;
    }
    
    elements.holdingsListContainer.innerHTML = '';
    holdings.forEach(item => {
        const div = document.createElement('div');
        div.className = `holding-item ${selectedSymbol === item.symbol ? 'selected' : ''}`;
        div.dataset.symbol = item.symbol;
        
        const pnlSign = item.pnl >= 0 ? '+' : '';
        const daySign = item.change_percent >= 0 ? '+' : '';
        
        // Build the top main row
        let innerHtml = `
            <div class="holding-main">
                <div class="holding-info">
                    <span class="h-symbol">${item.symbol}</span>
                    <span class="h-name">${item.name}</span>
                    <span class="h-shares">${item.shares.toFixed(2)} Shares @ Avg ₹${formatCurrency(item.purchase_price)}</span>
                </div>
                <div class="holding-vals">
                    <span class="h-price">₹${formatCurrency(item.current_price * item.shares)}</span>
                    <span class="h-pnl ${item.pnl >= 0 ? 'positive' : 'negative'}">${pnlSign}${item.pnl_percent.toFixed(1)}% P&L</span>
                    <span class="h-pnl ${item.change_percent >= 0 ? 'positive' : 'negative'}">${daySign}${item.change_percent.toFixed(1)}% Today</span>
                </div>
                <button class="holding-delete-btn" title="Remove Stock" data-symbol="${item.symbol}">
                    <i class="fa-solid fa-trash-can"></i>
                </button>
            </div>
        `;
        
        // Append the audit badge based on state
        if (item.prediction_status === 'VERIFIED') {
            const matchClass = item.directional_match ? 'verified-success' : 'verified-fail';
            const matchIcon = item.directional_match ? '<i class="fa-solid fa-circle-check"></i>' : '<i class="fa-solid fa-circle-xmark"></i>';
            const matchLabel = item.directional_match ? 'ML Hit' : 'ML Miss';
            const pctErr = item.percentage_error ? item.percentage_error.toFixed(1) : '0.0';
            
            innerHtml += `
                <div class="audit-badge ${matchClass}">
                    <span class="badge-title">${matchIcon} ${matchLabel} (${item.prediction_horizon_days}D Horizon)</span>
                    <span class="badge-details">Target: ₹${formatCurrency(item.predicted_target_price)} | Actual: ₹${formatCurrency(item.actual_target_price)} (Error: ${pctErr}%) on ${item.evaluation_date}</span>
                </div>
            `;
        } else if (item.prediction_status === 'ACTIVE') {
            const remaining = item.trading_days_remaining;
            const progressText = remaining > 0 ? `${remaining} trading days left` : `Verifying...`;
            const directionIcon = item.predicted_direction === 'UP' ? '<i class="fa-solid fa-arrow-trend-up"></i>' : '<i class="fa-solid fa-arrow-trend-down"></i>';
            
            // Calculate tracking difference
            const diff = ((item.current_price - item.predicted_target_price) / item.predicted_target_price) * 100;
            const diffSign = diff >= 0 ? '+' : '';
            const diffText = ` (Track: ${diffSign}${diff.toFixed(1)}%)`;
            
            innerHtml += `
                <div class="audit-badge active">
                    <span class="badge-title">⏳ ACTIVE AUDIT (${item.prediction_horizon_days}D Horizon)</span>
                    <span class="badge-details">Target: ${directionIcon} ₹${formatCurrency(item.predicted_target_price)} | Status: ${progressText}${diffText}</span>
                </div>
            `;
        } else if (item.prediction_status === 'NONE') {
            innerHtml += `
                <div class="audit-badge none">
                    <span class="badge-title"><i class="fa-solid fa-circle-minus"></i> No Forecast Logged</span>
                    <span class="badge-details">No forecast was active when this stock was added.</span>
                </div>
            `;
        }
        
        div.innerHTML = innerHtml;
        
        // Add row clicks (load stock)
        div.addEventListener('click', (e) => {
            // If clicking the delete icon, don't select
            if (e.target.closest('.holding-delete-btn')) return;
            selectStock(item.symbol);
        });
        
        // Delete button click
        div.querySelector('.holding-delete-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            handleDeleteHolding(item.symbol);
        });
        
        elements.holdingsListContainer.appendChild(div);
    });
}

function handleAddHolding(e) {
    e.preventDefault();
    const symbol = elements.stockSearchInput.value.trim();
    const shares = parseFloat(elements.holdingShares.value);
    const purchase_price = parseFloat(elements.holdingPrice.value);
    
    if (!symbol || isNaN(shares) || isNaN(purchase_price)) return;
    
    elements.addHoldingBtn.disabled = true;
    elements.addHoldingBtn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Adding...`;
    
    // Normalize symbols to see if the stock being added has an active forecast displayed in the right panel
    const normInput = symbol.split('.')[0].toUpperCase();
    const normSelected = selectedSymbol ? selectedSymbol.split('.')[0].toUpperCase() : '';
    
    const payload = { symbol, shares, purchase_price };
    
    if (normInput === normSelected && currentForecast && currentForecast.length > 0) {
        const horizon = currentForecast.length;
        const targetPrice = currentForecast[horizon - 1].close;
        const direction = targetPrice >= purchase_price ? 'UP' : 'DOWN';
        const vector = currentForecast.map(f => f.close);
        
        payload.predicted_target_price = targetPrice;
        payload.prediction_horizon_days = horizon;
        payload.predicted_direction = direction;
        payload.prediction_vector = vector;
    }
    
    fetch('/api/portfolio', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            showToast(data.message, 'success');
            elements.addHoldingForm.reset();
            fetchPortfolio();
        } else {
            showToast(data.error || 'Failed to add holding', 'error');
        }
    })
    .catch(err => {
        console.error(err);
        showToast('Connection error: Failed to register transaction.', 'error');
    })
    .finally(() => {
        elements.addHoldingBtn.disabled = false;
        elements.addHoldingBtn.innerHTML = `<i class="fa-solid fa-folder-plus"></i> Add Transaction`;
    });
}

function handleDeleteHolding(symbol) {
    // Perform deletion directly to avoid blocking dialog prompts
    fetch(`/api/portfolio?symbol=${symbol}`, { method: 'DELETE' })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                showToast(data.message, 'success');
                // If currently viewing deleted stock, hide sections
                if (selectedSymbol === symbol) {
                    selectedSymbol = null;
                    elements.stockDetailsSection.classList.add('hidden');
                    elements.chartActionSection.classList.add('hidden');
                    elements.analysisSection.classList.add('hidden');
                    elements.chartWrapper.innerHTML = `
                        <div class="chart-placeholder flex-column align-center text-center gap-3">
                            <i class="fa-solid fa-chart-line placeholder-icon"></i>
                            <div>
                                <h3>No Stock Selected</h3>
                                <p class="muted">Search for a ticker or click a portfolio holding to generate interactive visualizations and ML-driven forecasts.</p>
                            </div>
                        </div>
                    `;
                }
                fetchPortfolio();
            } else {
                showToast(data.error || 'Failed to delete holding', 'error');
            }
        })
        .catch(err => {
            console.error(err);
            showToast('Connection error: Failed to delete transaction.', 'error');
        });
}

// --- Ticker Details & Chart loading ---
function selectStock(symbol) {
    selectedSymbol = symbol;
    currentHistory = [];
    currentForecast = [];
    
    // Highlight item in list
    document.querySelectorAll('.holding-item').forEach(item => {
        if (item.dataset.symbol === symbol) {
            item.classList.add('selected');
        } else {
            item.classList.remove('selected');
        }
    });
    
    // UI states
    elements.stockDetailsSection.classList.add('hidden');
    elements.chartActionSection.classList.add('hidden');
    elements.analysisSection.classList.add('hidden');
    
    // Show chart loader
    elements.loaderStatus.textContent = 'Loading stock metrics...';
    elements.chartLoader.classList.remove('hidden');
    
    // Fetch details
    fetch(`/api/stock-details?symbol=${symbol}`)
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                showToast(data.error, 'error');
                elements.chartLoader.classList.add('hidden');
                return;
            }
            
            // Populating Identity
            elements.stockDetailName.textContent = data.name;
            elements.stockDetailSymbol.textContent = data.symbol;
            elements.stockDetailExchange.textContent = data.symbol.endsWith('.NS') ? 'NSE' : (data.symbol.endsWith('.BO') ? 'BSE' : 'EQ');
            elements.stockDetailPrice.textContent = `₹${formatCurrency(data.current_price)}`;
            
            const sign = data.change >= 0 ? '+' : '';
            elements.stockDetailChange.textContent = `${sign}₹${formatCurrency(data.change)} (${sign}${data.change_percent.toFixed(2)}%)`;
            elements.stockDetailChange.className = `live-price-change ${data.change >= 0 ? 'positive' : 'negative'}`;
            
            // Stats Grid
            elements.metricOpen.textContent = `₹${formatCurrency(data.open_price)}`;
            elements.metricRange.textContent = `₹${formatCurrency(data.day_low)} - ₹${formatCurrency(data.day_high)}`;
            elements.metric52w.textContent = `₹${formatCurrency(data.fifty_two_week_low)} - ₹${formatCurrency(data.fifty_two_week_high)}`;
            elements.metricMcap.textContent = formatMarketCap(data.market_cap);
            elements.metricPe.textContent = typeof data.pe_ratio === 'number' ? data.pe_ratio.toFixed(1) : data.pe_ratio;
            elements.metricVolume.textContent = formatNumber(data.volume);
            elements.metricDiv.textContent = data.dividend_yield;
            elements.metricSector.textContent = data.sector;
            
            elements.stockDetailsSection.classList.remove('hidden');
            
            // Enable Actions
            elements.chartSubLabel.textContent = `Historical OHLCV for ${symbol}`;
            elements.chartActionSection.classList.remove('hidden');
            
            // Fetch clean preview chart (historical) before prediction
            drawPreviewChart(symbol);
        })
        .catch(err => {
            console.error(err);
            showToast('Failed to fetch details for stock symbol.', 'error');
            elements.chartLoader.classList.add('hidden');
        });
}

function drawPreviewChart(symbol) {
    elements.loaderStatus.textContent = 'Generating preview chart...';
    
    // Fetch historical-only data (extremely fast)
    fetch(`/api/historical-chart?symbol=${symbol}`)
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            elements.chartWrapper.innerHTML = `
                <div class="empty-state text-center">
                    <i class="fa-solid fa-circle-exclamation empty-icon" style="color: var(--negative)"></i>
                    <h3>Data Load Failed</h3>
                    <p>${data.error}</p>
                </div>
            `;
        } else {
            currentHistory = data.history || [];
            currentForecast = []; // reset forecast
            currentForecastEnhanced = []; // reset enhanced forecast
            elements.sentimentNewsSection.classList.add('hidden'); // hide news feed
            
            // Auto initialize Date filters - removed for period selector
            
            updateChart();
            
            // Hide forecasting analysis card until user clicks 'Predict Trend'
            elements.analysisSection.classList.add('hidden');
        }
    })
    .catch(err => {
        console.error(err);
        showToast('Failed to load historical chart preview.', 'error');
    })
    .finally(() => {
        elements.chartLoader.classList.add('hidden');
    });
}

function handleRunPrediction() {
    if (!selectedSymbol || isPredicting) return;
    
    isPredicting = true;
    elements.runPredictionBtn.disabled = true;
    elements.runPredictionBtn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Predicting...`;
    if (elements.enhancedPredictionToggle) elements.enhancedPredictionToggle.disabled = true;
    
    // Show progress bar and loader overlay
    elements.enhancedProgressContainer.classList.remove('hidden');
    elements.enhancedProgressBarFill.style.width = '0%';
    elements.enhancedProgressPct.textContent = '0%';
    
    elements.loaderStatus.textContent = 'Analyzing Historical Trends...';
    elements.chartLoader.classList.remove('hidden');
    
    const temperature = parseFloat(elements.paramTemp.value);
    const top_p = parseFloat(elements.paramTopp.value);
    const pred_len = parseInt(elements.paramPredLen.value);
    
    let pct = 0;
    const progressTexts = [
        { pct: 15, text: 'Analyzing Historical Trends...' },
        { pct: 55, text: 'Generating Price Projections...' },
        { pct: 85, text: 'Finalizing Projections...' }
    ];
    
    elements.loaderStatus.textContent = progressTexts[0].text;
    elements.enhancedProgressBarFill.style.width = '5%';
    elements.enhancedProgressPct.textContent = '5%';
    
    // Interpolate progress percentages smoothly
    const progressInterval = setInterval(() => {
        if (pct < 95) {
            pct += Math.floor(Math.random() * 4) + 1;
            elements.enhancedProgressBarFill.style.width = `${pct}%`;
            elements.enhancedProgressPct.textContent = `${pct}%`;
            
            // Check if we should update text description
            const currentStep = progressTexts.slice().reverse().find(s => pct >= s.pct);
            if (currentStep) {
                elements.loaderStatus.textContent = currentStep.text;
            }
        }
    }, 200);

    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            symbol: selectedSymbol, 
            temperature, 
            top_p,
            pred_len
        })
    })
    .then(res => res.json())
    .then(data => {
        clearInterval(progressInterval);
        if (data.error) {
            showToast(data.error, 'error');
            elements.enhancedProgressContainer.classList.add('hidden');
        } else {
            elements.enhancedProgressBarFill.style.width = '100%';
            elements.enhancedProgressPct.textContent = '100%';
            elements.loaderStatus.textContent = 'Rendering Charts...';
            
            setTimeout(() => {
                currentHistory = data.history || [];
                currentForecast = data.forecast || [];
                currentForecastEnhanced = []; // reset enhanced forecast
                elements.sentimentNewsSection.classList.add('hidden'); // hide news feed
                
                updateChart();
                fetchModelStatus();
                
                elements.analysisSignal.textContent = data.analysis.trend_signal;
                elements.analysisSignal.style.color = data.analysis.signal_color;
                elements.analysisSignal.style.textShadow = `0 0 15px ${data.analysis.signal_color}`;
                
                const tgtSign = data.analysis.change_percent >= 0 ? '+' : '';
                elements.analysisTarget.textContent = `₹${formatCurrency(data.analysis.target_price)} (${tgtSign}${data.analysis.change_percent.toFixed(2)}%)`;
                elements.analysisTarget.className = `stat-val ${data.analysis.change_percent >= 0 ? 'positive' : 'negative'}`;
                
                elements.analysisMax.textContent = `₹${formatCurrency(data.analysis.max_predicted)}`;
                elements.analysisMin.textContent = `₹${formatCurrency(data.analysis.min_predicted)}`;
                elements.analysisSummary.textContent = data.analysis.summary;
                
                elements.analysisSection.classList.remove('hidden');
                elements.enhancedProgressContainer.classList.add('hidden');
                showToast(`Inference completed for ${selectedSymbol}. ${pred_len}-Day trend projected.`, 'success');
            }, 400);
        }
    })
    .catch(err => {
        clearInterval(progressInterval);
        console.error(err);
        showToast('Prediction failed due to server inference error.', 'error');
        elements.enhancedProgressContainer.classList.add('hidden');
    })
    .finally(() => {
        isPredicting = false;
        elements.runPredictionBtn.disabled = false;
        elements.runPredictionBtn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Predict Trend`;
        if (elements.enhancedPredictionToggle) elements.enhancedPredictionToggle.disabled = false;
        elements.chartLoader.classList.add('hidden');
    });
}

function handleRunPredictionEnhanced() {
    if (!selectedSymbol || isPredicting) return;
    
    isPredicting = true;
    elements.runPredictionBtn.disabled = true;
    elements.runPredictionBtn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Predicting...`;
    if (elements.enhancedPredictionToggle) elements.enhancedPredictionToggle.disabled = true;
    
    // Show progress bar and loader overlay
    elements.enhancedProgressContainer.classList.remove('hidden');
    elements.enhancedProgressBarFill.style.width = '0%';
    elements.enhancedProgressPct.textContent = '0%';
    
    elements.loaderStatus.textContent = 'Initializing Sentiment-Enhanced Engine...';
    elements.chartLoader.classList.remove('hidden');
    
    const temperature = parseFloat(elements.paramTemp.value);
    const top_p = parseFloat(elements.paramTopp.value);
    const pred_len = parseInt(elements.paramPredLen.value);
    
    // Enhanced progress bar steps (0% to 100%)
    let pct = 0;
    const progressTexts = [
        { pct: 15, text: 'Analyzing Historical Trends...' },
        { pct: 40, text: 'Synthesizing Market Intelligence...' },
        { pct: 70, text: 'Generating Price Projections...' },
        { pct: 90, text: 'Finalizing Projections...' }
    ];
    
    elements.loaderStatus.textContent = progressTexts[0].text;
    elements.enhancedProgressBarFill.style.width = '5%';
    elements.enhancedProgressPct.textContent = '5%';
    
    // Interpolate progress percentages smoothly
    const progressInterval = setInterval(() => {
        if (pct < 95) {
            pct += Math.floor(Math.random() * 4) + 1;
            elements.enhancedProgressBarFill.style.width = `${pct}%`;
            elements.enhancedProgressPct.textContent = `${pct}%`;
            
            // Check if we should update text description
            const currentStep = progressTexts.slice().reverse().find(s => pct >= s.pct);
            if (currentStep) {
                elements.loaderStatus.textContent = currentStep.text;
            }
        }
    }, 200);

    fetch('/api/predict-enhanced', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            symbol: selectedSymbol, 
            temperature, 
            top_p,
            pred_len
        })
    })
    .then(res => res.json())
    .then(data => {
        clearInterval(progressInterval);
        if (data.error) {
            showToast(data.error, 'error');
            elements.enhancedProgressContainer.classList.add('hidden');
        } else {
            elements.enhancedProgressBarFill.style.width = '100%';
            elements.enhancedProgressPct.textContent = '100%';
            elements.loaderStatus.textContent = 'Rendering Enhanced Overlays...';
            
            setTimeout(() => {
                currentHistory = data.history || [];
                currentForecast = data.forecast || [];
                currentForecastEnhanced = data.forecast_enhanced || [];
                
                // Redraw Plotly chart (which will now overlay BOTH lines)
                updateChart();
                
                // Update analysis card
                elements.analysisSignal.textContent = data.analysis.trend_signal;
                elements.analysisSignal.style.color = data.analysis.signal_color;
                elements.analysisSignal.style.textShadow = `0 0 15px ${data.analysis.signal_color}`;
                
                const tgtSign = data.analysis.change_percent >= 0 ? '+' : '';
                elements.analysisTarget.textContent = `₹${formatCurrency(data.analysis.target_price)} (${tgtSign}${data.analysis.change_percent.toFixed(2)}%)`;
                elements.analysisTarget.className = `stat-val ${data.analysis.change_percent >= 0 ? 'positive' : 'negative'}`;
                
                elements.analysisMax.textContent = `₹${formatCurrency(data.analysis.max_predicted)}`;
                elements.analysisMin.textContent = `₹${formatCurrency(data.analysis.min_predicted)}`;
                elements.analysisSummary.textContent = data.analysis.summary;
                elements.analysisSection.classList.remove('hidden');
                
                // Render live news articles
                renderSentimentNewsFeed(data.net_sentiment, data.news_feed);
                
                elements.enhancedProgressContainer.classList.add('hidden');
                showToast(`Hybrid forecast compiled! Dotted purple overlay applied.`, 'success');
            }, 400);
        }
    })
    .catch(err => {
        clearInterval(progressInterval);
        console.error(err);
        showToast('Enhanced prediction failed due to server error.', 'error');
        elements.enhancedProgressContainer.classList.add('hidden');
    })
    .finally(() => {
        isPredicting = false;
        elements.runPredictionBtn.disabled = false;
        elements.runPredictionBtn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Predict Trend`;
        if (elements.enhancedPredictionToggle) elements.enhancedPredictionToggle.disabled = false;
        elements.chartLoader.classList.add('hidden');
    });
}

function handleSyncGoogleSheet() {
    if (!selectedSymbol) {
        showToast('Please select a stock from your portfolio first.', 'warning');
        return;
    }
    
    let spreadsheetId = localStorage.getItem('google_spreadsheet_id') || '';
    spreadsheetId = spreadsheetId.trim();
    const match = spreadsheetId.match(/\/d\/([a-zA-Z0-9-_]+)/);
    if (match) {
        spreadsheetId = match[1];
    }
    
    if (!spreadsheetId) {
        showToast('Google Sheet ID not configured. Please open Settings to set it.', 'warning');
        openSettingsModal();
        setTimeout(() => {
            if (elements.inputSpreadsheetId) {
                elements.inputSpreadsheetId.focus();
                elements.inputSpreadsheetId.style.border = '1px solid var(--primary-color)';
            }
        }, 300);
        return;
    }
    
    const enhanced = elements.enhancedPredictionToggle ? elements.enhancedPredictionToggle.checked : false;
    const forecastToSync = enhanced ? currentForecastEnhanced : currentForecast;
    
    if (!forecastToSync || forecastToSync.length === 0) {
        showToast('No active predictions found. Please click "Predict Trend" first.', 'warning');
        return;
    }
    
    elements.syncSheetBtn.disabled = true;
    const originalHTML = elements.syncSheetBtn.innerHTML;
    elements.syncSheetBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Syncing...`;
    
    fetch('/api/sync-to-google-sheet', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            symbol: selectedSymbol,
            spreadsheet_id: spreadsheetId,
            enhanced: enhanced,
            client_predictions: forecastToSync
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            if (data.error === 'service_account_missing') {
                showToast('Google Service Account config missing in server root.', 'error');
                alert(data.message || 'service_account.json is required.');
            } else {
                showToast(`Sync failed: ${data.error}`, 'error');
            }
        } else {
            showToast(data.message || 'Sync successful!', 'success');
        }
    })
    .catch(err => {
        console.error(err);
        showToast('Sync request failed. Check server connection.', 'error');
    })
    .finally(() => {
        elements.syncSheetBtn.disabled = false;
        elements.syncSheetBtn.innerHTML = originalHTML;
    });
}

function renderSentimentNewsFeed(netSentiment, newsArticles) {
    elements.netSentimentVal.textContent = netSentiment.toFixed(2);
    
    // Style compound color
    if (netSentiment >= 0.05) {
        elements.netSentimentVal.className = 'positive';
        elements.netSentimentVal.style.textShadow = '0 0 8px var(--positive)';
    } else if (netSentiment <= -0.05) {
        elements.netSentimentVal.className = 'negative';
        elements.netSentimentVal.style.textShadow = '0 0 8px var(--negative)';
    } else {
        elements.netSentimentVal.className = 'warning';
        elements.netSentimentVal.style.textShadow = '0 0 8px var(--warning)';
    }
    
    if (!newsArticles || newsArticles.length === 0) {
        elements.newsListContainer.innerHTML = `
            <div class="empty-state text-center py-3">
                <i class="fa-solid fa-newspaper empty-icon" style="font-size: 1.5rem;"></i>
                <p>No recent news found for this ticker.</p>
            </div>
        `;
        elements.sentimentNewsSection.classList.remove('hidden');
        return;
    }
    
    elements.newsListContainer.innerHTML = '';
    newsArticles.forEach(art => {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'news-item glass-panel';
        
        let sentimentClass = 'neutral';
        let sentimentBadge = 'NEUTRAL';
        if (art.sentiment_label === 'POSITIVE') {
            sentimentClass = 'positive';
            sentimentBadge = 'POS';
        } else if (art.sentiment_label === 'NEGATIVE') {
            sentimentClass = 'negative';
            sentimentBadge = 'NEG';
        }
        
        itemDiv.innerHTML = `
            <div class="news-item-header">
                <span class="news-source">${art.source}</span>
                <span class="news-sentiment-badge ${sentimentClass}">${sentimentBadge} (${art.sentiment_score.toFixed(2)})</span>
            </div>
            <a href="${art.link}" target="_blank" class="news-title-link">${art.title}</a>
        `;
        elements.newsListContainer.appendChild(itemDiv);
    });
    
    elements.sentimentNewsSection.classList.remove('hidden');
}

// --- Dynamic Plotly Chart Renderer ---
function updateChart() {
    if (currentHistory.length === 0) return;
    
    const chartType = elements.chartTypeSelect.value;
    const period = elements.chartPeriodSelect.value;
    
    // Calculate cutoff date from the end of history
    let cutoffDate = null;
    const lastHistPoint = currentHistory[currentHistory.length - 1];
    
    if (period !== '5y') {
        const lastDate = new Date(lastHistPoint.date);
        if (period === '1d') {
            cutoffDate = lastHistPoint.date;
        } else if (period === '1w') {
            lastDate.setDate(lastDate.getDate() - 7);
            cutoffDate = lastDate.toISOString().split('T')[0];
        } else if (period === '1m') {
            lastDate.setMonth(lastDate.getMonth() - 1);
            cutoffDate = lastDate.toISOString().split('T')[0];
        } else if (period === '1y') {
            lastDate.setFullYear(lastDate.getFullYear() - 1);
            cutoffDate = lastDate.toISOString().split('T')[0];
        } else if (period === '3y') {
            lastDate.setFullYear(lastDate.getFullYear() - 3);
            cutoffDate = lastDate.toISOString().split('T')[0];
        }
    }
    
    // Filter history by period
    const filteredHistory = currentHistory.filter(item => {
        if (period === '1d') {
            return item.date === cutoffDate;
        }
        return !cutoffDate || item.date >= cutoffDate;
    });
    
    // We always show all forecast points since they extend from the history
    const filteredForecast = currentForecast;
    
    if (filteredHistory.length === 0 && filteredForecast.length === 0) {
        elements.chartWrapper.innerHTML = `
            <div class="empty-state text-center">
                <i class="fa-solid fa-calendar-xmark empty-icon"></i>
                <h3>No Data in Selected Period</h3>
                <p class="muted">Try choosing a wider timeframe dropdown selection.</p>
            </div>
        `;
        return;
    }
    
    const traces = [];
    
    // 1. Plot Historical Data
    if (filteredHistory.length > 0) {
        const dates = filteredHistory.map(d => d.date);
        
        if (chartType === 'candlestick') {
            traces.push({
                x: dates,
                open: filteredHistory.map(d => d.open),
                high: filteredHistory.map(d => d.high),
                low: filteredHistory.map(d => d.low),
                close: filteredHistory.map(d => d.close),
                type: 'candlestick',
                name: 'Historical Data',
                increasing: { line: { color: '#26A69A' } },
                decreasing: { line: { color: '#EF5350' } }
            });
        } else if (chartType === 'line') {
            traces.push({
                x: dates,
                y: filteredHistory.map(d => d.close),
                type: 'scatter',
                mode: 'lines',
                name: 'Historical Close',
                line: { color: '#29B6F6', width: 2.5 }
            });
        } else if (chartType === 'bar') {
            traces.push({
                x: dates,
                y: filteredHistory.map(d => d.close),
                type: 'bar',
                name: 'Historical Close',
                marker: {
                    color: filteredHistory.map((d, i) => {
                        if (i === 0) return '#26A69A';
                        return d.close >= filteredHistory[i - 1].close ? '#26A69A' : '#EF5350';
                    })
                }
            });
        }
    }
    
    // 2. Plot Forecasted Data (if available)
    if (filteredForecast.length > 0) {
        const dates = filteredForecast.map(d => d.date);
        
        if (chartType === 'candlestick') {
            traces.push({
                x: dates,
                open: filteredForecast.map(d => d.open),
                high: filteredForecast.map(d => d.high),
                low: filteredForecast.map(d => d.low),
                close: filteredForecast.map(d => d.close),
                type: 'candlestick',
                name: 'Forecasted',
                increasing: { line: { color: '#66BB6A' } },
                decreasing: { line: { color: '#FF7043' } },
                opacity: 0.8
            });
        } else if (chartType === 'line') {
            traces.push({
                x: dates,
                y: filteredForecast.map(d => d.close),
                type: 'scatter',
                mode: 'lines',
                name: 'Forecasted Close',
                line: { color: '#66BB6A', width: 2.5, dash: 'dash' }
            });
        } else if (chartType === 'bar') {
            traces.push({
                x: dates,
                y: filteredForecast.map(d => d.close),
                type: 'bar',
                name: 'Forecasted Close',
                marker: {
                    color: '#66BB6A',
                    opacity: 0.8
                }
            });
        }
        
        // 3. Connection Trace (Extended dashed line joining history and forecast)
        if (filteredHistory.length > 0) {
            const lastHist = filteredHistory[filteredHistory.length - 1];
            const firstPred = filteredForecast[0];
            traces.push({
                x: [lastHist.date, firstPred.date],
                y: [lastHist.close, firstPred.open],
                type: 'scatter',
                mode: 'lines',
                line: { color: '#888888', width: 2, dash: 'dot' },
                name: 'Forecast Connection',
                showlegend: false
            });
        }
        
        // 4. Forecast Trend Line
        traces.push({
            x: dates,
            y: filteredForecast.map(d => d.close),
            type: 'scatter',
            mode: 'lines',
            line: { color: '#00e5ff', width: 1.5 },
            name: 'Forecast Trend',
            showlegend: true
        });
    }
    
    // 2.5 Plot Enhanced Forecast (if available)
    if (currentForecastEnhanced.length > 0) {
        const datesEnhanced = currentForecastEnhanced.map(d => d.date);
        
        // Plot Enhanced Trend Line
        traces.push({
            x: datesEnhanced,
            y: currentForecastEnhanced.map(d => d.close),
            type: 'scatter',
            mode: 'lines',
            line: { color: '#c5a059', width: 2, dash: 'dot' },
            name: 'Enhanced Trend (News)',
            showlegend: true
        });
        
        // Connection trace for enhanced prediction
        if (filteredHistory.length > 0) {
            const lastHist = filteredHistory[filteredHistory.length - 1];
            const firstEnhanced = currentForecastEnhanced[0];
            traces.push({
                x: [lastHist.date, firstEnhanced.date],
                y: [lastHist.close, firstEnhanced.open || firstEnhanced.close],
                type: 'scatter',
                mode: 'lines',
                line: { color: '#c5a059', width: 1.5, dash: 'dot' },
                name: 'Enhanced Connection',
                showlegend: false
            });
        }
    }
    
    // Layout configurations
    const layout = {
        template: 'plotly_dark',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            title: {
                text: 'Date',
                font: {
                    color: '#b0b8c4',
                    size: 11
                }
            },
            tickfont: {
                color: '#8492a6',
                size: 10
            },
            gridcolor: 'rgba(255, 255, 255, 0.05)',
            type: 'date',
            rangeslider: { visible: false }
        },
        yaxis: {
            title: {
                text: 'Stock Price (INR)',
                font: {
                    color: '#b0b8c4',
                    size: 11
                }
            },
            tickfont: {
                color: '#8492a6',
                size: 10
            },
            gridcolor: 'rgba(255, 255, 255, 0.05)',
            autorange: true,
            fixedrange: false
        },
        height: 500,
        margin: { l: 55, r: 15, t: 15, b: 45 },
        legend: {
            orientation: "h",
            yanchor: "bottom",
            y: 1.02,
            xanchor: "right",
            x: 1
        },
        grid: { rows: 1, columns: 1 }
    };
    
    elements.chartWrapper.innerHTML = '';
    Plotly.newPlot(elements.chartWrapper, traces, layout, { responsive: true, displayModeBar: false });
}

// --- Formatter & Toast Helpers ---
function formatCurrency(val) {
    if (isNaN(val)) return '0.00';
    // Format Indian style currency (comma formats)
    const x = val.toFixed(2).split('.');
    let x1 = x[0];
    const x2 = x.length > 1 ? '.' + x[1] : '';
    
    // Indian numbering format (e.g. 12,34,567.89)
    let lastThree = x1.substring(x1.length - 3);
    let otherNumbers = x1.substring(0, x1.length - 3);
    if (otherNumbers !== '') {
        lastThree = ',' + lastThree;
    }
    const res = otherNumbers.replace(/\B(?=(\d{2})+(?!\d))/g, ",") + lastThree + x2;
    return res;
}

function formatNumber(num) {
    if (isNaN(num) || num === null) return 'N/A';
    if (num >= 1.0e9) return (num / 1.0e9).toFixed(2) + 'B';
    if (num >= 1.0e6) return (num / 1.0e6).toFixed(2) + 'M';
    if (num >= 1.0e3) return (num / 1.0e3).toFixed(1) + 'K';
    return num.toString();
}

function formatMarketCap(mcap) {
    if (isNaN(mcap) || mcap === null || mcap === 'N/A' || mcap === 0) return 'N/A';
    // Format in INR style: Lakhs, Crores, or Trillions (L, Cr, T)
    if (mcap >= 1.0e12) return `₹${(mcap / 1.0e12).toFixed(2)}T`;
    if (mcap >= 1.0e7) return `₹${(mcap / 1.0e7).toFixed(2)}Cr`;
    if (mcap >= 1.0e5) return `₹${(mcap / 1.0e5).toFixed(2)}L`;
    return `₹${formatCurrency(mcap)}`;
}

// Show Toast alert
function showToast(message, type = 'info') {
    elements.toastText.textContent = message;
    
    if (type === 'success') {
        elements.toastMessage.className = 'toast-notification success';
        elements.toastIcon.className = 'fa-solid fa-circle-check toast-icon';
    } else if (type === 'error') {
        elements.toastMessage.className = 'toast-notification error';
        elements.toastIcon.className = 'fa-solid fa-circle-exclamation toast-icon';
    } else {
        elements.toastMessage.className = 'toast-notification';
        elements.toastIcon.className = 'fa-solid fa-circle-info toast-icon';
    }
    
    elements.toastMessage.classList.remove('hidden');
    
    // Auto fadeout after 4 seconds
    setTimeout(() => {
        elements.toastMessage.classList.add('hidden');
    }, 4000);
}
