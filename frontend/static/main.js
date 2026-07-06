document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const uploadForm = document.getElementById('upload-form');
    const parametersForm = document.getElementById('parameters-form');
    const datasetFileInput = document.getElementById('dataset-file');
    const dropZone = document.getElementById('drop-zone');
    const fileNameLabel = document.getElementById('file-name-label');
    const statusBadge = document.getElementById('status-badge');
    
    const uploadPanel = document.getElementById('upload-panel');
    const parametersPanel = document.getElementById('parameters-panel');
    const welcomePanel = document.getElementById('welcome-panel');
    const loaderPanel = document.getElementById('loader-panel');
    const loaderStatus = document.getElementById('loader-status');
    const resultsContainer = document.getElementById('results-container');
    const samplingInfoMessage = document.getElementById('sampling-info-message');
    const samplingInfoText = document.getElementById('sampling-info-text');
    
    const targetMinInput = document.getElementById('target-min');
    const targetMaxInput = document.getElementById('target-max');
    const dynamicFeaturesList = document.getElementById('dynamic-features-list');
    
    // Metrics
    const valTrainR2 = document.getElementById('val-train-r2');
    const valTestR2 = document.getElementById('val-test-r2');
    const valQueryPrediction = document.getElementById('val-query-prediction');
    const targetThresholdInput = document.getElementById('target-threshold');

    // Sync Min target to threshold value
    targetThresholdInput.addEventListener('input', () => {
        targetMinInput.value = targetThresholdInput.value;
    });
    
    // Plots
    const imgShapBeeswarm = document.getElementById('img-shap-beeswarm');
    const imgShapImportance = document.getElementById('img-shap-importance');
    const imgShapWaterfall = document.getElementById('img-shap-waterfall');
    const imgTrainFit = document.getElementById('img-train-fit');
    const imgCorrelationHeatmap = document.getElementById('img-correlation-heatmap');
    const correlationListContainer = document.getElementById('correlation-list-container');
    
    // Table
    const diceTableHeader = document.getElementById('dice-table-header');
    const diceTableBody = document.getElementById('dice-table-body');

    let uploadedDatasetInfo = null;
    let currentQueryInstance = {};

    // File Input Drag and Drop styling
    datasetFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            fileNameLabel.textContent = e.target.files[0].name;
            fileNameLabel.style.color = '#7c4dff';
        }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            datasetFileInput.files = files;
            fileNameLabel.textContent = files[0].name;
            fileNameLabel.style.color = '#7c4dff';
        }
    });

    // Step 1: Upload and Validate
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const file = datasetFileInput.files[0];
        const targetColumn = document.getElementById('target_column').value.trim();
        
        if (!file || !targetColumn) return;

        const formData = new FormData();
        formData.append('file', file);
        formData.append('target_column', targetColumn);

        // Show uploading status
        const btnUpload = document.getElementById('btn-upload');
        const origBtnText = btnUpload.innerHTML;
        btnUpload.disabled = true;
        btnUpload.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Validating...';

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (!response.ok || !data.success) {
                alert(`Validation failed: ${data.error || 'Unknown error'}`);
                return;
            }

            // Success! Store dataset metadata
            uploadedDatasetInfo = data;
            
            // Update Header Status Badge
            statusBadge.className = 'badge badge-active';
            statusBadge.innerHTML = `<span class="status-dot"></span> Dataset: ${file.name}`;

            // Set Target Range Inputs
            targetMinInput.value = data.target_default_range[0].toFixed(2);
            targetMaxInput.value = data.target_default_range[1].toFixed(2);
            targetThresholdInput.value = targetMinInput.value;

            // Unlock Step 2 Panel
            parametersPanel.classList.remove('disabled');

            // Display sampling method message
            if (data.sampling_message) {
                samplingInfoText.textContent = data.sampling_message;
                samplingInfoMessage.style.display = 'block';
            } else {
                samplingInfoMessage.style.display = 'none';
            }

            // Update Correlation Heatmap and top 10 contributions list
            const tsUpload = new Date().getTime();
            imgCorrelationHeatmap.src = `${data.correlation_heatmap}?t=${tsUpload}`;
            
            correlationListContainer.innerHTML = '';
            data.correlation_top_10.forEach((pair, idx) => {
                const item = document.createElement('div');
                item.style.cssText = "display: flex; justify-content: space-between; align-items: center; padding: 10px; border-radius: 8px; background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); font-size: 0.85rem;";
                
                const absCorr = pair.abs_correlation;
                let colorVal = "var(--text-muted)";
                if (absCorr > 0.7) colorVal = "var(--secondary)"; // cyan
                else if (absCorr > 0.4) colorVal = "var(--primary-hover)"; // blueish
                
                const signSymbol = pair.correlation > 0 ? "+" : "-";
                
                item.innerHTML = `
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-weight: 700; color: var(--text-muted); font-size: 0.8rem;">#${idx+1}</span>
                        <span style="color: var(--text-main); font-weight: 500;">${pair.feature1}</span>
                        <span style="color: var(--text-muted); font-size: 0.75rem;"><i class="fa-solid fa-arrows-left-right"></i></span>
                        <span style="color: var(--text-main); font-weight: 500;">${pair.feature2}</span>
                    </div>
                    <div style="font-weight: 700; color: ${colorVal}; font-family: monospace;">
                        ${signSymbol}${absCorr.toFixed(2)}
                    </div>
                `;
                correlationListContainer.appendChild(item);
            });

            // Generate Dynamic Inputs for each feature
            generateDynamicFeatureInputs(data.features, data.defaults, data.ranges);

            // Show results container and display default query instance
            welcomePanel.classList.add('hidden');
            resultsContainer.classList.remove('hidden');
            updateQueryDisplayBox(data.defaults);

        } catch (error) {
            alert(`Network error: ${error.message}`);
        } finally {
            btnUpload.disabled = false;
            btnUpload.innerHTML = origBtnText;
        }
    });

    // Selects a random query and automatically runs the research workflow
    async function selectRandomQueryAndRun() {
        if (!uploadedDatasetInfo) return;
        
        try {
            const response = await fetch('/api/random_query');
            const data = await response.json();
            if (!response.ok || !data.success) {
                alert(`Failed to fetch random query: ${data.error || 'Unknown error'}`);
                return;
            }

            // Populate the feature input values directly in the display box
            currentQueryInstance = data.query_instance;
            uploadedDatasetInfo.features.forEach(feature => {
                const input = document.querySelector(`.query-instance-input[data-feature="${feature}"]`);
                if (input && currentQueryInstance[feature] !== undefined) {
                    const isInt = uploadedDatasetInfo.integer_features && uploadedDatasetInfo.integer_features.includes(feature);
                    input.value = isInt ? Math.round(currentQueryInstance[feature]) : currentQueryInstance[feature].toFixed(2);
                }
            });

            // Automatically submit/run the workflow
            parametersForm.requestSubmit();

        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    }

    // Generate Dynamic Inputs for features
    function generateDynamicFeatureInputs(features, defaults, ranges) {
        dynamicFeaturesList.innerHTML = '';
        currentQueryInstance = defaults;
        
        features.forEach(feature => {
            const featureCard = document.createElement('div');
            featureCard.className = 'feature-param-card';
            
            const defVal = defaults[feature];
            const minVal = ranges[feature][0];
            const maxVal = ranges[feature][1];
            const isInt = uploadedDatasetInfo.integer_features && uploadedDatasetInfo.integer_features.includes(feature);
            
            const displayVal = isInt ? Math.round(defVal) : defVal.toFixed(2);
            const displayMin = isInt ? Math.round(minVal) : minVal.toFixed(2);
            const displayMax = isInt ? Math.round(maxVal) : maxVal.toFixed(2);
            
            featureCard.innerHTML = `
                <div class="feature-param-header" style="display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; font-weight: 600; margin-bottom: 10px;">
                    <span class="feature-name" style="color: var(--primary-hover);">${feature}</span>
                    <select class="feature-type-select" data-feature="${feature}" style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); color: var(--text-muted); border-radius: 4px; padding: 2px 6px; font-size: 0.75rem; font-family: var(--font-outfit); cursor: pointer;">
                        <option value="float" ${isInt ? '' : 'selected'}>Float</option>
                        <option value="int" ${isInt ? 'selected' : ''}>Integer</option>
                    </select>
                </div>
                <div class="feature-grid-inputs" style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; align-items: center;">
                    <div>
                        <label style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase;">Permit Min</label>
                        <input type="number" class="feature-permit-min" data-feature="${feature}" step="${isInt ? '1' : 'any'}" value="${displayMin}" required>
                    </div>
                    <div>
                        <label style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase;">Permit Max</label>
                        <input type="number" class="feature-permit-max" data-feature="${feature}" step="${isInt ? '1' : 'any'}" value="${displayMax}" required>
                    </div>
                    <label style="display: flex; align-items: center; gap: 8px; font-size: 0.8rem; grid-column: span 2; margin-top: 6px; cursor: pointer; color: var(--text-muted);">
                        <input type="checkbox" class="feature-unchanged" data-feature="${feature}"> Keep Unchanged
                    </label>
                </div>
            `;
            
            // Add change listener to select type to update input step & round value
            const selectType = featureCard.querySelector('.feature-type-select');
            const permitMin = featureCard.querySelector('.feature-permit-min');
            const permitMax = featureCard.querySelector('.feature-permit-max');
            
            selectType.addEventListener('change', () => {
                const modeInt = selectType.value === 'int';
                permitMin.step = modeInt ? '1' : 'any';
                permitMax.step = modeInt ? '1' : 'any';
                if (modeInt) {
                    permitMin.value = Math.round(parseFloat(permitMin.value));
                    permitMax.value = Math.round(parseFloat(permitMax.value));
                }
                
                // Re-render query display box so the inputs match the type selection
                updateQueryDisplayBox(currentQueryInstance, parseFloat(valQueryPrediction.textContent));
            });

            dynamicFeaturesList.appendChild(featureCard);
        });
    }

    // Step 2: Run Workflow
    parametersForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!uploadedDatasetInfo) return;

        // Collect inputs
        const targetRange = [
            parseFloat(targetMinInput.value),
            parseFloat(targetMaxInput.value)
        ];

        const totalCFs = parseInt(document.getElementById('total-cfs').value) || 3;
        const threshold = parseFloat(document.getElementById('target-threshold').value) || 0.0;

        const queryInstance = {};
        const queryInputs = document.querySelectorAll('.query-instance-input');
        queryInputs.forEach(input => {
            const feature = input.getAttribute('data-feature');
            queryInstance[feature] = parseFloat(input.value);
        });
        currentQueryInstance = queryInstance;

        const permittedRanges = {};
        const featuresToVary = [];

        const unchangedCheckboxes = document.querySelectorAll('.feature-unchanged');
        unchangedCheckboxes.forEach(cb => {
            const feature = cb.getAttribute('data-feature');
            if (!cb.checked) {
                featuresToVary.push(feature);
            }
        });

        const permitMinInputs = document.querySelectorAll('.feature-permit-min');
        const permitMaxInputs = document.querySelectorAll('.feature-permit-max');

        permitMinInputs.forEach((input, index) => {
            const feature = input.getAttribute('data-feature');
            const minVal = parseFloat(input.value);
            const maxVal = parseFloat(permitMaxInputs[index].value);
            permittedRanges[feature] = [minVal, maxVal];
        });

        const optimizationStrategy = document.getElementById('optimization-strategy').value;

        const integerFeatures = [];
        document.querySelectorAll('.feature-type-select').forEach(select => {
            if (select.value === 'int') {
                integerFeatures.push(select.getAttribute('data-feature'));
            }
        });

        // UI transitions
        welcomePanel.classList.add('hidden');
        resultsContainer.classList.add('hidden');
        loaderPanel.classList.remove('hidden');

        // Status cycles for mock model fit speed
        loaderStatus.textContent = "Training Quantile Regressor Model...";
        setTimeout(() => {
            if (!loaderPanel.classList.contains('hidden')) {
                loaderStatus.textContent = "Calculating Shapley values & rendering plots...";
            }
        }, 3000);
        setTimeout(() => {
            if (!loaderPanel.classList.contains('hidden')) {
                loaderStatus.textContent = "Generating DiCE counterfactuals...";
            }
        }, 7000);

        try {
            const response = await fetch('/run', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query_instance: queryInstance,
                    target_range: targetRange,
                    permitted_ranges: permittedRanges,
                    features_to_vary: featuresToVary,
                    total_CFs: totalCFs,
                    threshold: threshold,
                    optimization_strategy: optimizationStrategy,
                    integer_features: integerFeatures
                })
            });

            const data = await response.json();
            if (!response.ok || !data.success) {
                alert(`Execution failed: ${data.error || 'Unknown error'}`);
                loaderPanel.classList.add('hidden');
                welcomePanel.classList.remove('hidden');
                return;
            }

            // Hide loader and show results
            loaderPanel.classList.add('hidden');
            resultsContainer.classList.remove('hidden');

            // Render Metrics
            valTrainR2.textContent = data.train_r2.toFixed(2);
            const valTrainOob = document.getElementById('val-train-oob');
            if (valTrainOob) {
                valTrainOob.textContent = data.oob_score.toFixed(2);
            }
            valTestR2.textContent = data.test_r2.toFixed(2);
            valQueryPrediction.textContent = data.query_prediction.toFixed(2);

            // Update Image Plots with cache buster
            const ts = new Date().getTime();
            imgShapBeeswarm.src = `${data.shap.beeswarm}?t=${ts}`;
            imgShapImportance.src = `${data.shap.feature_importance}?t=${ts}`;
            imgShapWaterfall.src = `${data.shap.waterfall}?t=${ts}`;
            imgTrainFit.src = `${data.plots.train_fit}?t=${ts}`;

            // Render DiCE Table
            renderDiceTable(queryInstance, data.dice, data.threshold, data.dice_error);

        } catch (error) {
            alert(`Network error: ${error.message}`);
            loaderPanel.classList.add('hidden');
            welcomePanel.classList.remove('hidden');
        }
    });

    // Helper to render the query instance details above the table
    function updateQueryDisplayBox(queryInstance, predictionVal) {
        const queryDisplayBox = document.getElementById('query-display-box');
        if (!queryDisplayBox || !uploadedDatasetInfo) return;

        const features = uploadedDatasetInfo.features;
        const targetColumn = uploadedDatasetInfo.target_column;
        const predText = (predictionVal !== undefined && !isNaN(predictionVal)) ? parseFloat(predictionVal).toFixed(2) : "Pending Run...";

        queryDisplayBox.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h4 style="margin: 0; font-size: 0.95rem; font-weight: 600; color: var(--primary-hover);">
                    <i class="fa-solid fa-circle-question"></i> Current Query Instance Details
                </h4>
                <div style="display: flex; gap: 8px;">
                    <button type="button" class="btn btn-success" id="btn-run-query" style="padding: 6px 12px; font-size: 0.8rem; border-radius: 6px; background: var(--success); color: var(--bg-dark);">
                        <i class="fa-solid fa-play"></i> Run Query
                    </button>
                    <button type="button" class="btn btn-primary" id="btn-random-query" style="padding: 6px 12px; font-size: 0.8rem; border-radius: 6px; background: var(--primary);">
                        <i class="fa-solid fa-shuffle"></i> Random Query
                    </button>
                </div>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 12px; align-items: center;">
                ${features.map(feat => {
                    const selectEl = document.querySelector(`.feature-type-select[data-feature="${feat}"]`);
                    const isInt = selectEl ? (selectEl.value === 'int') : (uploadedDatasetInfo.integer_features && uploadedDatasetInfo.integer_features.includes(feat));
                    const val = queryInstance[feat];
                    const displayVal = isInt ? Math.round(val) : val.toFixed(2);
                    return `
                        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem; display: flex; align-items: center;">
                            <span style="color: var(--text-muted); font-weight: 500;">${feat}:</span>
                            <input type="number" class="query-instance-input" data-feature="${feat}" step="${isInt ? '1' : 'any'}" value="${displayVal}" style="width: 90px; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 4px; padding: 2px 6px; color: white; font-family: var(--font-outfit); font-size: 0.85rem; margin-left: 6px; text-align: right;">
                        </div>
                    `;
                }).join('')}
                <div style="background: rgba(0, 229, 255, 0.08); border: 1px solid rgba(0, 229, 255, 0.2); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem; color: var(--secondary); display: flex; align-items: center;">
                    <span style="font-weight: 500;">${targetColumn} (Predicted):</span>
                    <span style="font-weight: 700; margin-left: 6px;">${predText}</span>
                </div>
            </div>
        `;

        // Attach event listeners to the newly rendered buttons
        const btnRandomQuery = document.getElementById('btn-random-query');
        if (btnRandomQuery) {
            btnRandomQuery.addEventListener('click', async () => {
                await selectRandomQueryAndRun();
            });
        }

        const btnRunQuery = document.getElementById('btn-run-query');
        if (btnRunQuery) {
            btnRunQuery.addEventListener('click', () => {
                parametersForm.requestSubmit();
            });
        }
    }

    // Render DiCE recommendations table
    function renderDiceTable(queryInstance, counterfactuals, threshold, diceError) {
        diceTableHeader.innerHTML = '';
        diceTableBody.innerHTML = '';

        const features = uploadedDatasetInfo.features;
        const targetColumn = uploadedDatasetInfo.target_column;

        // Render current query instance above the table
        updateQueryDisplayBox(queryInstance, parseFloat(valQueryPrediction.textContent));

        // Build header
        const thType = document.createElement('th');
        thType.textContent = 'Instance Type';
        diceTableHeader.appendChild(thType);

        features.forEach(feat => {
            const th = document.createElement('th');
            th.textContent = feat;
            diceTableHeader.appendChild(th);
        });

        const thTarget = document.createElement('th');
        thTarget.textContent = `${targetColumn} (Predicted)`;
        diceTableHeader.appendChild(thTarget);

        const thLowerBound = document.createElement('th');
        thLowerBound.textContent = 'Predicted Lower Bound';
        diceTableHeader.appendChild(thLowerBound);

        // Add Counterfactual Rows
        if (counterfactuals.length === 0) {
            const tr = document.createElement('tr');
            const colspan = features.length + 3;
            const msg = diceError ? `DiCE Execution Error: ${diceError}` : "No counterfactuals found matching the criteria. Try expanding permitted feature ranges or adjusting desired target range.";
            tr.innerHTML = `<td colspan="${colspan}" style="text-align: center; color: var(--danger); padding: 24px;"><i class="fa-solid fa-triangle-exclamation"></i> ${msg}</td>`;
            diceTableBody.appendChild(tr);
            return;
        }

        counterfactuals.forEach((cf, idx) => {
            const cfRow = document.createElement('tr');
            
            let td = document.createElement('td');
            td.innerHTML = `<i class="fa-solid fa-lightbulb" style="color: var(--success)"></i> Alternative #${idx + 1}`;
            cfRow.appendChild(td);

            features.forEach(feat => {
                td = document.createElement('td');
                const val = cf[feat];
                const origVal = queryInstance[feat];
                
                const selectEl = document.querySelector(`.feature-type-select[data-feature="${feat}"]`);
                const isInt = selectEl ? (selectEl.value === 'int') : (uploadedDatasetInfo.integer_features && uploadedDatasetInfo.integer_features.includes(feat));
                
                const displayVal = isInt ? Math.round(val) : val.toFixed(2);
                
                // Compare and highlight changes
                if (Math.abs(val - origVal) > 0.0001) {
                    td.innerHTML = `<span class="cell-changed">${displayVal}</span>`;
                } else {
                    td.textContent = displayVal;
                }
                cfRow.appendChild(td);
            });

            // Target column
            td = document.createElement('td');
            const targetVal = cf[targetColumn];
            if (targetVal !== undefined) {
                td.textContent = targetVal.toFixed(2);
                if (threshold !== undefined && targetVal >= threshold) {
                    td.style.color = 'var(--success)';
                    td.style.fontWeight = 'bold';
                }
            } else {
                td.textContent = 'N/A';
            }
            cfRow.appendChild(td);

            // Predicted Lower Bound column
            td = document.createElement('td');
            const lowerBoundVal = cf['predicted_lower_bound'];
            if (lowerBoundVal !== undefined) {
                td.textContent = parseFloat(lowerBoundVal).toFixed(2);
                td.style.color = 'var(--text-muted)';
            } else {
                td.textContent = 'N/A';
            }
            cfRow.appendChild(td);

            diceTableBody.appendChild(cfRow);
        });
    }

    // Tabs functionality
    const tabs = document.querySelectorAll('.tab-btn');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs
            tabs.forEach(t => t.classList.remove('active'));
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => content.classList.add('hidden'));
            
            // Add active to current tab
            tab.classList.add('active');
            // Show corresponding content
            const tabId = tab.getAttribute('data-tab');
            document.getElementById(tabId).classList.remove('hidden');
        });
    });
});
