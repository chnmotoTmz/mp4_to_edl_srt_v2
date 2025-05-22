const { ipcRenderer } = require('electron');

const TRANSLATIONS = {
    en: {
        appTitle: "MP4 to EDL/SRT Converter",
        languageSelectLabel: "Language:",
        inputFolderLabel: "Input Folder:",
        outputFolderLabel: "Output Folder:",
        browse: "Browse",
        initialPromptLabel: "Initial Prompt (Optional):",
        advancedOptionsLabel: "Advanced Options",
        useTimecodeOffsetLabel: "Use internal timecode from MP4 files",
        enableSceneAnalysisLabel: "Enable Scene Analysis",
        sceneAnalysisRateLabel: "Scene Analysis Frame Rate (fps):",
        startConversionBtn: "Start Conversion",
        cancelBtn: "Cancel",
        logAreaLabel: "Log / Progress",
        footerText: "Electron Version",
        selectInputFolder: "Select Input Folder",
        selectOutputFolder: "Select Output Folder",
        processingStarted: "Processing started...",
        processingError: "Error during processing:",
        configLoaded: "Configuration loaded.",
        configSaved: "Configuration saved.",
        errorLoadingConfig: "Error loading configuration:",
        errorSavingConfig: "Error saving configuration:",
        pathNotSelected: "Path not selected.",
        processingUpdate: "Update:",
        processingComplete: "Processing complete!",
        pythonServerError: "Python Server Error:",
        stage_audio_extraction: "Audio Extraction",
        stage_transcription: "Transcription",
        stage_scene_analysis: "Scene Analysis",
        stage_edl_generation: "EDL Generation",
        stage_srt_generation: "SRT Generation",
        stage_cleanup: "Cleaning up",
        stage_done: "Done"
    },
    ja: {
        appTitle: "MP4からEDL/SRTへの変換ツール",
        languageSelectLabel: "言語:",
        inputFolderLabel: "入力フォルダ:",
        outputFolderLabel: "出力フォルダ:",
        browse: "参照",
        initialPromptLabel: "初期プロンプト (任意):",
        advancedOptionsLabel: "詳細オプション",
        useTimecodeOffsetLabel: "MP4ファイルの内部タイムコードを使用する",
        enableSceneAnalysisLabel: "シーン分析を有効にする",
        sceneAnalysisRateLabel: "シーン分析フレームレート (fps):",
        startConversionBtn: "変換開始",
        cancelBtn: "キャンセル",
        logAreaLabel: "ログ / 進捗",
        footerText: "Electron版",
        selectInputFolder: "入力フォルダを選択",
        selectOutputFolder: "出力フォルダを選択",
        processingStarted: "処理を開始しました...",
        processingError: "処理中にエラーが発生しました:",
        configLoaded: "設定を読み込みました。",
        configSaved: "設定を保存しました。",
        errorLoadingConfig: "設定の読み込みエラー:",
        errorSavingConfig: "設定の保存エラー:",
        pathNotSelected: "パスが選択されていません。",
        processingUpdate: "進捗:",
        processingComplete: "処理が完了しました！",
        pythonServerError: "Pythonサーバーエラー:",
        stage_audio_extraction: "音声抽出",
        stage_transcription: "文字起こし",
        stage_scene_analysis: "シーン分析",
        stage_edl_generation: "EDL生成",
        stage_srt_generation: "SRT生成",
        stage_cleanup: "クリーンアップ",
        stage_done: "完了"
    }
};

// DOM Element References
const appTitle = document.getElementById('appTitle');
const appTitleHeading = document.getElementById('appTitleHeading');
const languageSelect = document.getElementById('languageSelect');
const languageSelectLabel = document.getElementById('languageSelectLabel');

const inputFolderInput = document.getElementById('inputFolder');
const inputFolderLabel = document.getElementById('inputFolderLabel');
const browseInputFolderBtn = document.getElementById('browseInputFolder');

const outputFolderInput = document.getElementById('outputFolder');
const outputFolderLabel = document.getElementById('outputFolderLabel');
const browseOutputFolderBtn = document.getElementById('browseOutputFolder');

const initialPromptInput = document.getElementById('initialPrompt');
const initialPromptLabel = document.getElementById('initialPromptLabel');

const advancedOptionsLabel = document.querySelector('details summary');
const useTimecodeOffsetCheckbox = document.getElementById('useTimecodeOffset');
const useTimecodeOffsetLabel = document.getElementById('useTimecodeOffsetLabel');
const enableSceneAnalysisCheckbox = document.getElementById('enableSceneAnalysis');
const enableSceneAnalysisLabel = document.getElementById('enableSceneAnalysisLabel');
const sceneAnalysisRateInput = document.getElementById('sceneAnalysisRate');
const sceneAnalysisRateLabel = document.getElementById('sceneAnalysisRateLabel');

const startConversionBtn = document.getElementById('startConversionBtn');
const cancelBtn = document.getElementById('cancelBtn');
const logAreaLabel = document.getElementById('logAreaLabel');
const logArea = document.getElementById('logArea');
const footerText = document.getElementById('footerText');

// --- UI Text Update Function ---
function updateUIText(lang) {
    const t = TRANSLATIONS[lang] || TRANSLATIONS.en;

    document.title = t.appTitle; // Update window title
    appTitleHeading.textContent = t.appTitle;
    languageSelectLabel.textContent = t.languageSelectLabel;

    inputFolderLabel.textContent = t.inputFolderLabel;
    browseInputFolderBtn.textContent = t.browse;
    outputFolderLabel.textContent = t.outputFolderLabel;
    browseOutputFolderBtn.textContent = t.browse;
    initialPromptLabel.textContent = t.initialPromptLabel;

    advancedOptionsLabel.textContent = t.advancedOptionsLabel;
    useTimecodeOffsetLabel.textContent = t.useTimecodeOffsetLabel;
    enableSceneAnalysisLabel.textContent = t.enableSceneAnalysisLabel;
    sceneAnalysisRateLabel.textContent = t.sceneAnalysisRateLabel;

    startConversionBtn.textContent = t.startConversionBtn;
    cancelBtn.textContent = t.cancelBtn;
    logAreaLabel.textContent = t.logAreaLabel;
    footerText.textContent = t.footerText;
}

// --- Configuration Loading ---
async function loadAndApplyConfig() {
    try {
        logToArea('Loading configuration...');
        const config = await ipcRenderer.invoke('get-config');
        if (config && !config.error) {
            inputFolderInput.value = config.input_folder || '';
            outputFolderInput.value = config.output_folder || '';
            initialPromptInput.value = config.initial_prompt || '';
            
            useTimecodeOffsetCheckbox.checked = config.use_timecode_offset !== undefined ? config.use_timecode_offset : false;
            enableSceneAnalysisCheckbox.checked = config.scene_analysis_enabled !== undefined ? config.scene_analysis_enabled : false;
            sceneAnalysisRateInput.value = config.scene_analysis_rate || 10;
            
            // Toggle scene analysis rate input based on checkbox
            sceneAnalysisRateInput.disabled = !enableSceneAnalysisCheckbox.checked;

            // Set language - assuming config might have 'ui_language'
            const uiLang = config.ui_language || 'ja'; // Default to Japanese as per original spec
            languageSelect.value = uiLang;
            updateUIText(uiLang);
            logToArea(TRANSLATIONS[uiLang].configLoaded || 'Configuration loaded.');

        } else if (config && config.error) {
            logToArea(`Error loading config: ${config.error}`);
        } else {
            logToArea('Received empty or invalid config.');
             // Apply default language if config is empty
            updateUIText(languageSelect.value || 'ja');
        }
    } catch (error) {
        console.error('Error loading config:', error);
        logToArea(`Error loading config: ${error.message}`);
        // Apply default language on error
        updateUIText(languageSelect.value || 'ja');
    }
}

// --- Logging Function ---
function logToArea(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('p');
    logEntry.innerHTML = `<strong>[${timestamp}] ${type === 'error' ? 'ERROR' : 'INFO'}:</strong> ${message}`;
    if (type === 'error') {
        logEntry.style.color = 'red';
    }
    logArea.appendChild(logEntry);
    logArea.scrollTop = logArea.scrollHeight; // Auto-scroll
}


// --- Configuration Saving ---
async function saveCurrentConfig() {
    const currentLang = languageSelect.value || 'en';
    const configData = {
        input_folder: inputFolderInput.value,
        output_folder: outputFolderInput.value,
        initial_prompt: initialPromptInput.value,
        use_timecode_offset: useTimecodeOffsetCheckbox.checked,
        scene_analysis_enabled: enableSceneAnalysisCheckbox.checked,
        scene_analysis_rate: parseInt(sceneAnalysisRateInput.value, 10) || 10,
        ui_language: languageSelect.value
    };

    try {
        // logToArea('Attempting to save configuration...', 'debug'); // Optional: for more verbose logging
        const result = await ipcRenderer.invoke('save-config', configData);
        if (result && result.error) {
            logToArea(`${TRANSLATIONS[currentLang].errorSavingConfig} ${result.error}`, 'error');
        } else {
            // logToArea(TRANSLATIONS[currentLang].configSaved); // Can be too noisy, log only on explicit save action if preferred
        }
    } catch (error) {
        logToArea(`${TRANSLATIONS[currentLang].errorSavingConfig} ${error.message}`, 'error');
    }
}


// --- UI State Management ---
function setControlsProcessing(isProcessing) {
    startConversionBtn.disabled = isProcessing;
    cancelBtn.disabled = !isProcessing;
    // Optionally disable other form inputs during processing
    inputFolderInput.disabled = isProcessing;
    browseInputFolderBtn.disabled = isProcessing;
    outputFolderInput.disabled = isProcessing;
    browseOutputFolderBtn.disabled = isProcessing;
    initialPromptInput.disabled = isProcessing;
    useTimecodeOffsetCheckbox.disabled = isProcessing;
    enableSceneAnalysisCheckbox.disabled = isProcessing;
    sceneAnalysisRateInput.disabled = isProcessing || !enableSceneAnalysisCheckbox.checked; // Keep original logic for scene rate
    languageSelect.disabled = isProcessing;
}


// --- Initial Setup & Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {
    loadAndApplyConfig(); // Load config and then update UI based on it (including language)

    // Event listener for language selection
    languageSelect.addEventListener('change', (event) => {
        updateUIText(event.target.value);
        saveCurrentConfig(); // Save language choice
    });

    // Event listener for scene analysis toggle
    enableSceneAnalysisCheckbox.addEventListener('change', () => {
        sceneAnalysisRateInput.disabled = !enableSceneAnalysisCheckbox.checked;
        saveCurrentConfig();
    });

    // Folder Selection
    browseInputFolderBtn.addEventListener('click', async () => {
        const currentLang = languageSelect.value || 'en';
        logToArea(TRANSLATIONS[currentLang].selectInputFolder);
        const path = await ipcRenderer.invoke('select-directory');
        if (path) {
            inputFolderInput.value = path;
            saveCurrentConfig();
        } else {
            logToArea(TRANSLATIONS[currentLang].pathNotSelected);
        }
    });

    browseOutputFolderBtn.addEventListener('click', async () => {
        const currentLang = languageSelect.value || 'en';
        logToArea(TRANSLATIONS[currentLang].selectOutputFolder);
        const path = await ipcRenderer.invoke('select-directory');
        if (path) {
            outputFolderInput.value = path;
            saveCurrentConfig();
        } else {
            logToArea(TRANSLATIONS[currentLang].pathNotSelected);
        }
    });

    // Auto-save for other input fields
    [initialPromptInput, sceneAnalysisRateInput].forEach(input => {
        input.addEventListener('change', saveCurrentConfig); // 'change' is better than 'input' for less frequent saves
    });
    [useTimecodeOffsetCheckbox].forEach(checkbox => { // 'enableSceneAnalysisCheckbox' already covered
        checkbox.addEventListener('change', saveCurrentConfig);
    });


    // Start Conversion Button
    startConversionBtn.addEventListener('click', async () => {
        const currentLang = languageSelect.value || 'en';
        logArea.innerHTML = ''; // Clear log area
        logToArea(TRANSLATIONS[currentLang].processingStarted);
        setControlsProcessing(true);

        const settings = {
            input_folder: inputFolderInput.value,
            output_folder: outputFolderInput.value,
            initial_prompt: initialPromptInput.value,
            use_timecode_offset: useTimecodeOffsetCheckbox.checked,
            scene_analysis_enabled: enableSceneAnalysisCheckbox.checked,
            scene_analysis_rate: parseInt(sceneAnalysisRateInput.value, 10)
        };

        if (!settings.input_folder || !settings.output_folder) {
            logToArea('Input and Output folders must be selected.', 'error');
            setControlsProcessing(false);
            return;
        }

        try {
            const result = await ipcRenderer.invoke('start-processing', settings);
            if (result && result.error) {
                logToArea(`${TRANSLATIONS[currentLang].processingError} ${result.error}`, 'error');
                setControlsProcessing(false);
            } else if (result) {
                logToArea(`Server response: ${result.message}`); // Or handle more structured response
            }
        } catch (error) {
            logToArea(`${TRANSLATIONS[currentLang].processingError} ${error.message}`, 'error');
            setControlsProcessing(false);
        }
    });

    // Cancel Button
    cancelBtn.addEventListener('click', () => {
        const currentLang = languageSelect.value || 'en';
        logToArea('Cancel requested by user. (Note: Backend cancellation not yet implemented)');
        // TODO: Implement actual cancellation via IPC if backend supports it
        setControlsProcessing(false); 
    });
    cancelBtn.disabled = true; // Initially disabled


    // IPC listener for progress updates from main process
    ipcRenderer.on('progress-update', (event, data) => {
        const lang = languageSelect.value || 'en';
        let stageKey = `stage_${data.stage?.toLowerCase().replace(/\s+/g, '_')}`;
        let stageMessage = (TRANSLATIONS[lang] && TRANSLATIONS[lang][stageKey]) ? TRANSLATIONS[lang][stageKey] : data.stage;
        
        let message;
        if (data.final) {
            message = `${TRANSLATIONS[lang].processingComplete || 'Processing complete!'}`;
            if (data.error) { // Assuming error might be passed in final message
                message = `${TRANSLATIONS[lang].processingError} ${data.error}`;
                logToArea(message, 'error');
            } else {
                logToArea(message);
            }
            setControlsProcessing(false);
        } else if (typeof data.value === 'number' && data.value >= 0 && data.value <=1000) { // value could be count or percentage
             if (data.value <= 1 && data.stage !== "Scene Analysis") { // Typically percentages are 0-1 from whisper, scale to 100
                message = `${stageMessage}: ${(data.value * 100).toFixed(1)}%`;
             } else { // Or it's a direct value/count (like scene analysis frame count)
                message = `${stageMessage}: ${data.value}`;
             }
             logToArea(message);
        } else {
            // Fallback for other types of messages
            message = `${stageMessage}: ${data.value}`;
            logToArea(message);
        }
    });
    
    // IPC listener for Python server errors
    ipcRenderer.on('python-server-error', (event, errorMsg) => {
        const lang = languageSelect.value || 'en';
        logToArea(`${TRANSLATIONS[lang].pythonServerError || 'Python Server Error:'} ${errorMsg}`, 'error');
        setControlsProcessing(false); // Stop processing state if server dies
    });

});
console.log("Renderer.js loaded.");
