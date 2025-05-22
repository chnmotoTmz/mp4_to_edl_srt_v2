const { app, BrowserWindow, ipcMain, net } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

let mainWindow;
let pythonServerProcess;
const PYTHON_SERVER_URL = 'http://127.0.0.1:5001'; // Make sure this matches server.py

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            nodeIntegration: true, // Required for IPC and other Node.js features in renderer
            contextIsolation: false, // Simplifies IPC for now, consider true + preload for security
            // preload: path.join(__dirname, 'preload.js') // Example if using contextIsolation: true
        },
    });

    mainWindow.loadFile('index.html');

    // Open DevTools - useful for debugging
    // mainWindow.webContents.openDevTools();

    mainWindow.on('closed', () => {
        mainWindow = null;
    });

    // After window is created, try to connect to SSE for progress updates
    connectToSSE();
}

function startPythonServer() {
    // Adjust the path to server.py if your project structure is different.
    // This assumes main.js is at the project root and server.py is in mp4_to_edl_srt/
    const scriptPath = path.join(__dirname, 'mp4_to_edl_srt', 'server.py');
    
    // Determine Python command (python or python3)
    // For simplicity, this example uses 'python'. This might need to be configurable.
    const pythonExecutable = 'python'; // or 'python3' or path to venv python

    pythonServerProcess = spawn(pythonExecutable, [scriptPath]);

    pythonServerProcess.stdout.on('data', (data) => {
        console.log(`PythonServer STDOUT: ${data}`);
        // Optionally, forward to renderer for display, though SSE is primary for progress
    });

    pythonServerProcess.stderr.on('data', (data) => {
        console.error(`PythonServer STDERR: ${data}`);
        // Optionally, forward critical errors to renderer
        if (mainWindow) {
            mainWindow.webContents.send('python-server-error', data.toString());
        }
    });

    pythonServerProcess.on('error', (error) => {
        console.error(`Failed to start Python server: ${error}`);
        if (mainWindow) {
            mainWindow.webContents.send('python-server-error', `Failed to start Python server: ${error.message}`);
        }
    });
    
    pythonServerProcess.on('close', (code) => {
        console.log(`Python server process exited with code ${code}`);
        pythonServerProcess = null;
        // Optionally, notify renderer or attempt restart
    });

    console.log("Python server started.");
}

function stopPythonServer() {
    if (pythonServerProcess) {
        console.log('Stopping Python server...');
        pythonServerProcess.kill('SIGINT'); // Send SIGINT for graceful shutdown if server handles it
        // Or use pythonServerProcess.kill() for a more forceful termination
        pythonServerProcess = null;
        console.log('Python server stopped.');
    }
}

function connectToSSE() {
    const sseUrl = `${PYTHON_SERVER_URL}/stream-progress`;
    const request = net.request(sseUrl);

    request.on('response', (response) => {
        console.log(`SSE Connection Status: ${response.statusCode}`);
        if (response.statusCode === 200) {
            response.on('data', (chunk) => {
                const rawData = chunk.toString();
                // SSE messages are "data: <json_string>\n\n"
                // Multiple messages can arrive in one chunk
                const messages = rawData.split('\n\n');
                messages.forEach(message => {
                    if (message.startsWith('data: ')) {
                        const jsonData = message.substring('data: '.length);
                        try {
                            const parsedData = JSON.parse(jsonData);
                            if (mainWindow) {
                                mainWindow.webContents.send('progress-update', parsedData);
                            }
                        } catch (e) {
                            console.error('Error parsing SSE data:', e, "\nRaw data:", jsonData);
                        }
                    } else if (message.startsWith(':')) {
                        // This is a comment (e.g., keep-alive)
                        console.log('SSE keep-alive received');
                    }
                });
            });

            response.on('end', () => {
                console.log('SSE stream ended.');
                // Optionally, try to reconnect after a delay
                // setTimeout(connectToSSE, 5000); // Example: retry after 5 seconds
            });

            response.on('error', (error) => {
                console.error('Error with SSE connection (response error):', error);
                // Optionally, try to reconnect
            });
        } else {
            console.error(`Failed to connect to SSE stream. Status: ${response.statusCode}`);
             // Optionally, try to reconnect
        }
    });

    request.on('error', (error) => {
        console.error('Error making SSE request:', error);
        // Server might not be up yet, retry
        if (mainWindow) { // Only retry if the app window is still open
             setTimeout(connectToSSE, 5000); // Example: retry after 5 seconds
        }
    });

    request.end();
    console.log('Attempting to connect to SSE endpoint:', sseUrl);
}


// --- App Lifecycle ---
app.on('ready', () => {
    startPythonServer(); // Start Python server first
    createWindow();    // Then create the window
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (mainWindow === null) {
        createWindow();
    }
});

app.on('will-quit', () => {
    stopPythonServer();
});

// --- IPC Handlers ---

// Helper function for making requests to Python server
async function makePythonRequest(endpoint, method = 'GET', body = null) {
    const url = `${PYTHON_SERVER_URL}${endpoint}`;
    let responseBody = '';
    
    return new Promise((resolve, reject) => {
        const request = net.request({ method, url });

        request.on('response', (response) => {
            // console.log(`Python request to ${url} STATUS: ${response.statusCode}`);
            response.on('data', (chunk) => {
                responseBody += chunk.toString();
            });
            response.on('end', () => {
                try {
                    resolve(JSON.parse(responseBody));
                } catch (e) {
                    console.error(`Error parsing JSON response from ${url}:`, e, "\nRaw body:", responseBody);
                    reject(new Error(`Invalid JSON response: ${responseBody.substring(0,100)}`));
                }
            });
            response.on('error', (error) => {
                console.error(`Error in response from ${url}:`, error);
                reject(error);
            });
        });

        request.on('error', (error) => {
            console.error(`Error making request to ${url}:`, error);
            reject(error);
        });

        if (body && (method === 'POST' || method === 'PUT')) {
            const jsonData = JSON.stringify(body);
            request.setHeader('Content-Type', 'application/json');
            request.setHeader('Content-Length', Buffer.byteLength(jsonData));
            request.write(jsonData);
        }
        request.end();
    });
}

ipcMain.handle('start-processing', async (event, args) => {
    try {
        console.log("IPC 'start-processing' received with args:", args);
        const response = await makePythonRequest('/process', 'POST', args);
        return response;
    } catch (error) {
        console.error("Error in 'start-processing' IPC handler:", error);
        return { error: error.message || 'Failed to start processing' };
    }
});

ipcMain.handle('get-config', async () => {
    try {
        const response = await makePythonRequest('/config');
        return response;
    } catch (error) {
        console.error("Error in 'get-config' IPC handler:", error);
        return { error: error.message || 'Failed to get config' };
    }
});

ipcMain.handle('save-config', async (event, configData) => {
    try {
        const response = await makePythonRequest('/config', 'POST', configData);
        return response;
    } catch (error) {
        console.error("Error in 'save-config' IPC handler:", error);
        return { error: error.message || 'Failed to save config' };
    }
});

// For selecting directories - useful for input/output folders
ipcMain.handle('select-directory', async () => {
    const { dialog } = require('electron');
    const result = await dialog.showOpenDialog(mainWindow, {
        properties: ['openDirectory']
    });
    if (result.canceled || result.filePaths.length === 0) {
        return null; // Or some indicator of cancellation
    }
    return result.filePaths[0];
});
