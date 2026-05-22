document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('text-input');
    const ghostText = document.getElementById('ghost-text');
    const suggestionsContainer = document.getElementById('suggestions');
    const chatWindow = document.getElementById('chat-window');
    const sendBtn = document.getElementById('send-btn');
    const confidenceBar = document.getElementById('confidence-bar');
    
    let debounceTimer;
    let currentTopPrediction = "";
    let chatHistory = [];

    textInput.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        const text = textInput.value;
        ghostText.textContent = text;
        
        // Auto-resize textarea
        textInput.style.height = 'auto';
        textInput.style.height = (textInput.scrollHeight) + 'px';
        ghostText.style.height = textInput.style.height;

        if (!text.trim()) {
            hideSuggestions();
            ghostText.textContent = "";
            return;
        }

        debounceTimer = setTimeout(() => {
            fetchPredictions(text);
        }, 150);
    });
    
    textInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
        
        if ((e.key === 'Tab' || e.key === 'ArrowRight') && currentTopPrediction) {
            const text = textInput.value;
            const words = text.split(/\s+/);
            const lastWord = words[words.length - 1];
            
            if (currentTopPrediction.startsWith(lastWord) || text.endsWith(' ')) {
                e.preventDefault();
                applySuggestion(currentTopPrediction);
            }
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    async function sendMessage() {
        const text = textInput.value.trim();
        if (!text) return;

        appendMessage('user', text);
        textInput.value = '';
        textInput.style.height = 'auto';
        ghostText.textContent = '';
        hideSuggestions();
        
        // Reset diagnostics HUD upon sending message
        document.getElementById('confidence-val').textContent = '0.0%';
        document.getElementById('pos-val').textContent = 'NONE';
        updateActiveBrainHUD('STANDBY');
        if (confidenceBar) confidenceBar.style.width = '0%';
        
        const botMsgDiv = appendMessage('bot', '...'); 
        botMsgDiv.classList.add('typing');

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, history: chatHistory })
            });

            if (response.ok) {
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullResponse = "";
                let currentBubbleText = "";
                let currentBotMsgDiv = botMsgDiv;
                currentBotMsgDiv.textContent = "";
                currentBotMsgDiv.classList.remove('typing');

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\n');
                    for (const line of lines) {
                        if (line.trim().startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.trim().slice(6));
                                
                                // Check for Active Brain process updates
                                if (data.process) {
                                    updateActiveBrainHUD(data.process);
                                    continue;
                                }
                                
                                const content = data.chunk;
                                if (!content) continue;
                                
                                // Check for paragraph breaks to split into bubbles
                                if (content.includes('\n\n') || (currentBubbleText.length > 500 && content.includes('\n'))) {
                                    const parts = content.split(/\n\n|\n/);
                                    
                                    // Add first part to current bubble
                                    currentBubbleText += parts[0];
                                    currentBotMsgDiv.textContent = currentBubbleText;
                                    
                                    // Start new bubble for subsequent parts
                                    for (let i = 1; i < parts.length; i++) {
                                        if (parts[i].trim()) {
                                            currentBubbleText = parts[i];
                                            currentBotMsgDiv = appendMessage('bot', currentBubbleText);
                                        }
                                    }
                                } else {
                                    currentBubbleText += content;
                                    currentBotMsgDiv.textContent = currentBubbleText;
                                }
                                
                                fullResponse += content;
                                chatWindow.scrollTop = chatWindow.scrollHeight;
                            } catch(e) {}
                        }
                    }
                }
                updateActiveBrainHUD('STANDBY');
                chatHistory.push(text);
                chatHistory.push(fullResponse);
                if (chatHistory.length > 10) chatHistory.shift();
            }
        } catch (error) {
            console.error('Error sending message:', error);
            botMsgDiv.textContent = "Connection lost. Neural link unstable.";
            updateActiveBrainHUD('STANDBY');
        }
    }

    function appendMessage(sender, text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-message`;
        msgDiv.textContent = text;
        chatWindow.appendChild(msgDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
        return msgDiv;
    }

    async function fetchPredictions(text) {
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, history: chatHistory })
            });

            if (response.ok) {
                const data = await response.json();
                renderSuggestions(data.predictions, data.scores);
                updateDiagnostics(data.diagnostic);
                updateGhostText(text, data.predictions[0]);
            }
        } catch (error) {
            console.error('Error fetching predictions:', error);
        }
    }

    function updateGhostText(currentText, topPrediction) {
        if (!topPrediction) {
            currentTopPrediction = "";
            return;
        }

        currentTopPrediction = topPrediction;
        const words = currentText.split(/\s+/);
        const lastWord = currentText.endsWith(' ') ? "" : words[words.length - 1];
        
        if (topPrediction.startsWith(lastWord)) {
            const completion = topPrediction.slice(lastWord.length);
            ghostText.textContent = currentText + completion;
        } else {
            ghostText.textContent = currentText;
        }
    }

    function renderSuggestions(predictions, scores) {
        suggestionsContainer.innerHTML = '';
        if (predictions && predictions.length > 0) {
            predictions.forEach(pred => {
                const btn = document.createElement('button');
                btn.className = 'suggestion-btn';
                const score = scores && scores[pred] ? ` (${scores[pred]}%)` : '';
                btn.textContent = pred + score;
                btn.addEventListener('click', () => applySuggestion(pred));
                suggestionsContainer.appendChild(btn);
            });
        }
    }

    function updateActiveBrainHUD(processName) {
        const valEl = document.getElementById('process-val');
        const waveEl = document.getElementById('neural-wave');
        if (!valEl) return;
        
        valEl.textContent = processName;
        
        // Remove existing status classes
        valEl.classList.remove('status-standby', 'status-system1', 'status-system2-rag', 'status-system2-llama3');
        
        if (processName.includes('RAG')) {
            valEl.classList.add('status-system2-rag');
            if (waveEl) waveEl.classList.remove('hidden');
        } else if (processName.includes('Llama') || processName.includes('System 2')) {
            valEl.classList.add('status-system2-llama3');
            if (waveEl) waveEl.classList.remove('hidden');
        } else if (processName.includes('System 1') || processName.includes('Stochastic') || processName.includes('Statistical') || processName.includes('Brain')) {
            valEl.classList.add('status-system1');
            if (waveEl) waveEl.classList.remove('hidden');
        } else {
            valEl.classList.add('status-standby');
            if (waveEl) waveEl.classList.add('hidden');
        }
    }

    function updateDiagnostics(diag) {
        if (!diag) return;
        document.getElementById('confidence-val').textContent = `${diag.confidence}%`;
        document.getElementById('pos-val').textContent = diag.pos;
        updateActiveBrainHUD(diag.process);
        
        // Update the sidebar bar
        if (confidenceBar) {
            confidenceBar.style.width = `${diag.confidence}%`;
        }
    }

    function applySuggestion(word) {
        const text = textInput.value;
        let newText = "";
        
        const isPunctuation = /^[.,!?]$/.test(word);
        
        if (isPunctuation) {
            newText = text.trimEnd() + word + ' ';
        } else if (text.endsWith(' ')) {
            newText = text + word + ' ';
        } else {
            const words = text.split(/\s+/);
            words.pop();
            newText = words.join(' ') + (words.length > 0 ? ' ' : '') + word + ' ';
        }
        
        textInput.value = newText;
        ghostText.textContent = newText;
        currentTopPrediction = "";
        textInput.focus();
        fetchPredictions(newText);
    }

    function hideSuggestions() {
        suggestionsContainer.innerHTML = '';
    }

    const syncBtn = document.getElementById('sync-btn');
    if (syncBtn) {
        syncBtn.addEventListener('click', async () => {
            syncBtn.disabled = true;
            syncBtn.querySelector('span').textContent = 'Syncing...';
            try {
                const response = await fetch('/sync', { method: 'POST' });
                if (response.ok) {
                    appendMessage('bot', 'System Alert: Declarative memory synchronized. Knowledge base re-indexed.');
                }
            } catch (e) {
                console.error(e);
            } finally {
                syncBtn.disabled = false;
                syncBtn.querySelector('span').textContent = 'Sync';
            }
        });
    }
});
