document.getElementById('userInput').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') sendMessage();
});

document.getElementById('sendBtn').addEventListener('click', sendMessage);

// توليد Session ID فريد لكل مستخدم
const userId = "user_session_" + Math.random().toString(36).substring(2, 9);

async function sendMessage() {
    const inputEl = document.getElementById('userInput');
    const chatArea = document.getElementById('chatArea');
    const statusBar = document.getElementById('pipelineStatusBar');
    const statusText = document.getElementById('statusText');
    
    const messageText = inputEl.value.trim();
    if (!messageText) return;

    // 1. عرض رسالة المستخدم فوراً بتنسيق الـ Capsule
    chatArea.innerHTML += `
        <div class="message user-message">
            <div class="message-content">${escapeHtml(messageText)}</div>
        </div>
    `;
    inputEl.value = '';
    chatArea.scrollTop = chatArea.scrollHeight;

    // 2. إظهار لودر الـ Pipeline الشفاف
    statusBar.style.display = 'flex';
    statusText.innerText = 'Initializing Pipeline...';

    // 3. حجز مكان لرسالة البوت الجديدة بـ ID فريد يعتمد على الوقت لضمان عزل الرسائل
    const botMessageId = 'bot-msg-' + Date.now();
    chatArea.innerHTML += `
        <div class="message bot-message" id="${botMessageId}" style="display:none;">
            <div class="message-content" id="${botMessageId}-content"></div>
        </div>
    `;

    const botMessageDiv = document.getElementById(botMessageId);
    const botContentDiv = document.getElementById(`${botMessageId}-content`);

    try {
        const response = await fetch('/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: userId,
                message: messageText
            })
        });

        if (!response.ok) throw new Error('Failed to connect to the brain server.');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const jsonString = line.replace('data: ', '').trim();
                    if (!jsonString) continue;

                    try {
                        const packet = JSON.parse(jsonString);

                        // معالجة الـ Status (اللودر السفلي فقط)
                        if (packet.type === 'status') {
                            statusBar.style.display = 'flex';
                            statusText.innerText = formatStatusMessage(packet.node, packet.emotion);
                        }

                        // معالجة الـ Tokens (الطباعة الحية الحقيقية المنسابة)
                        if (packet.type === 'token') {
                            statusBar.style.display = 'none'; 
                            if (botMessageDiv.style.display === 'none') {
                                botMessageDiv.style.display = 'flex';
                            }
                            
                            // السيرفر متفلتر وجاهز، بنطبع الحروف فوراً هنا بسلاسة
                            botContentDiv.innerText += packet.data;
                            chatArea.scrollTop = chatArea.scrollHeight;
                        }

                        // عرض رسائل السيستم والأخطاء بشكل صريح
                        if (packet.type === 'error') {
                            statusBar.style.display = 'none';
                            botMessageDiv.style.display = 'flex';
                            botContentDiv.innerText = `[System Notification]: ${packet.data}`;
                        }

                    } catch (e) {
                        // كتم أخطاء الـ Parsing للسطور الفارغة
                    }
                }
            }
        }

    } catch (error) {
        statusBar.style.display = 'none';
        botMessageDiv.style.display = 'flex';
        botContentDiv.innerHTML = `<span style="color:#e74c3c;"><i class="fa-solid fa-triangle-exclamation"></i> Connection lost. Please try again.</span>`;
        chatArea.scrollTop = chatArea.scrollHeight;
    }
}

// دالة تنسيق الرسائل التوضيحية لخطوات الـ Pipeline
function formatStatusMessage(node, emotion) {
    const mapping = {
        'detect_language': 'Recognizing input language...',
        'emotion_classifier': emotion ? `Empathy Engine: Detecting emotional tone (${emotion})...` : 'Analyzing emotional context...',
        'translate_into_english': 'Bridging language to clinical understanding...',
        'intent_classifier': 'Understanding your core intent...',
        'reshape_user_query_to_situation': 'Contextualizing clinical background situation...',
        'retrieve_relevant_docs': 'Searching the trusted clinical knowledge base...',
        'generate_rag_answer': 'Crafting a personalized, safe and supportive response...',
        'generate_normal_answer': 'Formulating general supportive guidance...'
    };
    return mapping[node] || `Processing stage: ${node}...`;
}

// حماية الـ XSS عند طباعة مدخلات المستخدم
function escapeHtml(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}