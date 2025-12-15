// Function to traverse the DOM and extract all text nodes
function extractTextNodesFromPage() {
    const textNodes = [];

    function traverse(element, path = '') {
        if (!element) return;

        for (let child of element.childNodes) {
            if (child.nodeType === Node.TEXT_NODE) {
                const text = child.textContent.trim();
                if (text) {
                    const nodePath = path || (element.tagName ? element.tagName.toLowerCase() : 'root');
                    textNodes.push({
                        path: nodePath,
                        parentTag: (element.tagName ? element.tagName.toLowerCase() : 'root'),
                        text: text
                    });
                }
            } else if (child.nodeType === Node.ELEMENT_NODE) {
                const newPath = path ? `${path}/${child.tagName.toLowerCase()}` : child.tagName.toLowerCase();
                traverse(child, newPath);
            }
        }
    }

    // Start traversal from the document root
    traverse(document.documentElement);

    return textNodes;
}

// Function to download the extracted data as JSON
function downloadExtractedText() {
    const textNodes = extractTextNodesFromPage();
    const dataStr = JSON.stringify(textNodes,['text'], 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);

    const exportFileDefaultName = 'extracted_text_nodes.json';

    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();

    console.log(`Extracted ${textNodes.length} text nodes and initiated download.`);
}
// Function to extract conversation from user-query and response-container elements
function extractConversation() {
    const conversationLines = [];
    const elements = document.querySelectorAll('user-query, response-container');

    elements.forEach(element => {
        const text = element.textContent.trim();
        if (text) {
            if (element.tagName.toLowerCase() === 'user-query') {
                conversationLines.push('user: ' + text);
            } else if (element.tagName.toLowerCase() === 'response-container') {
                conversationLines.push('bot: ' + text);
            }
        }
    });

    return conversationLines.join('\n\n');
}

// Function to download the conversation as TXT
function downloadConversation() {
    const conversationText = extractConversation();
    const dataUri = 'data:text/plain;charset=utf-8,' + encodeURIComponent(conversationText);

    const exportFileDefaultName = 'conversation.txt';

    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();

    console.log('Conversation extracted and download initiated.');
}

downloadConversation();