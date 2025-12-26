// Import the compiled Rust module
const rustCore = require('../../rust-core');

export const handleAnalysis = async (req, res) => {
    const { documentText } = req.body;
    
    // Call the Rust function directly!
    const result = rustCore.analyzeLegalDocument(documentText);
    
    res.json({ success: true, analysis: result });
};