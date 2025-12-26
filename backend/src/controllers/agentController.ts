// Import the compiled Rust module
const rustCore = require('../../rust-core');

export const handleAnalysis = async (req, res) => {
    const { documentText } = req.body;
    
    // Call the Rust function directly!
    const result = rustCore.analyzeLegalDocument(documentText);
    
    res.json({ success: true, analysis: result });
};

export const generateDraft = async (req: any, res: any) => {
    try {
        const { prompt } = req.body;
        
        // This calls the generate_legal_draft #[napi] function in rust-core/src/lib.rs
        const result = await rustCore.generateLegalDraft(prompt);
        
        res.status(200).json({
            success: true,
            analysis: result
        });
    } catch (error) {
        res.status(500).json({ error: "Sovereign Engine Error" });
    }
};