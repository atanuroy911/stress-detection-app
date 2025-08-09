from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI()

@app.get("/")
async def root():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stress Prediction - Test</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Stress Level Prediction Test</h1>
            <p>This is a test deployment to verify Vercel is working correctly.</p>
            
            <h2>Quick Test</h2>
            <button onclick="testAPI()">Test API</button>
            <div id="result"></div>
            
            <script>
                async function testAPI() {
                    try {
                        const response = await fetch('/test');
                        const result = await response.json();
                        document.getElementById('result').innerHTML = 
                            '<h3>✅ Success!</h3><pre>' + JSON.stringify(result, null, 2) + '</pre>';
                    } catch (error) {
                        document.getElementById('result').innerHTML = 
                            '<h3>❌ Error:</h3><p>' + error.message + '</p>';
                    }
                }
            </script>
        </div>
    </body>
    </html>
    """)

@app.get("/test")
async def test():
    return {
        "status": "success",
        "message": "Vercel deployment is working!",
        "stress_prediction": "Ready to analyze stress levels"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "test-1.0"}

# This is the correct way for Vercel
# No handler function needed
