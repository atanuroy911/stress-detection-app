# Vercel Debugging Guide

## 🔍 How to Debug Vercel Serverless Function Errors

### 1. **View Function Logs**
- Go to your Vercel Dashboard
- Click on your deployment
- Go to "Functions" tab
- Click on the failing function to see logs
- Look for error details and stack traces

### 2. **Common Vercel Error Patterns**

#### **FUNCTION_INVOCATION_FAILED**
```
500: INTERNAL_SERVER_ERROR
Code: FUNCTION_INVOCATION_FAILED
```

**Possible Causes:**
- Import errors (missing dependencies)
- File path issues (files not found)
- Syntax errors in Python code
- Memory/timeout issues
- Invalid function handler

### 3. **Debug Steps**

#### Step 1: Check Vercel Function Logs
```bash
# Install Vercel CLI
npm install -g vercel

# View logs in real-time
vercel logs https://your-app.vercel.app

# Or view specific function logs
vercel logs --since=1h
```

#### Step 2: Test Locally First
```bash
# Test the API locally
cd api
python demo.py

# Or use uvicorn
uvicorn demo:app --reload
```

#### Step 3: Verify Dependencies
```bash
# Check requirements.txt
cat requirements.txt

# Verify they install locally
pip install -r requirements.txt
```

#### Step 4: Check File Structure
```
your-app/
├── api/
│   └── demo.py          # Your main function
├── vercel.json          # Vercel configuration  
├── requirements.txt     # Python dependencies
└── README.md
```

### 4. **Fixed Common Issues**

#### Issue 1: File Reading Errors
**Problem**: `FileNotFoundError: public/index.html`
**Solution**: Embed HTML directly in the Python code instead of reading files

#### Issue 2: Import Errors  
**Problem**: `ModuleNotFoundError: No module named 'pandas'`
**Solution**: Use only lightweight dependencies in requirements.txt

#### Issue 3: Handler Export
**Problem**: Vercel can't find the function handler
**Solution**: Ensure proper FastAPI app export:
```python
from fastapi import FastAPI
app = FastAPI()

# For Vercel (choose one):
handler = app  # Option 1
# OR
def handler(request, response):  # Option 2
    return app
```

### 5. **Current Fix Applied**

I've fixed the following issues in your code:

✅ **Removed file reading**: No more `public/index.html` dependency
✅ **Embedded HTML**: All HTML is now inline in the Python code  
✅ **Simplified dependencies**: Only FastAPI and essentials
✅ **Added error handling**: Better exception handling
✅ **Fixed handler export**: Proper Vercel compatibility

### 6. **Test the Fixed Version**

Try these URLs after redeployment:
- `https://your-app.vercel.app/` - Main interface
- `https://your-app.vercel.app/health` - Health check
- `https://your-app.vercel.app/docs` - API documentation

### 7. **If Still Failing**

#### Check Vercel Logs:
1. Go to [vercel.com](https://vercel.com)
2. Open your project
3. Click on the failing deployment
4. Go to "Functions" tab
5. Click on your function to see logs

#### Manual Debug:
Create a minimal test function:
```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World", "status": "working"}

handler = app
```

### 8. **Environment Debugging**

Add debug endpoint:
```python
@app.get("/debug")
def debug_info():
    import sys, os
    return {
        "python_version": sys.version,
        "cwd": os.getcwd(),
        "files": os.listdir("."),
        "env_vars": dict(os.environ)
    }
```

### 9. **Monitoring & Alerts**

Set up monitoring:
- Enable Vercel Analytics
- Set up error notifications
- Monitor function duration and memory usage

The current version should work now. If you still see errors, check the Vercel function logs for specific error details!
