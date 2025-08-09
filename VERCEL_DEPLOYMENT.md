# Vercel Deployment Guide for Stress Level Prediction App

## 🚀 Quick Deployment Steps

### 1. GitHub Repository Setup ✅
You've already pushed your repo to GitHub. Great!

### 2. Vercel Configuration Files Created ✅

I've created the following files for Vercel deployment:

- `vercel.json` - Vercel configuration
- `api/index.py` - Main FastAPI application (Vercel-compatible)  
- `api/demo.py` - Lightweight demo version
- `requirements.txt` - Python dependencies (with fixed versions)
- `public/index.html` - Frontend interface

### 3. Deploy to Vercel

#### Option A: Vercel Web Dashboard (Recommended)

1. **Go to** [vercel.com](https://vercel.com) and sign in with GitHub
2. **Click** "New Project"
3. **Import** your `stress-detection-app` repository
4. **Configure** project settings:
   - Framework Preset: `Other`
   - Root Directory: `./` (leave default)
   - Build Command: (leave empty)
   - Output Directory: `public` 
5. **Environment Variables** (if needed):
   - No special environment variables required for basic version
6. **Click** "Deploy"

#### Option B: Vercel CLI

```bash
# Install Vercel CLI
npm install -g vercel

# Navigate to your project directory
cd "C:\Users\Roy\Desktop\ML COURSE\app2-stress"

# Deploy
vercel

# Follow prompts:
# - Link to existing project or create new one
# - Select settings (use defaults)
```

### 4. Expected Deployment URLs

After deployment, you'll get URLs like:
- **Main App**: `https://your-app-name.vercel.app/`
- **API Health**: `https://your-app-name.vercel.app/health`
- **Demo Form**: `https://your-app-name.vercel.app/demo`
- **API Docs**: `https://your-app-name.vercel.app/docs`

## 📁 File Structure for Vercel

```
stress-detection-app/
├── api/
│   ├── index.py          # Main FastAPI app (full version)
│   └── demo.py           # Lightweight demo version
├── public/
│   └── index.html        # Frontend interface
├── models/               # ML models (may be too large for Vercel)
├── dataset/             # Training data
├── templates/           # HTML templates
├── vercel.json          # Vercel configuration
├── requirements.txt     # Python dependencies
└── README.md
```

## ⚠️ Important Considerations

### 1. File Size Limitations
- **Vercel Limit**: 100MB total deployment size
- **Your Models**: ~1.7MB (should be fine)
- **Solution**: Models are included in the package

### 2. Cold Start Performance  
- **First Request**: May take 5-15 seconds (Vercel cold start)
- **Subsequent Requests**: Much faster (1-2 seconds)
- **Solution**: Implement model caching and optimization

### 3. Memory Limitations
- **Vercel Limit**: 1GB RAM for Hobby plan
- **ML Models**: May use significant memory
- **Solution**: Lightweight models or model optimization

## 🛠️ Deployment Strategies

### Strategy 1: Full Model Deployment
- **Use**: `api/index.py` (includes all 15+ ML models)
- **Pros**: Complete functionality
- **Cons**: Larger size, slower cold starts
- **Best for**: Production use with all features

### Strategy 2: Demo Deployment
- **Use**: `api/demo.py` (rule-based predictions)
- **Pros**: Fast, lightweight, always works
- **Cons**: Simplified predictions
- **Best for**: Quick demonstration and testing

### Strategy 3: Hybrid Approach
- **Deploy**: Demo version first to test
- **Upgrade**: To full models once verified working
- **Benefits**: Incremental deployment with fallback

## 🔧 Troubleshooting Common Issues

### Issue 1: "Build Failed" 
```
Error: Could not install requirements
```
**Solution**: Check `requirements.txt` versions are compatible
```bash
# Test locally first:
pip install -r requirements.txt
```

### Issue 2: "Function Timeout"
```
Error: Function execution timed out
```
**Solution**: Optimize model loading or use caching
```python
# Add to your FastAPI app
from functools import lru_cache

@lru_cache()
def load_models():
    # Model loading code here
    pass
```

### Issue 3: "Models Not Found"
```
Error: models directory not found
```
**Solution**: Ensure models/ directory is in the repository and not in .gitignore

### Issue 4: "Import Error"
```
ImportError: No module named 'sklearn'
```
**Solution**: Verify requirements.txt has correct package names:
```txt
scikit-learn==1.3.2  # NOT sklearn
```

## 🚀 Post-Deployment Steps

### 1. Test Your Deployment
```bash
# Test health endpoint
curl https://your-app.vercel.app/health

# Test prediction endpoint
curl -X POST https://your-app.vercel.app/predict-simple \
  -F "age=30" \
  -F "sleep_duration=7" \
  -F "physical_activity_level=60" \
  -F "heart_rate=70" \
  -F "quality_of_sleep=7"
```

### 2. Monitor Performance
- Check Vercel dashboard for function logs
- Monitor response times and error rates
- Set up alerts for failures

### 3. Custom Domain (Optional)
- Add custom domain in Vercel dashboard
- Configure DNS settings
- Enable HTTPS (automatic with Vercel)

## 📊 Expected Performance

### Deployment Size
- **Demo Version**: ~10-20MB
- **Full Version**: ~50-100MB
- **With Models**: Should fit within Vercel limits

### Response Times
- **Cold Start**: 5-15 seconds (first request)
- **Warm Requests**: 1-3 seconds
- **API Endpoints**: < 1 second

### Scalability
- **Concurrent Users**: Vercel handles automatically
- **Geographic Distribution**: Global edge network
- **Uptime**: 99.9% SLA

## 🎯 Next Steps After Deployment

1. **Verify** deployment works with `/health` endpoint
2. **Test** prediction functionality with `/demo`
3. **Share** your live URL for testing
4. **Monitor** performance and optimize as needed
5. **Scale** up to full model version if demo works

## 🔗 Useful Links

- [Vercel Python Runtime](https://vercel.com/docs/functions/serverless-functions/runtimes/python)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/vercel/)
- [Vercel CLI Documentation](https://vercel.com/docs/cli)

---

Your app should be ready to deploy! Start with the demo version to verify everything works, then upgrade to the full model version. Let me know if you encounter any issues during deployment!
