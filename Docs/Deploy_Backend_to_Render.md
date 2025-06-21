# 🚀 Deploying Your NBA 2K25 eSports Backend to Render

This guide will walk you through deploying your Python backend (API, CLI, and prediction pipeline) to [Render](https://render.com/). Render is a great choice for hosting long-running Python services, APIs, and background jobs.

---

## 1. Prerequisites

- A free [Render account](https://dashboard.render.com/register)
- Your project code pushed to a GitHub repository (private or public)
- Supabase credentials (URL, API key) and any other secrets you need

---

## 2. Prepare Your Codebase

- Ensure your `requirements.txt` is up to date
- Make sure your API entry point (e.g., `backend/app/api.py`) exposes a web server (e.g., using FastAPI, Flask, or similar)
- If using a custom start command (like `python -m backend.app.api`), note it for Render setup
- Add a `.env` file (locally) for secrets, but **do not commit it**

---

## 3. Push to GitHub

1. Commit and push all your code to a GitHub repository
2. Make sure your main branch is up to date

---

## 4. Create a New Web Service on Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account and select your repository
4. Fill out the form:
   - **Name:** e.g., `nba2k25-backend`
   - **Branch:** `main` or your deployment branch
   - **Build Command:** (leave blank for Python)
   - **Start Command:**
     - If using FastAPI: `uvicorn backend.app.api:app --host 0.0.0.0 --port 10000`
     - If using Flask: `python backend/app/api.py`
     - Or your custom command
   - **Environment:** Python 3.8+ (set in Render settings)
   - **Instance Type:** Free or Starter (upgrade if you need more resources)
5. Click **"Create Web Service"**

---

## 5. Set Environment Variables

- In the Render dashboard, go to your service → **Environment** tab
- Add your Supabase URL, API key, and any other secrets as environment variables
- Example:
  - `SUPABASE_URL=https://your-project.supabase.co`
  - `SUPABASE_KEY=your-anon-or-service-key`

---

## 6. Deploy and Test

- Render will build and deploy your service automatically
- Once deployed, you’ll get a public URL (e.g., `https://nba2k25-backend.onrender.com`)
- Test your API endpoints using Postman, curl, or your browser

---

## 7. (Optional) Background Jobs & Pipeline

- For scheduled jobs (like daily model retraining), use Render’s **Background Worker** service
- Create a new Worker, point it to your CLI or pipeline script (e.g., `python -m backend.app.cli run-pipeline --train`)
- Use Render’s built-in scheduler to run jobs on a schedule

---

## 8. Next Steps

- Build your frontend (React/Next.js) and deploy it to Vercel
- Connect your frontend to the backend API URL from Render
- Monitor logs and performance in the Render dashboard

---

## 📝 Tips

- Keep secrets out of your codebase—use environment variables
- Use the free tier to start; upgrade if you need more resources
- Check Render’s docs for advanced features (custom domains, SSL, scaling)

---

**You’re ready to go!**

For more details, see: [Render Python Docs](https://render.com/docs/deploy-python)
