# Frontend
Start-Process powershell -ArgumentList "npm i; npm run dev" -WorkingDirectory "$PWD\Frontend"

# Backend
Start-Process powershell -ArgumentList "npm i; npm run dev:full" -WorkingDirectory "$PWD\Backend"

# Server
Start-Process powershell -ArgumentList "uv sync; uv run uvicorn src.app.main:app" -WorkingDirectory "$PWD\Server"