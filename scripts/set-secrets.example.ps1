# PowerShell template script to set Cloudflare Secrets for Neta Companion Backend
# Copy this to set-secrets.ps1 and fill in your actual API keys.
Write-Host "Setting Cloudflare Secrets for Cognitive Core (Neta)..." -ForegroundColor Cyan

$secrets = @{
    "SUPABASE_URL" = "https://your-project-id.supabase.co"
    "SUPABASE_SERVICE_KEY" = "your-supabase-service-key"
    "MCP_API_KEY" = "your-mcp-api-key"
    "CRON_SECRET" = "your-cron-secret"
    "GEMINI_API_KEYS" = "gemini-key-1,gemini-key-2"
    "XKIRO_API_KEYS" = "xkiro-key-1,xkiro-key-2"
    "GROQ_API_KEYS" = "groq-key-1,groq-key-2"
    "TELEGRAM_BOT_TOKEN" = "your-telegram-bot-token"
}

foreach ($key in $secrets.Keys) {
    Write-Host "Setting secret: $key ..." -ForegroundColor Yellow
    $val = $secrets[$key]
    $val | npx wrangler secret put $key
}

Write-Host "`nAll secrets successfully set! Deploying worker..." -ForegroundColor Green
npm run deploy
