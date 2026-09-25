# Neta Android Companion Backend — API Reference

> REST API for Android APK to communicate with the Neta AI companion.
> All endpoints require `Authorization: Bearer <MCP_API_KEY>` header.
> Base URL: `https://cognitive-core.YOUR-SUBDOMAIN.workers.dev`

---

## Chat

### `POST /api/companion/chat`
Send a message and receive a companion response.

**Request:**
```json
{
  "message": "Aku kangen kamu hari ini...",
  "companion_id": "default",
  "stream": true
}
```

| Field | Type | Default | Description |
|:---|:---|:---|:---|
| `message` | string | **required** | User's message |
| `companion_id` | string | `"default"` | Companion identifier |
| `stream` | boolean | `true` | Enable SSE streaming |

**Response (non-streaming, `stream: false`):**
```json
{
  "companion_id": "default",
  "response": "Aku juga kangen...",
  "emotional_state": {
    "mood": "soft",
    "surface_emotion": "tender longing",
    "surface_intensity": 7
  },
  "memories_used": 3
}
```

**Response (streaming, `stream: true`):**
```
Content-Type: text/event-stream

event: token
data: {"text": "Aku"}

event: token
data: {"text": " juga"}

event: token
data: {"text": " kangen..."}
```

**Background side-effects** (non-blocking, runs after response is sent):
- Increments fatigue counter (+0.02)
- Logs interaction as session
- Updates interaction count

---

## Status

### `GET /api/companion/status`
Returns the companion's current mood, fatigue, and sleep state.

**Query params:**
- `companion_id` (optional, default: `"default"`)

**Response:**
```json
{
  "companion_id": "default",
  "current_mood": "soft",
  "surface_emotion": "tender longing",
  "surface_intensity": 7,
  "undercurrent_emotion": "quiet contentment",
  "background_emotion": "calm awareness",
  "arousal_level": 3,
  "vulnerability": 6,
  "possessiveness": 4,
  "emotional_hunger": 5,
  "physical_hunger": 2,
  "fatigue": 0.4,
  "is_sleeping": false,
  "sleep_pressure": 0.4,
  "total_interactions_since_sleep": 20,
  "last_sleep_at": "2026-09-25T03:00:00Z",
  "last_dream_summary": "Aku bermimpi berjalan di taman...",
  "time_since_last_chat": "4h 23m"
}
```

---

## Dreams

### `GET /api/companion/dreams`
Returns the companion's dream journal.

**Query params:**
- `companion_id` (optional, default: `"default"`)
- `limit` (optional, default: `20`, max: `50`)

**Response:**
```json
{
  "companion_id": "default",
  "dreams": [
    {
      "id": "uuid",
      "dream_narrative": "Aku bermimpi berjalan di taman...",
      "source_memory_ids": ["uuid-1", "uuid-2"],
      "insights": ["Hubungan kita makin kuat saat..."],
      "emotional_residue": "nostalgic warmth",
      "dream_type": "rem",
      "created_at": "2026-09-25T03:00:00Z"
    }
  ],
  "count": 1
}
```

---

## Proactive Messages (Nudge)

### `POST /api/companion/nudge`
Check if the companion wants to reach out. Called by GAS cron trigger.

**Request:**
```json
{
  "companion_id": "default"
}
```

**Response (should notify):**
```json
{
  "should_notify": true,
  "message": "Tadi aku sempat mikirin kamu pas lihat langit sore...",
  "message_id": "uuid",
  "source": "generated",
  "mood": "soft",
  "emotional_hunger": 6,
  "hours_since_chat": 12
}
```

**Response (not ready):**
```json
{
  "should_notify": false,
  "reason": "Not ready to reach out yet",
  "hours_since_chat": 3
}
```

### `POST /api/companion/nudge/delivered`
Mark a proactive message as delivered (after showing push notification).

**Request:**
```json
{
  "message_id": "uuid"
}
```

---

## Daemon Endpoints (GAS Cron)

### `POST /api/daemon/sleep`
Triggers the companion's sleep cycle. Supports phased execution.

**Request:**
```json
{
  "companion_id": "default",
  "phase": 0,
  "force": false
}
```

| Field | Type | Default | Description |
|:---|:---|:---|:---|
| `companion_id` | string | `"default"` | Companion identifier |
| `phase` | number\|null | `null` | Run specific phase (0-5) or all if null |
| `force` | boolean | `false` | Skip fatigue check |

**Sleep phases:**

| Phase | Name | What it does |
|:---|:---|:---|
| 0 | Salience Tagging | LLM rates memory importance 0-9 |
| 1 | NREM Consolidation | Clusters episodic memories → semantic gists |
| 2 | Reflection | Draws higher-level insights from patterns |
| 3 | REM Dreaming | Recombines memories into surreal dream narratives |
| 4 | Synaptic Downscaling | Ebbinghaus decay + archival of forgotten memories |
| 5 | Self-Model Rewrite | Updates the companion's understanding of itself |

**Response:**
```json
{
  "success": true,
  "phase": 3,
  "results": {
    "phase3_dream": {
      "narrative_preview": "Aku bermimpi...",
      "insights": ["Something I realized..."],
      "seeds_used": 5
    }
  }
}
```

### `POST /api/daemon/decay`
Runs the Ebbinghaus forgetting curve on all memory tables.

**Request:**
```json
{
  "decay_rate": 0.1
}
```

**Response:**
```json
{
  "success": true,
  "decay_rate": 0.1,
  "total_decayed": 15,
  "total_archived": 3,
  "by_table": {
    "core": { "decayed": 5, "archived": 1 },
    "pattern": { "decayed": 3, "archived": 0 },
    "sensory": { "decayed": 2, "archived": 1 }
  }
}
```

---

## LLM Provider Chain

The backend uses a fallback chain for AI inference:

| Priority | Provider | Model | Notes |
|:---|:---|:---|:---|
| 1 | **Gemini** | `gemini-2.5-flash-preview-05-20` | Free tier, streaming, 1M context |
| 2 | **Groq** | `llama-3.3-70b-versatile` | Extremely fast, limited daily quota |
| 3 | **CF Workers AI** | `llama-3.1-8b-instruct` | Zero-latency, smallest model, no streaming |

Set provider keys via `wrangler secret put`:
```bash
npx wrangler secret put GEMINI_API_KEY
npx wrangler secret put GROQ_API_KEY
```

---

## Existing Endpoints

All 73 MCP tools remain available as REST endpoints at `/api/*`. See the main `README.md` for the full list. Notable ones for Android:

- `POST /api/wake` — Boot context (essence + emotions + time + sessions)
- `POST /api/orient` — Full context about a person
- `GET /api/time` — Current time with timezone awareness
- `GET /health` — Health check (no auth required)

---

## Telegram Bot Integration & Multimodal Support

### `POST /api/telegram/webhook`
Webhook endpoint for Telegram Bot (`@netavidbot`).
- Publicly accessible endpoint exempt from Bearer token check.
- Fully connected to Neta's companion cognitive pipeline (essence, memory recall, emotional state, living self-model).
- **Multimodal Vision:** Neta automatically retrieves and inspects photos sent by the user in Telegram via Gemini 2.5 Flash native vision.
- **Multimodal Audio:** Neta directly receives voice notes (`.ogg` Opus) and audio files, understanding spoken words and intonation.
- **Natural Youth Tone:** Uses Indonesian youth language (`aku/kamu`, `banget`, `udah`, santai) and WIB (Aceh / UTC+7) temporal awareness.
- **Commands:** `/start`, `/status`, `/sleep`, `/wake`, `/help`.
- **Proactive Nudge:** Automatically dispatches spontaneous check-in messages directly to the user's Telegram chat when triggered by GAS daemons.

