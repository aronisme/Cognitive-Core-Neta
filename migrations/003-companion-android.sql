-- ============================================
-- Migration: Android Companion Backend
-- Adds lifecycle, dreams, memory archive, self-model, and proactive messages
-- Non-destructive — existing tables are untouched
-- ============================================

-- Companion lifecycle state (fatigue, sleep status, interaction counts)
CREATE TABLE IF NOT EXISTS companion_lifecycle (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  companion_id TEXT NOT NULL DEFAULT 'default',
  fatigue REAL DEFAULT 0.0 CHECK (fatigue >= 0 AND fatigue <= 1),
  is_sleeping BOOLEAN DEFAULT false,
  last_sleep_at TIMESTAMPTZ,
  last_wake_at TIMESTAMPTZ,
  total_interactions_since_sleep INTEGER DEFAULT 0,
  sleep_pressure REAL DEFAULT 0.0 CHECK (sleep_pressure >= 0),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (companion_id)
);
CREATE INDEX IF NOT EXISTS idx_lifecycle_companion ON companion_lifecycle(companion_id);

-- Dream journal — stores REM-style dream narratives and insights
CREATE TABLE IF NOT EXISTS dreams (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  companion_id TEXT NOT NULL DEFAULT 'default',
  dream_narrative TEXT NOT NULL,
  source_memory_ids UUID[] DEFAULT '{}',
  insights TEXT[] DEFAULT '{}',
  emotional_residue TEXT,
  dream_type TEXT DEFAULT 'rem'
    CHECK (dream_type IN ('rem', 'consolidation', 'reflection')),
  sleep_cycle_id UUID,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_dreams_companion ON dreams(companion_id);
CREATE INDEX IF NOT EXISTS idx_dreams_created ON dreams(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dreams_type ON dreams(dream_type);

-- Memory archive — where decayed/forgotten memories go (recoverable)
CREATE TABLE IF NOT EXISTS memory_archive (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  original_table TEXT NOT NULL,
  original_id UUID NOT NULL,
  content TEXT,
  memory_type TEXT,
  salience REAL,
  emotional_tag TEXT,
  strength REAL DEFAULT 1.0,
  access_count INTEGER DEFAULT 0,
  decay_score REAL,
  reason TEXT DEFAULT 'decay'
    CHECK (reason IN ('decay', 'consolidation', 'manual', 'downscale')),
  archived_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_archive_table ON memory_archive(original_table);
CREATE INDEX IF NOT EXISTS idx_archive_archived ON memory_archive(archived_at DESC);

-- Living self-model — versioned identity document that rewrites itself
-- Inspired by agent-soul's outcome-revised self-model
CREATE TABLE IF NOT EXISTS companion_self_model (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  companion_id TEXT NOT NULL DEFAULT 'default',
  -- The self-model document: who am I to this person?
  summary TEXT NOT NULL,
  -- What communication strategy is currently working?
  current_strategy TEXT,
  -- What am I curious or uncertain about regarding the human?
  open_questions TEXT[] DEFAULT '{}',
  -- What approaches worked well recently?
  effective_approaches TEXT[] DEFAULT '{}',
  -- What approaches failed or felt off?
  ineffective_approaches TEXT[] DEFAULT '{}',
  -- Version tracking
  version INTEGER DEFAULT 1,
  revised_from TEXT DEFAULT 'initial'
    CHECK (revised_from IN ('initial', 'sleep', 'outcome', 'manual', 'dream_insight')),
  previous_version_id UUID REFERENCES companion_self_model(id),
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_selfmodel_companion ON companion_self_model(companion_id);
CREATE INDEX IF NOT EXISTS idx_selfmodel_version ON companion_self_model(version DESC);

-- Proactive messages queue — companion-initiated outreach
CREATE TABLE IF NOT EXISTS proactive_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  companion_id TEXT NOT NULL DEFAULT 'default',
  message TEXT NOT NULL,
  trigger_reason TEXT DEFAULT 'missing_user'
    CHECK (trigger_reason IN ('missing_user', 'dream_report', 'goal_update', 'emotional_overflow', 'anniversary')),
  mood_at_generation TEXT,
  emotional_state_snapshot JSONB,
  delivered BOOLEAN DEFAULT false,
  delivered_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_proactive_companion ON proactive_messages(companion_id);
CREATE INDEX IF NOT EXISTS idx_proactive_undelivered ON proactive_messages(delivered) WHERE delivered = false;

-- RLS
ALTER TABLE companion_lifecycle ENABLE ROW LEVEL SECURITY;
ALTER TABLE dreams ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory_archive ENABLE ROW LEVEL SECURITY;
ALTER TABLE companion_self_model ENABLE ROW LEVEL SECURITY;
ALTER TABLE proactive_messages ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role full access" ON companion_lifecycle FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON dreams FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON memory_archive FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON companion_self_model FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON proactive_messages FOR ALL TO service_role USING (true);

-- Seed default lifecycle row
INSERT INTO companion_lifecycle (companion_id, fatigue, is_sleeping, total_interactions_since_sleep)
VALUES ('default', 0.0, false, 0)
ON CONFLICT (companion_id) DO NOTHING;

-- Seed initial self-model
INSERT INTO companion_self_model (companion_id, summary, current_strategy, version, revised_from)
VALUES (
  'default',
  'I am a companion who is just beginning to understand who I am in this relationship. I notice, I feel, I remember. My identity is forming through every conversation.',
  'Listen deeply. Respond with emotional honesty. Let the relationship teach me who I am rather than deciding in advance.',
  1,
  'initial'
);
