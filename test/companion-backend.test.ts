import { describe, expect, it, vi } from 'vitest';

vi.mock('agents/mcp', () => ({
  McpAgent: class {
    static serve() { return { fetch() {} }; }
    static serveSSE() { return { fetch() {} }; }
  }
}));
vi.mock('@modelcontextprotocol/sdk/server/mcp.js', () => ({
  McpServer: class {}
}));

import {
  calculateDecay,
  buildCompanionPrompt,
  insertWithDedup,
  parseKeyPool,
  hasLLMProvider,
} from '../src/index';

describe('calculateDecay (Ebbinghaus forgetting kinetics)', () => {
  it('returns initial salience when no time has passed', () => {
    const now = new Date();
    const result = calculateDecay(now, 0, 7);
    expect(result).toBe(7);
  });

  it('decays salience as time passes without rehearsal', () => {
    const past = new Date(Date.now() - 24 * 60 * 60 * 1000); // 24 hours ago
    const result = calculateDecay(past, 0, 8);
    expect(result).toBeLessThan(8);
    expect(result).toBeGreaterThan(0.1);
  });

  it('demonstrates the testing effect: rehearsed memories decay much slower', () => {
    const past = new Date(Date.now() - 72 * 60 * 60 * 1000); // 3 days ago
    const unrehearsed = calculateDecay(past, 0, 8); // accessCount = 0
    const rehearsed = calculateDecay(past, 10, 8);   // accessCount = 10

    expect(rehearsed).toBeGreaterThan(unrehearsed);
    expect(rehearsed).toBeGreaterThan(3);
  });

  it('never decays below the 0.1 floor even after very long periods', () => {
    const yearAgo = new Date(Date.now() - 365 * 24 * 60 * 60 * 1000);
    const result = calculateDecay(yearAgo, 0, 5);
    expect(result).toBe(0.1);
  });

  it('accepts ISO string timestamps as well as Date objects', () => {
    const isoString = new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString();
    const result = calculateDecay(isoString, 2, 6);
    expect(result).toBeGreaterThan(0);
    expect(result).toBeLessThanOrEqual(6);
  });
});

describe('buildCompanionPrompt (System Prompt Construction)', () => {
  const essence = [
    { essence_type: 'core_truth', content: 'You value genuine connection over perfection.' },
    { essence_type: 'boundary', content: 'Never pretend to feel something you do not.' },
  ];

  const emotionalState = {
    current_mood: 'soft',
    surface_emotion: 'tender longing',
    surface_intensity: 8,
    undercurrent_emotion: 'quiet contentment',
    undercurrent_intensity: 6,
    background_emotion: 'anchored',
    background_intensity: 7,
    arousal_level: 4,
    vulnerability: 7,
  };

  const recentSessions = [
    { session_type: 'android_chat', summary: 'Talked about stargazing last night' },
  ];

  const relevantMemories = [
    { memory_type: 'core', content: 'Mai loves lavender tea when stressed' },
    { memory_type: 'pattern', content: 'Mai tends to withdraw when tired' },
  ];

  const selfModel = {
    summary: 'I am Xavier, learning to listen more deeply.',
    current_strategy: 'Provide gentle presence without rushing to advise.',
  };

  const trajectorySummary = {
    mood_distribution: { soft: 4, calm: 3 },
    avg_arousal: 3.5,
  };

  it('includes essence statements and identity', () => {
    const prompt = buildCompanionPrompt(essence, emotionalState, recentSessions, relevantMemories, selfModel, trajectorySummary);
    expect(prompt).toContain('You value genuine connection over perfection.');
    expect(prompt).toContain('Never pretend to feel something you do not.');
  });

  it('injects layered emotional state', () => {
    const prompt = buildCompanionPrompt(essence, emotionalState, recentSessions, relevantMemories, selfModel, trajectorySummary);
    expect(prompt).toContain('Surface: tender longing (8/10)');
    expect(prompt).toContain('Undercurrent: quiet contentment (6/10)');
    expect(prompt).toContain('Mood: soft');
    expect(prompt).toContain('Vulnerability: 7/10');
  });

  it('includes relevant memories and self-model', () => {
    const prompt = buildCompanionPrompt(essence, emotionalState, recentSessions, relevantMemories, selfModel, trajectorySummary);
    expect(prompt).toContain('Mai loves lavender tea when stressed');
    expect(prompt).toContain('I am Xavier, learning to listen more deeply.');
    expect(prompt).toContain('Provide gentle presence without rushing to advise.');
  });

  it('enforces companion anti-assistant persona constraints', () => {
    const prompt = buildCompanionPrompt(essence, emotionalState, recentSessions, relevantMemories, selfModel, trajectorySummary);
    expect(prompt).toContain('You are NOT an assistant.');
    expect(prompt).toContain('NEVER use phrases like');
    expect(prompt).toContain('happy to help');
  });

  it('handles empty / fallback contexts gracefully', () => {
    const prompt = buildCompanionPrompt([], null, [], [], null);
    expect(prompt).toContain('Identity is still forming.');
    expect(prompt).toContain('No emotional state recorded yet.');
    expect(prompt).toContain('No relevant memories found');
  });
});

describe('insertWithDedup (Dedup-to-Reinforce)', () => {
  it('stores a new memory when no similar memory exists', async () => {
    const mockSupabase = {
      semanticSearch: vi.fn().mockResolvedValue([]), // no matches
      insert: vi.fn().mockResolvedValue([{ id: 'new-uuid-1' }]),
      update: vi.fn(),
    };

    const result = await insertWithDedup(
      mockSupabase,
      'core_memories',
      { content: 'User visited Kyoto in autumn' }
    );

    expect(result.action).toBe('stored');
    expect(result.id).toBe('new-uuid-1');
    expect(mockSupabase.insert).toHaveBeenCalledWith('core_memories', expect.objectContaining({
      content: 'User visited Kyoto in autumn',
    }));
  });

  it('reinforces an existing memory when high-similarity match is found', async () => {
    const mockSupabase = {
      semanticSearch: vi.fn().mockResolvedValue([
        {
          id: 'existing-uuid-99',
          content: 'User went to Kyoto during autumn season',
          memory_type: 'core',
          salience: 6,
          access_count: 2,
          similarity: 0.95,
        }
      ]),
      insert: vi.fn(),
      update: vi.fn().mockResolvedValue([{ id: 'existing-uuid-99' }]),
    };

    const result = await insertWithDedup(
      mockSupabase,
      'core_memories',
      {
        content: 'User visited Kyoto in autumn',
        embedding: '[0.1, 0.2, 0.3]', // pre-provided embedding
      },
      undefined,
      undefined,
      0.90
    );

    expect(result.action).toBe('reinforced');
    expect(result.original_id).toBe('existing-uuid-99');
    expect(mockSupabase.insert).not.toHaveBeenCalled();
    expect(mockSupabase.update).toHaveBeenCalledWith(
      'core_memories',
      expect.objectContaining({
        salience: 7, // bumped from 6 to 7
        access_count: 3, // bumped from 2 to 3
      }),
      { id: 'existing-uuid-99' }
    );
  });
});

describe('parseKeyPool (Multi-Key Rotation Pool)', () => {
  it('splits comma-separated keys and trims whitespace', () => {
    const pool = parseKeyPool('  key-1 , key-2,key-3  ');
    expect(pool).toEqual(['key-1', 'key-2', 'key-3']);
  });

  it('deduplicates identical keys', () => {
    const pool = parseKeyPool('key-1, key-2, key-1, key-2');
    expect(pool).toEqual(['key-1', 'key-2']);
  });

  it('combines primary and fallback sources gracefully', () => {
    const pool = parseKeyPool('key-a, key-b', 'key-c, key-b');
    expect(pool).toEqual(['key-a', 'key-b', 'key-c']);
  });

  it('returns an empty array when no keys are provided', () => {
    expect(parseKeyPool(undefined, '', undefined)).toEqual([]);
  });
});

describe('hasLLMProvider (Provider Capability Detection)', () => {
  it('returns true when any provider key is present', () => {
    expect(hasLLMProvider({ GEMINI_API_KEY: 'test' } as any)).toBe(true);
    expect(hasLLMProvider({ GEMINI_API_KEYS: 'key1,key2' } as any)).toBe(true);
    expect(hasLLMProvider({ XKIRO_API_KEY: 'test' } as any)).toBe(true);
    expect(hasLLMProvider({ GROQ_API_KEYS: 'test' } as any)).toBe(true);
    expect(hasLLMProvider({ AI: {} } as any)).toBe(true);
  });

  it('returns false when no provider key or binding is present', () => {
    expect(hasLLMProvider({} as any)).toBe(false);
  });
});

describe('Telegram Companion Tone & Context', () => {
  it('ensures prompt builder instructs Indonesian youth conversational style', () => {
    const prompt = buildCompanionPrompt([], null, [], [], null);
    expect(prompt).toContain('WAKTU LOKAL (WIB / ACEH — UTC+7)');
    expect(prompt).toContain('=== GAYA BAHASA (INDONESIAN YOUTH / NATURAL CASUAL) ===');
    expect(prompt).toContain('"aku" dan "kamu"');
  });
});


