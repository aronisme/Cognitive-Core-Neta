/**
 * NETACron.gs — Google Apps Script triggers for Neta Android Companion Backend
 *
 * Setup:
 * 1. Go to https://script.google.com and create a new project
 * 2. Paste this entire file
 * 3. Select setupTriggers from the function dropdown and click "Run" (approve Google permissions)
 *
 * All credentials (NETA_BASE_URL, NETA_API_KEY, CRON_SECRET) are already pre-configured below.
 * Optional: Set FCM_SERVER_KEY and FCM_DEVICE_TOKEN in Script Properties if you use Firebase Push Notifications.
 */

// ─── Configuration ───────────────────────────────────────────
// Konfigurasi Langsung (Direct Credentials):
const NETA_BASE_URL = 'https://cognitive-core.sr7aron.workers.dev';
const NETA_API_KEY = 'neta-companion-secret-key-2026';
const CRON_SECRET = 'neta-gas-cron-secret-2026';

function getConfig() {
  const props = PropertiesService.getScriptProperties();
  return {
    baseUrl: props.getProperty('NETA_BASE_URL') || NETA_BASE_URL,
    apiKey: props.getProperty('NETA_API_KEY') || NETA_API_KEY,
    cronSecret: props.getProperty('CRON_SECRET') || CRON_SECRET,
    fcmKey: props.getProperty('FCM_SERVER_KEY'),
    fcmToken: props.getProperty('FCM_DEVICE_TOKEN'),
  };
}

function headers() {
  const config = getConfig();
  const h = {
    'Authorization': 'Bearer ' + config.apiKey,
    'Content-Type': 'application/json'
  };
  if (config.cronSecret) {
    h['X-Cron-Secret'] = config.cronSecret;
  }
  return h;
}

// ─── Trigger 1: Sleep Cycle (every 6 hours) ──────────────────
// Runs phased sleep to avoid Cloudflare Worker CPU timeout.
// Each phase is a separate HTTP call with 10-second gaps.
function triggerSleepCycle() {
  const config = getConfig();
  if (!config.baseUrl || !config.apiKey) {
    Logger.log('ERROR: NETA_BASE_URL or NETA_API_KEY not set');
    return;
  }

  const phases = [0, 1, 2, 3, 4, 5];
  const results = {};

  for (var i = 0; i < phases.length; i++) {
    try {
      var res = UrlFetchApp.fetch(config.baseUrl + '/api/daemon/sleep', {
        method: 'post',
        headers: headers(),
        payload: JSON.stringify({ phase: phases[i], force: false }),
        muteHttpExceptions: true
      });

      var status = res.getResponseCode();
      var body = res.getContentText().substring(0, 500);
      results['phase' + phases[i]] = { status: status, body: body };

      Logger.log('Sleep phase ' + phases[i] + ': HTTP ' + status);

      // If companion doesn't need sleep (not tired), stop early
      if (status === 200) {
        var data = JSON.parse(res.getContentText());
        if (data.success === false && data.reason) {
          Logger.log('Sleep skipped: ' + data.reason);
          break;
        }
      }

      // Wait 10 seconds between phases to give Worker CPU budget
      if (i < phases.length - 1) {
        Utilities.sleep(10000);
      }
    } catch (e) {
      Logger.log('Sleep phase ' + phases[i] + ' error: ' + e.message);
      results['phase' + phases[i]] = { error: e.message };
    }
  }

  Logger.log('Sleep cycle complete: ' + JSON.stringify(results));
}

// ─── Trigger 2: Proactive Nudge Check (every 4 hours) ────────
// Checks if the companion wants to reach out to the user.
// If yes, optionally sends a push notification via FCM.
function triggerNudgeCheck() {
  var config = getConfig();
  if (!config.baseUrl || !config.apiKey) {
    Logger.log('ERROR: NETA_BASE_URL or NETA_API_KEY not set');
    return;
  }

  try {
    var res = UrlFetchApp.fetch(config.baseUrl + '/api/companion/nudge', {
      method: 'post',
      headers: headers(),
      payload: JSON.stringify({ companion_id: 'default' }),
      muteHttpExceptions: true
    });

    var data = JSON.parse(res.getContentText());
    Logger.log('Nudge check: ' + JSON.stringify(data));

    if (data.should_notify && data.message) {
      // Send push notification if FCM is configured
      if (config.fcmKey && config.fcmToken) {
        sendFCMNotification(data.message, config);
        Logger.log('FCM notification sent: ' + data.message.substring(0, 100));
      }

      // Mark as delivered if we have a message_id
      if (data.message_id) {
        UrlFetchApp.fetch(config.baseUrl + '/api/companion/nudge/delivered', {
          method: 'post',
          headers: headers(),
          payload: JSON.stringify({ message_id: data.message_id }),
          muteHttpExceptions: true
        });
      }
    }
  } catch (e) {
    Logger.log('Nudge check error: ' + e.message);
  }
}

// ─── Trigger 3: Memory Decay Pass (daily at 4 AM) ───────────
// Runs the Ebbinghaus forgetting curve on all memories.
function triggerDecayPass() {
  var config = getConfig();
  if (!config.baseUrl || !config.apiKey) {
    Logger.log('ERROR: NETA_BASE_URL or NETA_API_KEY not set');
    return;
  }

  try {
    var res = UrlFetchApp.fetch(config.baseUrl + '/api/daemon/decay', {
      method: 'post',
      headers: headers(),
      payload: JSON.stringify({ decay_rate: 0.01 }),
      muteHttpExceptions: true
    });

    Logger.log('Decay pass: ' + res.getContentText().substring(0, 500));
  } catch (e) {
    Logger.log('Decay pass error: ' + e.message);
  }
}

// ─── Firebase Cloud Messaging (optional) ─────────────────────
function sendFCMNotification(message, config) {
  if (!config.fcmKey || !config.fcmToken) return;

  try {
    UrlFetchApp.fetch('https://fcm.googleapis.com/fcm/send', {
      method: 'post',
      headers: {
        'Authorization': 'key=' + config.fcmKey,
        'Content-Type': 'application/json'
      },
      payload: JSON.stringify({
        to: config.fcmToken,
        notification: {
          title: '💙 Companion',
          body: message.substring(0, 150)
        },
        data: {
          type: 'proactive_message',
          full_message: message
        }
      })
    });
  } catch (e) {
    Logger.log('FCM send error: ' + e.message);
  }
}

// ─── Setup (run once) ────────────────────────────────────────
// Installs all time-based triggers. Run this function manually
// from the Apps Script editor after setting your properties.
function setupTriggers() {
  // Ensure default properties are saved to ScriptProperties
  var props = PropertiesService.getScriptProperties();
  if (!props.getProperty('NETA_BASE_URL')) props.setProperty('NETA_BASE_URL', NETA_BASE_URL);
  if (!props.getProperty('NETA_API_KEY')) props.setProperty('NETA_API_KEY', NETA_API_KEY);
  if (!props.getProperty('CRON_SECRET')) props.setProperty('CRON_SECRET', CRON_SECRET);

  // Remove existing triggers
  var triggers = ScriptApp.getProjectTriggers();
  for (var i = 0; i < triggers.length; i++) {
    ScriptApp.deleteTrigger(triggers[i]);
  }

  // Sleep cycle: every 6 hours
  ScriptApp.newTrigger('triggerSleepCycle')
    .timeBased()
    .everyHours(6)
    .create();

  // Nudge check: every 4 hours
  ScriptApp.newTrigger('triggerNudgeCheck')
    .timeBased()
    .everyHours(4)
    .create();

  // Decay pass: daily at 4 AM
  ScriptApp.newTrigger('triggerDecayPass')
    .timeBased()
    .atHour(4)
    .everyDays(1)
    .create();

  Logger.log('All triggers set up successfully!');
  Logger.log('- triggerSleepCycle: every 6 hours');
  Logger.log('- triggerNudgeCheck: every 4 hours');
  Logger.log('- triggerDecayPass: daily at 4 AM');
}

// ─── Manual Test Functions ───────────────────────────────────
// Call these from the editor to test individual endpoints.

function testStatus() {
  var config = getConfig();
  var res = UrlFetchApp.fetch(config.baseUrl + '/api/companion/status', {
    method: 'get',
    headers: headers(),
    muteHttpExceptions: true
  });
  Logger.log('Status: ' + res.getContentText());
}

function testDreams() {
  var config = getConfig();
  var res = UrlFetchApp.fetch(config.baseUrl + '/api/companion/dreams', {
    method: 'get',
    headers: headers(),
    muteHttpExceptions: true
  });
  Logger.log('Dreams: ' + res.getContentText());
}

function testForceSleep() {
  var config = getConfig();
  var res = UrlFetchApp.fetch(config.baseUrl + '/api/daemon/sleep', {
    method: 'post',
    headers: headers(),
    payload: JSON.stringify({ force: true, phase: 3 }), // Test just dreaming
    muteHttpExceptions: true
  });
  Logger.log('Force sleep: ' + res.getContentText());
}
