// background.js - Service worker: extract video configs, handle downloads

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'updateBadge') {
    const tabId = sender.tab?.id;
    if (tabId) {
      const text = request.count > 0 ? String(request.count) : '';
      chrome.action.setBadgeText({ text, tabId });
      chrome.action.setBadgeBackgroundColor({ color: '#1a9fbf', tabId });
    }
    return false;
  }

  if (request.action === 'extractVideosFromTab') {
    extractVideosFromTab(request.tabId)
      .then((videos) => sendResponse({ success: true, videos }))
      .catch((err) => sendResponse({ success: false, error: err.message }));
    return true;
  }

  if (request.action === 'probeMuxRenditions') {
    probeMuxRenditions(request.playbackId)
      .then((renditions) => sendResponse({ success: true, renditions }))
      .catch((err) => sendResponse({ success: false, error: err.message }));
    return true;
  }

  if (request.action === 'downloadVideo') {
    downloadVideo(request.url, request.filename)
      .then((result) => sendResponse({ success: true, downloadId: result }))
      .catch((err) => sendResponse({ success: false, error: err.message }));
    return true;
  }
});

// ---------------------------------------------------------------------------
// Execute a script inside every frame on the tab to detect and extract videos
// ---------------------------------------------------------------------------
async function extractVideosFromTab(tabId) {
  try {
    const results = await chrome.scripting.executeScript({
      target: { tabId, allFrames: true },
      world: 'MAIN',
      func: extractVideosInFrame,
    });

    const videos = [];
    for (const r of results) {
      if (r.result && Array.isArray(r.result)) {
        videos.push(...r.result);
      }
    }
    return videos;
  } catch (err) {
    console.error('executeScript failed:', err);
    return [];
  }
}

// This function is serialized and injected into every frame on the page.
// It runs in the MAIN world so it can access page JS variables and make
// same-origin fetches with proper cookies/referer.
async function extractVideosInFrame() {
  const results = [];

  // --- Vimeo detection (runs only inside Vimeo player iframes) ---
  if (location.hostname.includes('player.vimeo.com')) {
    const idMatch = location.pathname.match(/\/video\/(\d+)/);
    if (idMatch) {
      const videoId = idMatch[1];
      const config = await extractVimeoConfig(videoId);
      if (config) {
        results.push({ type: 'vimeo', videoId, config });
      }
    }
  }

  // --- Mux detection (runs in main page or any frame with mux-player) ---
  const seen = new Set();

  document.querySelectorAll('mux-player[playback-id]').forEach((player) => {
    const playbackId = player.getAttribute('playback-id');
    if (!playbackId || seen.has(playbackId)) return;
    seen.add(playbackId);

    const title = player.getAttribute('metadata-video-title') || '';
    results.push({ type: 'mux', playbackId, title });
  });

  // Fallback: mux-video without mux-player wrapper
  document.querySelectorAll('mux-video[src*="stream.mux.com"]').forEach((el) => {
    const src = el.getAttribute('src') || '';
    const match = src.match(/stream\.mux\.com\/([^/.]+)/);
    if (!match) return;
    const playbackId = match[1];
    if (seen.has(playbackId)) return;
    seen.add(playbackId);

    const player = el.closest('mux-player');
    const title = player?.getAttribute('metadata-video-title') || '';
    results.push({ type: 'mux', playbackId, title });
  });

  return results.length > 0 ? results : null;

  // --- Vimeo config extraction helpers ---
  async function extractVimeoConfig(videoId) {
    // Strategy 1: Known global JS variables
    for (const key of ['playerConfig', '__playerConfig', '__vimeo_config']) {
      const val = window[key];
      if (val?.request?.files?.progressive) {
        return buildVimeoResult(val);
      }
    }

    // Strategy 2: Parse inline <script> tags
    for (const s of document.querySelectorAll('script:not([src])')) {
      const text = s.textContent;
      if (!text || text.length < 100) continue;
      if (!text.includes('progressive') && !text.includes('playerConfig')) continue;

      for (const pat of [
        /playerConfig\s*=\s*(\{[\s\S]+?\})\s*;/,
        /var\s+config\s*=\s*(\{[\s\S]+?\})\s*;/,
        /window\.vimeo\.clip_page_config\s*=\s*(\{[\s\S]+?\});/,
      ]) {
        const m = text.match(pat);
        if (m) {
          try {
            const data = JSON.parse(m[1]);
            if (data.request?.files?.progressive) return buildVimeoResult(data);
          } catch (e) { /* ignore */ }
        }
      }
    }

    // Strategy 3: Same-origin fetch for /config
    try {
      const h = new URLSearchParams(location.search).get('h');
      let url = `/video/${videoId}/config`;
      if (h) url += `?h=${h}`;
      const resp = await fetch(url, { credentials: 'include' });
      if (resp.ok) {
        const data = await resp.json();
        if (data.request?.files?.progressive) return buildVimeoResult(data);
      }
    } catch (e) { /* ignore */ }

    // Strategy 4: <video> element src fallback
    const videoEl = document.querySelector('video');
    if (videoEl) {
      const src = videoEl.src || videoEl.querySelector('source')?.src;
      if (src && src.startsWith('http')) {
        return {
          title: document.title || `Vimeo ${videoId}`,
          duration: videoEl.duration || 0,
          progressive: [{
            url: src,
            quality: 'auto',
            width: videoEl.videoWidth || 0,
            height: videoEl.videoHeight || 0,
          }],
        };
      }
    }

    return null;
  }

  function buildVimeoResult(data) {
    const video = data.video || {};
    return {
      title: video.title || document.title || 'Untitled',
      duration: video.duration || 0,
      progressive: (data.request?.files?.progressive || [])
        .map((p) => ({
          url: p.url,
          quality: p.quality || (p.height ? p.height + 'p' : 'unknown'),
          width: p.width || 0,
          height: p.height || 0,
        }))
        .sort((a, b) => b.height - a.height),
    };
  }
}

// ---------------------------------------------------------------------------
// Probe Mux static MP4 renditions (high/medium/low)
// ---------------------------------------------------------------------------
const MUX_QUALITIES = [
  { key: 'high', label: '1080p (high)', height: 1080 },
  { key: 'medium', label: '720p (medium)', height: 720 },
  { key: 'low', label: '480p (low)', height: 480 },
];

async function probeMuxRenditions(playbackId) {
  const available = [];

  for (const q of MUX_QUALITIES) {
    const url = `https://stream.mux.com/${playbackId}/${q.key}.mp4`;
    try {
      const resp = await fetch(url, { method: 'HEAD' });
      if (resp.ok) {
        available.push({
          url,
          quality: q.label,
          height: q.height,
        });
      }
    } catch (e) { /* not available */ }
  }

  return available;
}

// ---------------------------------------------------------------------------
// Download
// ---------------------------------------------------------------------------
async function downloadVideo(url, filename) {
  return new Promise((resolve, reject) => {
    chrome.downloads.download(
      { url, filename, saveAs: false },
      (downloadId) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else {
          resolve(downloadId);
        }
      }
    );
  });
}
