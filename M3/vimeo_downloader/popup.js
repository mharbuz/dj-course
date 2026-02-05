// popup.js - UI logic for Video Lesson Downloader

const loadingEl = document.getElementById('loading');
const noVideosEl = document.getElementById('no-videos');
const errorEl = document.getElementById('error');
const videoListEl = document.getElementById('video-list');
const settingsEl = document.getElementById('settings');
const moduleInput = document.getElementById('module');
const lessonInput = document.getElementById('lesson');
const subfolderInput = document.getElementById('subfolder');
const previewEl = document.getElementById('filename-preview');
const downloadBtn = document.getElementById('download-btn');
const statusEl = document.getElementById('status');

// Each entry: { type, id, config: { title, duration?, progressive: [{url,quality,height}] }, error? }
let detectedVideos = [];
let currentVideoIndex = 0;

init();

async function init() {
  const saved = await chrome.storage.local.get(['subfolder', 'lastModule', 'lastLesson']);
  if (saved.subfolder) subfolderInput.value = saved.subfolder;
  if (saved.lastModule) moduleInput.value = saved.lastModule;
  if (saved.lastLesson) lessonInput.value = saved.lastLesson;
  updatePreview();

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id) {
      showError('No active tab found.');
      return;
    }

    // Run executeScript in all frames to detect both Vimeo and Mux videos
    const extractResult = await chrome.runtime.sendMessage({
      action: 'extractVideosFromTab',
      tabId: tab.id,
    });

    if (!extractResult.success || extractResult.videos.length === 0) {
      // Fallback: try content script detection (for badge-detected videos)
      try {
        const cs = await chrome.tabs.sendMessage(tab.id, { action: 'detectVideos' });
        const total = (cs.vimeoVideos?.length || 0) + (cs.muxVideos?.length || 0);
        if (total === 0) {
          loadingEl.classList.add('hidden');
          noVideosEl.classList.remove('hidden');
          return;
        }
        // Videos were detected but config extraction failed
        for (const v of (cs.vimeoVideos || [])) {
          detectedVideos.push({
            type: 'vimeo', id: v.videoId,
            error: 'Could not extract config. Try reloading the page.',
          });
        }
        for (const v of (cs.muxVideos || [])) {
          detectedVideos.push({
            type: 'mux', id: v.playbackId,
            config: { title: v.title || `Mux ${v.playbackId}`, progressive: [] },
          });
          // Still try probing Mux renditions below
        }
      } catch (e) {
        loadingEl.classList.add('hidden');
        noVideosEl.classList.remove('hidden');
        return;
      }
    }

    // Process results from executeScript
    const rawVideos = extractResult.success ? extractResult.videos : [];
    for (const v of rawVideos) {
      if (v.type === 'vimeo') {
        detectedVideos.push({
          type: 'vimeo',
          id: v.videoId,
          config: v.config, // already has { title, duration, progressive }
        });
      } else if (v.type === 'mux') {
        detectedVideos.push({
          type: 'mux',
          id: v.playbackId,
          config: { title: v.title || `Mux ${v.playbackId}`, progressive: [] },
        });
      }
    }

    // For each Mux video, probe MP4 renditions
    for (const video of detectedVideos) {
      if (video.type !== 'mux' || video.error) continue;
      try {
        const result = await chrome.runtime.sendMessage({
          action: 'probeMuxRenditions',
          playbackId: video.id,
        });
        if (result.success && result.renditions.length > 0) {
          video.config.progressive = result.renditions;
        } else {
          video.config.hlsUrl = `https://stream.mux.com/${video.id}.m3u8`;
          video.error = 'No MP4 renditions. Use yt-dlp with the HLS URL.';
        }
      } catch (e) {
        video.error = 'Failed to probe Mux renditions.';
      }
    }

    if (detectedVideos.length === 0) {
      loadingEl.classList.add('hidden');
      noVideosEl.classList.remove('hidden');
      return;
    }

    renderVideos();
  } catch (err) {
    showError('Please reload the page and try again.');
  }
}

function renderVideos() {
  loadingEl.classList.add('hidden');

  if (detectedVideos.length === 0) {
    noVideosEl.classList.remove('hidden');
    return;
  }

  videoListEl.innerHTML = '';
  videoListEl.classList.remove('hidden');
  settingsEl.classList.remove('hidden');

  detectedVideos.forEach((video, index) => {
    const item = document.createElement('div');
    item.className = 'video-item';
    const badge = video.type === 'mux' ? '<span class="badge mux">MUX</span>' : '<span class="badge vimeo">VIMEO</span>';

    if (video.error && (!video.config || video.config.progressive.length === 0)) {
      const hlsInfo = video.config?.hlsUrl
        ? `<div class="hls-url">${escapeHtml(video.config.hlsUrl)}</div>`
        : '';
      item.innerHTML = `
        <div class="video-title">${badge} ${escapeHtml(video.config?.title || video.id)}</div>
        <div class="video-meta" style="color: #e06060;">${video.error}</div>
        ${hlsInfo}
      `;
    } else {
      const config = video.config;
      const duration = formatDuration(config.duration);
      const qualityOptions = config.progressive
        .map((p, i) => `<option value="${i}">${p.quality}</option>`)
        .join('');

      item.innerHTML = `
        <div class="video-title">${badge} ${escapeHtml(config.title)}</div>
        <div class="video-meta">
          <span>${duration}</span>
          <label>Quality:</label>
          <select data-video-index="${index}">${qualityOptions}</select>
        </div>
      `;

      if (currentVideoIndex === 0 || !detectedVideos[currentVideoIndex]?.config?.progressive?.length) {
        currentVideoIndex = index;
      }
    }

    item.addEventListener('click', () => {
      if (video.config?.progressive?.length) {
        currentVideoIndex = index;
        highlightSelected();
      }
    });

    videoListEl.appendChild(item);
  });

  highlightSelected();
}

function highlightSelected() {
  const items = videoListEl.querySelectorAll('.video-item');
  items.forEach((item, i) => {
    item.style.background = i === currentVideoIndex ? '#2a2a3e' : 'transparent';
  });
}

function formatDuration(seconds) {
  if (!seconds) return '';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function getFilename() {
  const mod = String(moduleInput.value).padStart(2, '0');
  const les = String(lessonInput.value).padStart(2, '0');
  const sub = subfolderInput.value.trim();
  const name = `M${mod}L${les}.mp4`;
  return sub ? `${sub}/${name}` : name;
}

function updatePreview() {
  previewEl.textContent = getFilename();
}

moduleInput.addEventListener('input', updatePreview);
lessonInput.addEventListener('input', updatePreview);
subfolderInput.addEventListener('input', updatePreview);

// Download
downloadBtn.addEventListener('click', async () => {
  const video = detectedVideos[currentVideoIndex];
  if (!video?.config?.progressive?.length) {
    setStatus('No downloadable video selected.', true);
    return;
  }

  const qualitySelect = videoListEl.querySelector(
    `select[data-video-index="${currentVideoIndex}"]`
  );
  const qualityIndex = qualitySelect ? parseInt(qualitySelect.value) : 0;
  const progressive = video.config.progressive[qualityIndex];

  if (!progressive?.url) {
    setStatus('No download URL available.', true);
    return;
  }

  const filename = getFilename();

  await chrome.storage.local.set({
    subfolder: subfolderInput.value.trim(),
    lastModule: parseInt(moduleInput.value) || 1,
    lastLesson: parseInt(lessonInput.value) || 1,
  });

  downloadBtn.disabled = true;
  setStatus('Downloading...');

  try {
    const result = await chrome.runtime.sendMessage({
      action: 'downloadVideo',
      url: progressive.url,
      filename,
    });

    if (result.success) {
      setStatus(`Download started: ${filename}`);
      lessonInput.value = parseInt(lessonInput.value) + 1;
      updatePreview();
      await chrome.storage.local.set({
        lastLesson: parseInt(lessonInput.value),
      });
    } else {
      setStatus(`Error: ${result.error}`, true);
    }
  } catch (err) {
    setStatus(`Error: ${err.message}`, true);
  }

  downloadBtn.disabled = false;
});

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.style.color = isError ? '#e06060' : '#888';
}

function showError(message) {
  loadingEl.classList.add('hidden');
  errorEl.textContent = message;
  errorEl.classList.remove('hidden');
}
