// content.js - Detects Vimeo iframes and Mux players on the page

function findVimeoVideos() {
  const videos = [];
  const iframes = document.querySelectorAll('iframe[src*="player.vimeo.com/video/"]');

  iframes.forEach((iframe) => {
    const src = iframe.getAttribute('src');
    const match = src.match(/player\.vimeo\.com\/video\/(\d+)/);
    if (!match) return;

    const videoId = match[1];
    const url = new URL(src, window.location.href);
    const hash = url.searchParams.get('h') || '';

    if (videos.some((v) => v.videoId === videoId)) return;

    videos.push({
      videoId,
      hash,
      iframeSrc: src,
      pageUrl: window.location.href,
    });
  });

  return videos;
}

function findMuxVideos() {
  const videos = [];

  // Detect <mux-player playback-id="...">
  document.querySelectorAll('mux-player[playback-id]').forEach((player) => {
    const playbackId = player.getAttribute('playback-id');
    if (!playbackId) return;
    if (videos.some((v) => v.playbackId === playbackId)) return;

    const title = player.getAttribute('metadata-video-title') || '';
    videos.push({ playbackId, title });
  });

  // Fallback: detect <mux-video src="...stream.mux.com/...">
  document.querySelectorAll('mux-video[src*="stream.mux.com"]').forEach((el) => {
    const src = el.getAttribute('src') || '';
    const match = src.match(/stream\.mux\.com\/([^/.]+)/);
    if (!match) return;

    const playbackId = match[1];
    if (videos.some((v) => v.playbackId === playbackId)) return;

    const player = el.closest('mux-player');
    const title = player?.getAttribute('metadata-video-title') || '';
    videos.push({ playbackId, title });
  });

  return videos;
}

function getTotalCount() {
  return findVimeoVideos().length + findMuxVideos().length;
}

function updateBadge(count) {
  chrome.runtime.sendMessage({ action: 'updateBadge', count });
}

// Initial scan
const initialCount = getTotalCount();
if (initialCount > 0) {
  updateBadge(initialCount);
}

// Watch for dynamically added elements (SPA)
const observer = new MutationObserver(() => {
  updateBadge(getTotalCount());
});

observer.observe(document.body, {
  childList: true,
  subtree: true,
});

// Respond to messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'detectVideos') {
    sendResponse({
      vimeoVideos: findVimeoVideos(),
      muxVideos: findMuxVideos(),
    });
  }
  return true;
});
