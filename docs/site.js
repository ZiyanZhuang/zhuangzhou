(() => {
  const root = document.documentElement;
  let language = new URLSearchParams(location.search).get('lang') === 'en' ? 'en' : 'zh';
  let metadata;
  let metadataUnavailable = false;
  const revision = document.getElementById('build-revision');
  const updated = document.getElementById('build-updated');

  function showMetadata() {
    if (!metadata) {
      if (metadataUnavailable) updated.textContent = language === 'zh' ? '构建信息暂不可用' : 'Build metadata unavailable';
      return;
    }
    revision.textContent = `v${metadata.version} · main@${metadata.revision.slice(0, 7)} ↗`;
    revision.href = `https://github.com/ZiyanZhuang/zhuangzhou/commit/${metadata.revision}`;
    const date = new Date(metadata.updated);
    updated.textContent = (language === 'zh' ? '代码更新：' : 'Code updated: ') +
      date.toLocaleString(language === 'zh' ? 'zh-CN' : 'en-GB', {timeZone: 'UTC', hour12: false}) + ' UTC';
  }

  function setLanguage(value) {
    language = value;
    root.dataset.language = value;
    root.lang = value === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('.language-switch button').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.language === value));
    });
    document.getElementById('copy-status').textContent = '';
    showMetadata();
  }

  document.querySelector('.language-switch').hidden = false;
  document.querySelectorAll('.language-switch button').forEach(button => {
    button.addEventListener('click', () => {
      setLanguage(button.dataset.language);
      const url = new URL(location.href);
      url.searchParams.set('lang', language);
      history.replaceState(null, '', url);
    });
  });
  setLanguage(language);

  if (navigator.clipboard && window.isSecureContext) {
    document.querySelectorAll('.copy-button').forEach(button => {
      button.hidden = false;
      button.addEventListener('click', async () => {
        const status = document.getElementById('copy-status');
        try {
          await navigator.clipboard.writeText(button.parentElement.querySelector('code').textContent);
          status.textContent = language === 'zh' ? '安装命令已复制。' : 'Installation commands copied.';
        } catch {
          status.textContent = language === 'zh' ? '无法访问剪贴板，请手动选取命令。' : 'Clipboard unavailable. Select the commands manually.';
        }
      });
    });
  }

  if (location.protocol === 'http:' || location.protocol === 'https:') {
    fetch('build.json').then(response => {
      if (!response.ok) throw new Error('Build metadata unavailable');
      return response.json();
    }).then(data => {
      if (typeof data.version !== 'string' || !/^[a-f0-9]{40}$/.test(data.revision) ||
          !Number.isFinite(Date.parse(data.updated))) throw new Error('Invalid build metadata');
      metadata = data;
      showMetadata();
    }).catch(() => {
      // The commit-history link remains usable when build metadata is unavailable.
      metadataUnavailable = true;
      showMetadata();
    });
  }
})();
