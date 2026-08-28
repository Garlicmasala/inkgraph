/* Inkgraph keeps state small and explicit so each behavior can be tested in isolation. */
const examples = {
  routine: 'A quiet morning routine → coffee → deep work → a walk outside',
  network: 'Research → sketch → prototype → share',
  journey: 'Idea begins → takes shape → finds its people'
};

const state = { source: 'graph', style: 'sumi', density: 64, wobble: 18, zoom: 100 };
const translations = {
  en: { pageTitle: 'Inkgraph — Turn anything into ink', studio: 'Studio', library: 'Library', about: 'About', signIn: 'Sign in with Google', signedIn: 'Signed in', newCanvas: 'Studio / New canvas', headline: 'Make it feel<br><em>handmade.</em>', introCopy: 'Transform graphs, diagrams, and ideas into expressive ink. Tune the character, keep the meaning.', source: 'Source', graph: 'Graph', diagram: 'Diagram', text: 'Text', pasteSource: 'Paste a prompt or source', character: 'Character', export: 'Export' },
  zh: { pageTitle: 'Inkgraph — 将一切变成水墨', studio: '工作台', library: '素材库', about: '关于', signIn: '使用 Google 登录', signedIn: '已登录', newCanvas: '工作台 / 新画布', headline: '让它拥有<br><em>手作感。</em>', introCopy: '将图表、示意图和想法转化为富有表现力的水墨作品。保留结构，也保留灵魂。', source: '来源', graph: '图表', diagram: '示意图', text: '文本', pasteSource: '粘贴提示词或来源', character: '风格', export: '导出' },
  ja: { pageTitle: 'Inkgraph — すべてをインクに', studio: 'スタジオ', library: 'ライブラリ', about: '概要', signIn: 'Google でログイン', signedIn: 'ログイン済み', newCanvas: 'スタジオ / 新しいキャンバス', headline: '手作りの<br><em>温度を。</em>', introCopy: 'グラフや図、アイデアを表情豊かなインクに。意味を保ったまま、個性を調整できます。', source: 'ソース', graph: 'グラフ', diagram: '図', text: 'テキスト', pasteSource: 'プロンプトまたはソースを貼り付け', character: '表現', export: '書き出す' }
};
function updateCharCount() { document.querySelector('#char-count').textContent = `${document.querySelector('#source-input').value.length} / 500`; }
function setRangeOutput(id, value) { document.querySelector(`#${id}-output`).textContent = `${value}%`; }
function setZoom(nextZoom) { state.zoom = Math.max(75, Math.min(125, nextZoom)); document.querySelector('#paper').style.transform = `scale(${state.zoom / 100})`; document.querySelector('#zoom-value').textContent = `${state.zoom}%`; }
function exportSvg() {
  const svg = document.querySelector('#ink-svg').cloneNode(true);
  svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  const blob = new Blob([new XMLSerializer().serializeToString(svg)], { type: 'image/svg+xml' });
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = 'inkgraph-export.svg'; link.click(); URL.revokeObjectURL(link.href);
}
function applyLanguage(language) { const dictionary = translations[language] || translations.en; document.documentElement.lang = language; document.title = dictionary.pageTitle; document.querySelectorAll('[data-i18n]').forEach((element) => { element.innerHTML = dictionary[element.dataset.i18n] || element.innerHTML; }); localStorage.setItem('inkgraph-language', language); }
function signInWithGoogle() {
  const clientId = document.querySelector('meta[name="google-client-id"]').content.trim();
  const button = document.querySelector('#google-login');
  if (!clientId || !window.google?.accounts?.id) { button.title = 'Add a Google OAuth client ID to enable sign-in'; button.textContent = 'Google login needs client ID'; return; }
  window.google.accounts.id.initialize({ client_id: clientId, callback: (response) => { if (response.credential) { button.textContent = translations[document.documentElement.lang].signedIn; button.classList.add('signed-in'); document.querySelector('.avatar').hidden = false; } } });
  window.google.accounts.id.prompt();
}
async function localAuth(path) {
  const email = document.querySelector('#auth-email').value.trim();
  const password = document.querySelector('#auth-password').value;
  const status = document.querySelector('#auth-status');
  status.textContent = '';
  try {
    const response = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'same-origin', body: JSON.stringify({ email, password }) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Authentication failed');
    status.textContent = result.message;
    document.querySelector('#auth-dialog').close();
    document.querySelector('#local-login').textContent = email;
    document.querySelector('.avatar').hidden = false;
  } catch (error) { status.textContent = error.message; }
}

if (typeof document !== 'undefined') {
  const input = document.querySelector('#source-input');
  input.addEventListener('input', updateCharCount);
  document.querySelectorAll('.source-tab').forEach((button) => button.addEventListener('click', () => { state.source = button.dataset.source; document.querySelectorAll('.source-tab').forEach((tab) => tab.classList.toggle('active', tab === button)); }));
  document.querySelectorAll('.style-option').forEach((button) => button.addEventListener('click', () => { state.style = button.dataset.style; document.querySelectorAll('.style-option').forEach((option) => { const selected = option === button; option.classList.toggle('selected', selected); option.setAttribute('aria-checked', selected); }); }));
  document.querySelectorAll('.tradition').forEach((button) => button.addEventListener('click', () => { document.querySelectorAll('.tradition').forEach((option) => { const selected = option === button; option.classList.toggle('selected', selected); option.setAttribute('aria-checked', selected); }); }));
  document.querySelectorAll('.recent-thumb').forEach((button) => button.addEventListener('click', () => { input.value = examples[button.dataset.example]; updateCharCount(); document.querySelectorAll('.recent-thumb').forEach((thumb) => thumb.classList.toggle('active', thumb === button)); }));
  document.querySelector('#density').addEventListener('input', (event) => { state.density = event.target.value; setRangeOutput('density', event.target.value); });
  document.querySelector('#wobble').addEventListener('input', (event) => { state.wobble = event.target.value; setRangeOutput('wobble', event.target.value); });
  document.querySelector('#zoom-in').addEventListener('click', () => setZoom(state.zoom + 10));
  document.querySelector('#zoom-out').addEventListener('click', () => setZoom(state.zoom - 10));
  document.querySelector('#export-btn').addEventListener('click', exportSvg);
  document.querySelector('#theme-toggle').addEventListener('click', () => document.body.classList.toggle('warm-mode'));
  document.querySelector('#language-select').addEventListener('change', (event) => applyLanguage(event.target.value));
  document.querySelector('#google-login').addEventListener('click', signInWithGoogle);
  document.querySelector('#local-login').addEventListener('click', () => document.querySelector('#auth-dialog').showModal());
  document.querySelector('#auth-form').addEventListener('submit', (event) => { event.preventDefault(); localAuth('/login'); });
  document.querySelector('#register-button').addEventListener('click', () => localAuth('/register'));
  const savedLanguage = localStorage.getItem('inkgraph-language') || 'en'; document.querySelector('#language-select').value = savedLanguage; applyLanguage(savedLanguage);
  updateCharCount();
}

if (typeof module !== 'undefined') module.exports = { examples, translations, setZoom, updateCharCount };