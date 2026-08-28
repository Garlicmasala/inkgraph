/* Lightweight TDD/BDD checks for the pure parts of the studio model. Run: node app.test.js */
const assert = require('node:assert/strict');
const { examples, translations } = require('./app.js');

function scenario(description, test) { try { test(); console.log(`✓ ${description}`); } catch (error) { console.error(`✗ ${description}\n  ${error.message}`); process.exitCode = 1; } }

scenario('Given a library example, when it is selected, then it supplies source text', () => {
  assert.equal(typeof examples.routine, 'string');
  assert.match(examples.routine, /coffee/);
});

scenario('Given a source model, when the user chooses a different example, then examples remain distinct', () => {
  assert.notEqual(examples.routine, examples.network);
  assert.ok(Object.keys(examples).length >= 3);
});

scenario('Given an export action, then the preview contains an SVG document', () => {
  assert.match(require('node:fs').readFileSync('./index.html', 'utf8'), /id="ink-svg"/);
});

scenario('Given a supported language, then the studio has translated core labels', () => {
  assert.match(translations.zh.signIn, /Google/);
  assert.ok(translations.ja.export);
  assert.ok(translations.en.headline.includes('handmade'));
});

scenario('Given Google login configuration, then the browser exposes a credential slot without a secret', () => {
  assert.match(require('node:fs').readFileSync('./index.html', 'utf8'), /name="google-client-id" content=""/);
  assert.match(require('node:fs').readFileSync('./index.html', 'utf8'), /accounts\.google\.com\/gsi\/client/);
});