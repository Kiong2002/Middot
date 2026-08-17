'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const page = fs.readFileSync(
  path.join(__dirname, '..', '..', 'static', 'index.html'),
  'utf8',
);

function functionSource(name) {
  const marker = `function ${name}(`;
  const start = page.indexOf(marker);
  assert.notEqual(start, -1, `${name} must exist`);
  const open = page.indexOf('{', start);
  let depth = 0;
  for (let index = open; index < page.length; index += 1) {
    if (page[index] === '{') depth += 1;
    if (page[index] === '}') depth -= 1;
    if (depth === 0) return page.slice(start, index + 1);
  }
  throw new Error(`${name} is not closed`);
}

const hasActiveSource = functionSource('assistHasActiveChoiceCard');
const deferSource = functionSource('assistDeferDraftCardUntilChoicesFinish');
const renderDraftSource = functionSource('assistRenderDraftCard');
const renderLocationSource = functionSource('assistRenderLocationChoices');
const applyDraftsSource = functionSource('applyDrafts');
const searchProgressStartSource = functionSource('draftApplySearchProgressStart');

const state = {
  drafts: [{ kind: 'set_participant_location' }],
  draftCardEl: null,
  msgsEl: { querySelector: () => ({ className: 'assist-choice-card active' }) },
};
const helpers = new Function(
  'ASSIST_STATE',
  `${hasActiveSource}\n${deferSource}\nreturn { assistHasActiveChoiceCard, assistDeferDraftCardUntilChoicesFinish };`,
)(state);

assert.equal(helpers.assistHasActiveChoiceCard(), true);

let removed = false;
state.draftCardEl = { remove: () => { removed = true; } };
helpers.assistDeferDraftCardUntilChoicesFinish();
assert.equal(removed, true, 'visible draft card should be removed while a choice is active');
assert.equal(state.draftCardEl, null);
assert.equal(state.drafts.length, 1, 'deferred drafts must stay queued');

assert.match(
  renderDraftSource,
  /if \(assistHasActiveChoiceCard\(\)\) return;/,
  'draft rendering must be blocked by an active choice card',
);
assert.match(
  renderLocationSource,
  /assistDeferDraftCardUntilChoicesFinish\(\);/,
  'a new location choice must hide an already visible draft card',
);
assert.match(
  applyDraftsSource,
  /draftApplySearchProgressStart\(chosen\)/,
  'applying participant drafts must start visible recommendation progress',
);
assert.match(
  applyDraftsSource,
  /draftApplySearchProgressComplete\(\)/,
  'successful draft application must complete visible recommendation progress',
);
assert.match(
  applyDraftsSource,
  /draftApplySearchProgressFail\(e\.message \|\| '应用失败'\)/,
  'failed draft application must expose a visible progress error',
);
assert.match(
  searchProgressStartSource,
  /showResultsPanel\(\);[\s\S]*progressReset\(\);/,
  'draft-triggered search must reuse the manual search result panel and progress card',
);

const optimisticProjectionAt = applyDraftsSource.indexOf('chosen.forEach(applyDraftLocally)');
const searchStartAt = applyDraftsSource.indexOf('draftApplySearchProgressStart(chosen)');
const networkRequestAt = applyDraftsSource.indexOf("fetch('/api/v2/session/apply-drafts'");
assert.ok(
  optimisticProjectionAt >= 0
    && optimisticProjectionAt < searchStartAt
    && searchStartAt < networkRequestAt,
  'confirmed cards must update locally before search progress and the network request start',
);
assert.match(
  applyDraftsSource,
  /catch \(e\) \{[\s\S]*restoreDraftSnapshotLocally\(snapshot\);[\s\S]*draftApplySearchProgressFail/,
  'failed atomic application must roll the optimistic card update back before showing an error',
);
