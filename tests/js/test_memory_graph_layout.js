'use strict';

const assert = require('node:assert/strict');
const {
  computeSemanticLayout,
  sanitizeSnapshot,
  textHalfWidth,
} = require('../../static/memory_graph_layout.js');

const nodes = [
  { id: 'user:me', type: 'user', label: '我' },
  { id: 'person:lisa', type: 'person', label: 'Lisa' },
  { id: 'place:huanglong', type: 'place', label: '黄龙国际中心' },
  { id: 'place:guomao', type: 'place', label: '国贸地铁站' },
  { id: 'place:xhs', type: 'place', label: '小红书(杭州)公司' },
  { id: 'place:nanjing', type: 'place', label: '南京' },
  { id: 'person:ajie', type: 'person', label: '阿杰' },
  { id: 'place:henan', type: 'place', label: '河南' },
  { id: 'fact:quiet', type: 'fact', label: '安静' },
  { id: 'candidate:cafe', type: 'candidate', label: '安静的咖啡馆', status: 'candidate' },
];
const edges = [
  { source: 'user:me', target: 'person:lisa' },
  { source: 'person:lisa', target: 'place:huanglong' },
  { source: 'person:lisa', target: 'place:guomao' },
  { source: 'user:me', target: 'place:xhs' },
  { source: 'place:xhs', target: 'place:nanjing' },
  { source: 'user:me', target: 'person:ajie' },
  { source: 'person:ajie', target: 'place:henan' },
  { source: 'user:me', target: 'fact:quiet' },
  { source: 'fact:quiet', target: 'candidate:cafe' },
];

const first = computeSemanticLayout(nodes, edges);
const second = computeSemanticLayout(nodes, edges);
assert.deepEqual(first, second, 'layout must be deterministic');
assert.deepEqual(first.positions['user:me'], { x: 0, y: 0 });

assert.equal(first.meta.branchByNode['place:huanglong'], 'person:lisa');
assert.equal(first.meta.branchByNode['place:guomao'], 'person:lisa');
assert.equal(first.meta.branchByNode['place:nanjing'], 'place:xhs');
assert.equal(first.meta.branchByNode['place:henan'], 'person:ajie');

const nestedEdgeLengths = edges
  .filter(edge => edge.source !== 'user:me')
  .map(edge => Math.hypot(
    first.positions[edge.source].x - first.positions[edge.target].x,
    first.positions[edge.source].y - first.positions[edge.target].y,
  ));
assert.ok(Math.max(...nestedEdgeLengths) < 250, 'related nodes should remain visually compact');

for (let i = 0; i < nodes.length; i += 1) {
  for (let j = i + 1; j < nodes.length; j += 1) {
    const a = nodes[i];
    const b = nodes[j];
    const pa = first.positions[a.id];
    const pb = first.positions[b.id];
    const overlapX = textHalfWidth(a) + textHalfWidth(b) + 8 - Math.abs(pa.x - pb.x);
    const overlapY = 64 - Math.abs(pa.y - pb.y);
    assert.ok(overlapX <= 0 || overlapY <= 0, `${a.label} overlaps ${b.label}`);
  }
}

function orientation(a, b, c) {
  return Math.sign((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
}

function edgesCross(a, b, c, d) {
  return orientation(a, b, c) !== orientation(a, b, d)
    && orientation(c, d, a) !== orientation(c, d, b);
}

for (let i = 0; i < edges.length; i += 1) {
  for (let j = i + 1; j < edges.length; j += 1) {
    const a = edges[i];
    const b = edges[j];
    if ([a.source, a.target].some(id => id === b.source || id === b.target)) continue;
    assert.equal(
      edgesCross(
        first.positions[a.source],
        first.positions[a.target],
        first.positions[b.source],
        first.positions[b.target],
      ),
      false,
      `${a.source}-${a.target} crosses ${b.source}-${b.target}`,
    );
  }
}

assert.deepEqual(
  sanitizeSnapshot(
    {
      positions: {
        'person:lisa': { x: 12, y: -30 },
        forged: { x: 4, y: 5 },
        broken: { x: Infinity, y: 1 },
      },
      x: 20,
      y: -10,
      k: 99,
    },
    new Set(['person:lisa', 'broken']),
  ),
  { positions: { 'person:lisa': { x: 12, y: -30 } }, x: 20, y: -10, k: 2.8 },
);

console.log('memory graph layout tests passed');
