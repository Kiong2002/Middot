(function attachMemoryGraphLayout(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.MiddotMemoryGraphLayout = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function memoryGraphLayoutFactory() {
  'use strict';

  const TAU = Math.PI * 2;
  const ROOT_ID = 'user:me';

  function compareNodes(a, b) {
    const priority = { user: 0, person: 1, organization: 2, school: 2, place: 3, poi: 3, fact: 4, candidate: 5 };
    const pa = priority[a?.type] ?? 4;
    const pb = priority[b?.type] ?? 4;
    if (pa !== pb) return pa - pb;
    return String(a?.label || a?.id || '').localeCompare(String(b?.label || b?.id || ''), 'zh-CN');
  }

  function textHalfWidth(node) {
    const length = Array.from(String(node?.label || node?.id || '')).length;
    const radius = node?.id === ROOT_ID || node?.type === 'user' ? 34 : node?.status === 'candidate' ? 26 : 29;
    return Math.max(radius, Math.min(104, 10 + length * 5.4));
  }

  function buildForest(nodes, edges, rootId) {
    const byId = new Map(nodes.map(node => [node.id, node]));
    const adjacency = new Map(nodes.map(node => [node.id, []]));
    edges.forEach(edge => {
      if (!byId.has(edge.source) || !byId.has(edge.target)) return;
      adjacency.get(edge.source).push(edge.target);
      adjacency.get(edge.target).push(edge.source);
    });
    adjacency.forEach(list => list.sort((a, b) => compareNodes(byId.get(a), byId.get(b))));

    const parent = new Map([[rootId, null]]);
    const depth = new Map([[rootId, 0]]);
    const branch = new Map([[rootId, rootId]]);
    const children = new Map(nodes.map(node => [node.id, []]));
    const visited = new Set([rootId]);

    function visitComponent(seed, seedBranch, seedDepth, seedParent) {
      visited.add(seed);
      parent.set(seed, seedParent);
      depth.set(seed, seedDepth);
      branch.set(seed, seedBranch);
      if (seedParent) children.get(seedParent).push(seed);
      const queue = [seed];
      while (queue.length) {
        const current = queue.shift();
        for (const neighbor of adjacency.get(current) || []) {
          if (visited.has(neighbor)) continue;
          visited.add(neighbor);
          parent.set(neighbor, current);
          depth.set(neighbor, (depth.get(current) || 0) + 1);
          branch.set(neighbor, seedBranch);
          children.get(current).push(neighbor);
          queue.push(neighbor);
        }
      }
    }

    for (const neighbor of adjacency.get(rootId) || []) {
      if (!visited.has(neighbor)) visitComponent(neighbor, neighbor, 1, rootId);
    }
    const disconnected = nodes.filter(node => !visited.has(node.id)).sort(compareNodes);
    for (const node of disconnected) {
      if (!visited.has(node.id)) visitComponent(node.id, node.id, 1, rootId);
    }
    children.forEach(list => list.sort((a, b) => compareNodes(byId.get(a), byId.get(b))));
    return { byId, parent, depth, branch, children };
  }

  function subtreeSizes(rootId, children) {
    const sizes = new Map();
    function count(id) {
      if (sizes.has(id)) return sizes.get(id);
      const size = 1 + (children.get(id) || []).reduce((sum, child) => sum + count(child), 0);
      sizes.set(id, size);
      return size;
    }
    count(rootId);
    return sizes;
  }

  function resolveCollisions(nodes, positions, anchors, rootId) {
    const movable = nodes.filter(node => node.id !== rootId && node.type !== 'user');
    for (let iteration = 0; iteration < 72; iteration += 1) {
      const cooling = 1 - iteration / 90;
      for (let i = 0; i < nodes.length; i += 1) {
        for (let j = i + 1; j < nodes.length; j += 1) {
          const aNode = nodes[i];
          const bNode = nodes[j];
          const a = positions[aNode.id];
          const b = positions[bNode.id];
          if (!a || !b) continue;
          const overlapX = textHalfWidth(aNode) + textHalfWidth(bNode) + 20 - Math.abs(a.x - b.x);
          const overlapY = 76 - Math.abs(a.y - b.y);
          if (overlapX <= 0 || overlapY <= 0) continue;
          const aFixed = aNode.id === rootId || aNode.type === 'user';
          const bFixed = bNode.id === rootId || bNode.type === 'user';
          const ratioA = aFixed ? 0 : bFixed ? 1 : 0.5;
          const ratioB = bFixed ? 0 : aFixed ? 1 : 0.5;
          if (overlapX < overlapY) {
            const direction = a.x === b.x ? (String(aNode.id) < String(bNode.id) ? -1 : 1) : Math.sign(a.x - b.x);
            const push = overlapX * 0.52 * cooling;
            a.x += direction * push * ratioA;
            b.x -= direction * push * ratioB;
          } else {
            const direction = a.y === b.y ? (String(aNode.id) < String(bNode.id) ? -1 : 1) : Math.sign(a.y - b.y);
            const push = overlapY * 0.52 * cooling;
            a.y += direction * push * ratioA;
            b.y -= direction * push * ratioB;
          }
        }
      }
      movable.forEach(node => {
        const point = positions[node.id];
        const anchor = anchors[node.id];
        point.x += (anchor.x - point.x) * 0.035;
        point.y += (anchor.y - point.y) * 0.035;
      });
    }
    positions[rootId] = { x: 0, y: 0 };
  }

  function computeSemanticLayout(rawNodes, rawEdges) {
    const nodes = (rawNodes || []).filter(node => node && node.id);
    if (!nodes.length) return { positions: {}, meta: { rootId: '', branchByNode: {}, sectors: [] } };
    const rootNode = nodes.find(node => node.id === ROOT_ID) || nodes.find(node => node.type === 'user') || nodes[0];
    const rootId = rootNode.id;
    const edges = (rawEdges || []).filter(edge => edge && edge.source && edge.target);
    const forest = buildForest(nodes, edges, rootId);
    const sizes = subtreeSizes(rootId, forest.children);
    const rootChildren = [...(forest.children.get(rootId) || [])].sort((a, b) => compareNodes(forest.byId.get(a), forest.byId.get(b)));
    const weights = rootChildren.map(id => Math.max(1, Math.sqrt(sizes.get(id) || 1)));
    const totalWeight = weights.reduce((sum, value) => sum + value, 0) || 1;
    const positions = { [rootId]: { x: 0, y: 0 } };
    const anchors = { [rootId]: { x: 0, y: 0 } };
    const sectors = [];
    let cursor = -Math.PI / 2 - Math.PI / 10;

    function placeSubtree(id, start, end, level) {
      const children = forest.children.get(id) || [];
      const angle = (start + end) / 2;
      const ownSize = sizes.get(id) || 1;
      const radius = level === 1
        ? 178 + Math.min(34, Math.sqrt(ownSize) * 9)
        : 178 + level * 142 + Math.min(24, children.length * 5);
      positions[id] = { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius };
      anchors[id] = { ...positions[id] };
      if (!children.length) return;
      const usableStart = start + (end - start) * 0.08;
      const usableEnd = end - (end - start) * 0.08;
      const childWeights = children.map(child => Math.max(1, Math.sqrt(sizes.get(child) || 1)));
      const childTotal = childWeights.reduce((sum, value) => sum + value, 0) || 1;
      let childCursor = usableStart;
      children.forEach((child, index) => {
        const span = (usableEnd - usableStart) * (childWeights[index] / childTotal);
        placeSubtree(child, childCursor, childCursor + span, level + 1);
        childCursor += span;
      });
    }

    rootChildren.forEach((id, index) => {
      const span = TAU * (weights[index] / totalWeight);
      const start = cursor;
      const end = cursor + span;
      sectors.push({ id, start, end, weight: weights[index], size: sizes.get(id) || 1 });
      placeSubtree(id, start, end, 1);
      cursor = end;
    });
    resolveCollisions(nodes, positions, anchors, rootId);
    const branchByNode = {};
    forest.branch.forEach((value, key) => { branchByNode[key] = value; });
    return { positions, meta: { rootId, branchByNode, sectors } };
  }

  function sanitizeSnapshot(raw, validNodeIds) {
    const ids = validNodeIds instanceof Set ? validNodeIds : new Set(validNodeIds || []);
    const positions = {};
    Object.entries(raw?.positions || {}).slice(0, 100).forEach(([id, point]) => {
      const x = Number(point?.x);
      const y = Number(point?.y);
      if ((!ids.size || ids.has(id)) && Number.isFinite(x) && Number.isFinite(y) && Math.abs(x) < 10000 && Math.abs(y) < 10000) {
        positions[id] = { x, y };
      }
    });
    const x = Number(raw?.x);
    const y = Number(raw?.y);
    const k = Number(raw?.k);
    return {
      positions,
      x: Number.isFinite(x) && Math.abs(x) < 10000 ? x : 0,
      y: Number.isFinite(y) && Math.abs(y) < 10000 ? y : 0,
      k: Number.isFinite(k) ? Math.max(0.3, Math.min(2.8, k)) : 1,
    };
  }

  return { computeSemanticLayout, sanitizeSnapshot, textHalfWidth };
});
