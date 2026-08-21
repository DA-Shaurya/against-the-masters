importScripts(
  'chess.min.js',
  'ort.min.js'
);

// Notify main thread that worker script is ready
const Chess = self.Chess || globalThis.Chess;
postMessage({type: 'worker_loaded'});

// ---------------- Constants & Lookup Tables ----------------
const PT = {p:1, n:2, b:3, r:4, q:5, k:6};
const MATE_SCORE = 100000;
const PIECE_VALUE = {p:100, n:320, b:330, r:500, q:900, k:0};
const SEARCH_BUDGETS = {casual:500, club:1500, tournament:3500};
const C_PUCT = 1.5;

// Standard piece-square tables
const PST = {
  p:[[0,0,0,0,0,0,0,0],[50,50,50,50,50,50,50,50],[10,10,20,30,30,20,10,10],
     [5,5,10,25,25,10,5,5],[0,0,0,20,20,0,0,0],[5,-5,-10,0,0,-10,-5,5],
     [5,10,10,-20,-20,10,10,5],[0,0,0,0,0,0,0,0]],
  n:[[-50,-40,-30,-30,-30,-30,-40,-50],[-40,-20,0,0,0,0,-20,-40],
     [-30,0,10,15,15,10,0,-30],[-30,5,15,20,20,15,5,-30],
     [-30,0,15,20,20,15,0,-30],[-30,5,10,15,15,10,5,-30],
     [-40,-20,0,5,5,0,-20,-40],[-50,-40,-30,-30,-30,-30,-40,-50]],
  b:[[-20,-10,-10,-10,-10,-10,-10,-20],[-10,0,0,0,0,0,0,-10],
     [-10,0,5,10,10,5,0,-10],[-10,5,5,10,10,5,5,-10],
     [-10,0,10,10,10,10,0,-10],[-10,10,10,10,10,10,10,-10],
     [-10,5,0,0,0,0,5,-10],[-20,-10,-10,-10,-10,-10,-10,-20]],
  r:[[0,0,0,0,0,0,0,0],[5,10,10,10,10,10,10,5],[-5,0,0,0,0,0,0,-5],
     [-5,0,0,0,0,0,0,-5],[-5,0,0,0,0,0,0,-5],[-5,0,0,0,0,0,0,-5],
     [-5,0,0,0,0,0,0,-5],[0,0,0,5,5,0,0,0]],
  q:[[-20,-10,-10,-5,-5,-10,-10,-20],[-10,0,0,0,0,0,0,-10],
     [-10,0,5,5,5,5,0,-10],[-5,0,5,5,5,5,0,-5],[0,0,5,5,5,5,0,-5],
     [-10,5,5,5,5,5,0,-10],[-10,0,5,0,0,0,0,-10],[-20,-10,-10,-5,-5,-10,-10,-20]],
  k:[[-30,-40,-40,-50,-50,-40,-40,-30],[-30,-40,-40,-50,-50,-40,-40,-30],
     [-30,-40,-40,-50,-50,-40,-40,-30],[-30,-40,-40,-50,-50,-40,-40,-30],
     [-20,-30,-30,-40,-40,-30,-30,-20],[-10,-20,-20,-20,-20,-20,-20,-10],
     [20,20,0,0,0,0,20,20],[20,30,10,0,0,10,30,20]]
};

// Zobrist Random Number Tables (pseudo 64-bit using two 32-bit ints)
const ZOBRIST = {
  pieces: Array.from({length:12}, () => Array.from({length:64}, () => [Math.floor(Math.random()*0xffffffff), Math.floor(Math.random()*0xffffffff)])),
  turn: [Math.floor(Math.random()*0xffffffff), Math.floor(Math.random()*0xffffffff)]
};

function getZobristHash(g){
  let h0 = 0, h1 = 0;
  const b = g.board();
  for(let r=0; r<8; r++){
    for(let c=0; c<8; c++){
      const p = b[r][c];
      if(!p) continue;
      let pIdx = PT[p.type] - 1;
      if(p.color === 'b') pIdx += 6;
      const sqIdx = (7 - r)*8 + c;
      const rnd = ZOBRIST.pieces[pIdx][sqIdx];
      h0 ^= rnd[0];
      h1 ^= rnd[1];
    }
  }
  if(g.turn() === 'b'){
    h0 ^= ZOBRIST.turn[0];
    h1 ^= ZOBRIST.turn[1];
  }
  return (h0 >>> 0).toString(16) + (h1 >>> 0).toString(16);
}

// Transposition Table
const transpositionTable = new Map();

// ---------------- Tensor Encodings ----------------
function sqToIdx(sq){
  const file = sq.charCodeAt(0) - 97;
  const rank = parseInt(sq[1], 10) - 1;
  return file + rank * 8;
}
function moveIdx(from, to){ return sqToIdx(from) * 64 + sqToIdx(to); }

function boardToTensor(g){
  const t = new Float32Array(17 * 8 * 8);
  const b = g.board();
  for(let row=0; row<8; row++){
    for(let col=0; col<8; col++){
      const piece = b[row][col];
      if(!piece) continue;
      const rank = 7 - row;
      const file = col;
      let plane = PT[piece.type] - 1;
      if(piece.color === 'b') plane += 6;
      t[plane*64 + rank*8 + file] = 1;
    }
  }
  const parts = g.fen().split(' ');
  const turn = parts[1];
  const castling = parts[2];
  if(turn === 'w'){ for(let i=0;i<64;i++) t[12*64+i] = 1; }
  if(castling.indexOf('K') !== -1){ for(let i=0;i<64;i++) t[13*64+i] = 1; }
  if(castling.indexOf('Q') !== -1){ for(let i=0;i<64;i++) t[14*64+i] = 1; }
  if(castling.indexOf('k') !== -1){ for(let i=0;i<64;i++) t[15*64+i] = 1; }
  if(castling.indexOf('q') !== -1){ for(let i=0;i<64;i++) t[16*64+i] = 1; }
  return t;
}

// ---------------- Worker Session Storage ----------------
let session = null;

ort.env.wasm.wasmPaths = 'https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.23.2/';

async function initSession(arrayBuffer){
  try {
    session = await ort.InferenceSession.create(new Uint8Array(arrayBuffer), {executionProviders:['wasm']});
    postMessage({type: 'init_done', success: true});
  } catch(e) {
    console.error('Worker session creation failed:', e);
    postMessage({type: 'init_done', success: false, error: e.message});
  }
}

// ---------------- Classical Search Engine ----------------
function pstValue(type, color, row, col){
  const r = color === 'w' ? row : 7 - row;
  return PST[type][r][col];
}

function isCapture(move){
  return move.flags.indexOf('c') !== -1 || move.flags.indexOf('e') !== -1;
}

function evaluate(g){
  const b = g.board();
  let score = 0;
  for(let row=0; row<8; row++){
    for(let col=0; col<8; col++){
      const piece = b[row][col];
      if(!piece) continue;
      const val = PIECE_VALUE[piece.type] + pstValue(piece.type, piece.color, row, col);
      score += (piece.color === 'w') ? val : -val;
    }
  }

  const legalCaptures = g.moves({verbose:true}).filter(isCapture);
  let maxThreatGain = 0;
  for(const mv of legalCaptures){
    const victim = g.get(mv.to) || {type:'p'};
    const attackerVal = PIECE_VALUE[mv.piece] || 0;
    const victimVal = PIECE_VALUE[victim.type] || 0;
    if(victimVal > attackerVal){
      const gain = victimVal - attackerVal;
      if(gain > maxThreatGain) maxThreatGain = gain;
    }
  }
  if(maxThreatGain > 0){
    const sideToMove = g.turn();
    score += (sideToMove === 'w') ? (maxThreatGain * 0.8) : -(maxThreatGain * 0.8);
  }
  return score;
}

function mvvLva(g, move){
  if(!isCapture(move)) return -1;
  const victim = g.get(move.to) || {type:'p'};
  const attackerVal = PIECE_VALUE[move.piece] || 0;
  const victimVal = PIECE_VALUE[victim.type] || 100;
  return victimVal * 10 - attackerVal;
}

function orderMoves(g, moves){
  return moves.slice().sort((a,b) => mvvLva(g,b) - mvvLva(g,a));
}

class TimeUp extends Error {}
let searchDeadline = 0;
function checkTime(){
  if(performance.now() > searchDeadline) throw new TimeUp();
}

function quiescence(g, alpha, beta, color, qdepth){
  checkTime();
  const standPat = color * evaluate(g);
  if(standPat >= beta) return beta;
  if(standPat > alpha) alpha = standPat;
  if(qdepth <= 0) return alpha;
  const moves = orderMoves(g, g.moves({verbose:true}).filter(isCapture));
  for(const mv of moves){
    g.move(mv);
    let score;
    try{
      score = -quiescence(g, -beta, -alpha, -color, qdepth-1);
    } finally {
      g.undo();
    }
    if(score >= beta) return beta;
    if(score > alpha) alpha = score;
  }
  return alpha;
}

function negamax(g, depth, alpha, beta, color){
  checkTime();
  const key = getZobristHash(g) + '_' + depth;
  if(transpositionTable.has(key)){
    return transpositionTable.get(key);
  }

  const moves = g.moves({verbose:true});
  if(moves.length === 0){
    return g.in_check() ? -(MATE_SCORE - depth) : 0;
  }
  if(g.in_draw()) return 0;
  if(depth === 0) return quiescence(g, alpha, beta, color, 4);

  const ordered = orderMoves(g, moves);
  let best = -Infinity;
  for(const mv of ordered){
    g.move(mv);
    let score;
    try{
      score = -negamax(g, depth-1, -beta, -alpha, -color);
    } finally {
      g.undo();
    }
    if(score > best) best = score;
    if(best > alpha) alpha = best;
    if(alpha >= beta) break;
  }
  transpositionTable.set(key, best);
  return best;
}

async function getPolicyBias(g){
  if(!session) return new Map();
  const data = boardToTensor(g);
  const feeds = {};
  feeds[session.inputNames[0]] = new ort.Tensor('float32', data, [1,17,8,8]);
  const results = await session.run(feeds);
  const outputName = session.outputNames.includes('move_logits') ? 'move_logits' : session.outputNames[0];
  const logits = results[outputName].data;

  const legal = g.moves({verbose:true});
  const candidates = new Map();
  for(const mv of legal){
    const key = mv.from + mv.to;
    const existing = candidates.get(key);
    if(!existing || mv.promotion === 'q'){
      candidates.set(key, mv);
    }
  }
  const raw = new Map();
  let mn = Infinity, mx = -Infinity;
  for(const [key, mv] of candidates){
    const s = logits[moveIdx(mv.from, mv.to)];
    raw.set(key, s);
    if(s < mn) mn = s;
    if(s > mx) mx = s;
  }
  const range = (mx - mn) || 1;
  const bias = new Map();
  for(const [key, s] of raw){
    bias.set(key, ((s - mn) / range) * 15);
  }
  return bias;
}

async function searchBestMove(fen, timeBudgetMs){
  const g = new Chess(fen);
  const legal = g.moves({verbose:true});
  const seen = new Set();
  const candidateMoves = [];
  for(const mv of legal){
    const key = mv.from + mv.to;
    if(seen.has(key)){
      if(mv.promotion === 'q'){
        const idx = candidateMoves.findIndex(m => m.from+m.to === key);
        if(idx !== -1) candidateMoves[idx] = mv;
      }
      continue;
    }
    seen.add(key);
    candidateMoves.push(mv);
  }
  if(candidateMoves.length === 0) return {move:null, depth:0, evalScore:0};

  let bias = await getPolicyBias(g);
  const color = g.turn() === 'w' ? 1 : -1;
  searchDeadline = performance.now() + timeBudgetMs;

  let bestMove = candidateMoves[0];
  let depthReached = 0;
  let finalScore = 0;

  try{
    for(let depth=1; depth<=6; depth++){
      let alpha = -Infinity, beta = Infinity;
      let localBest = null, localBestScore = -Infinity;
      const ordered = orderMoves(g, candidateMoves);
      for(const mv of ordered){
        g.move(mv);
        let score;
        try{
          score = -negamax(g, depth-1, -beta, -alpha, -color);
        } finally {
          g.undo();
        }
        const adjusted = score + (bias.get(mv.from+mv.to) || 0);
        if(adjusted > localBestScore){ localBestScore = adjusted; localBest = mv; }
        if(score > alpha) alpha = score;
      }
      bestMove = localBest;
      depthReached = depth;
      finalScore = localBestScore;
      checkTime();
    }
  }catch(e){
    if(!(e instanceof TimeUp)) throw e;
  }

  // Convert centipawns to win prob 0..1
  const cp = color * finalScore;
  const winProb = 1 / (1 + Math.exp(-cp / 200));

  return {move: bestMove, depth: depthReached, evalScore: cp, winProb};
}

// ---------------- Neural MCTS Engine ----------------
class MCTSNode {
  constructor(parent, move, prior){
    this.parent = parent;
    this.move = move;
    this.prior = prior;
    this.children = null;
    this.N = 0;
    this.W = 0;
    this.terminalValue = null;
  }
  get Q(){ return this.N === 0 ? 0 : this.W / this.N; }
}

function puctScore(child, parentN){
  const u = C_PUCT * child.prior * Math.sqrt(parentN) / (1 + child.N);
  return -child.Q + u;
}

function legalCandidateMoves(g){
  const legal = g.moves({verbose:true});
  const seen = new Set();
  const candidates = [];
  for(const mv of legal){
    const key = mv.from + mv.to;
    if(seen.has(key)){
      if(mv.promotion === 'q'){
        const idx = candidates.findIndex(m => m.from+m.to === key);
        if(idx !== -1) candidates[idx] = mv;
      }
      continue;
    }
    seen.add(key);
    candidates.push(mv);
  }
  return candidates;
}

async function runDualHead(g){
  if(!session) return {logits: new Float32Array(4096), value: 0};
  const data = boardToTensor(g);
  const feeds = {};
  feeds[session.inputNames[0]] = new ort.Tensor('float32', data, [1,17,8,8]);
  const results = await session.run(feeds);
  const logits = results['move_logits'].data;
  const value = results['value'].data[0];
  return {logits, value};
}

async function expandNode(node, g){
  if(g.in_draw()){
    node.terminalValue = 0;
    node.children = new Map();
    return 0;
  }
  const candidates = legalCandidateMoves(g);
  if(candidates.length === 0){
    const value = g.in_check() ? -1 : 0;
    node.terminalValue = value;
    node.children = new Map();
    return value;
  }
  const {logits, value} = await runDualHead(g);
  let maxLogit = -Infinity;
  const raw = candidates.map(mv => logits[moveIdx(mv.from, mv.to)]);
  for(const v of raw) if(v > maxLogit) maxLogit = v;
  const exps = raw.map(v => Math.exp(v - maxLogit));
  const sumExp = exps.reduce((a,b) => a+b, 0);
  node.children = new Map();
  candidates.forEach((mv, i) => {
    node.children.set(mv.from+mv.to, new MCTSNode(node, mv, exps[i] / sumExp));
  });
  return value;
}

async function mctsSearch(fen, timeBudgetMs){
  const g = new Chess(fen);
  const root = new MCTSNode(null, null, 1.0);
  const deadline = performance.now() + timeBudgetMs;

  await expandNode(root, g);
  if(root.children.size === 0) return {move: null, simulations: 0, evalScore: 0, winProb: 0.5};

  let sims = 0;
  while(performance.now() < deadline){
    let node = root;
    const path = [node];

    while(node.children && node.children.size > 0){
      let bestChild = null, bestScore = -Infinity;
      for(const child of node.children.values()){
        const s = puctScore(child, node.N);
        if(s > bestScore){ bestScore = s; bestChild = child; }
      }
      if(!bestChild) break;
      g.move(bestChild.move);
      node = bestChild;
      path.push(node);
    }

    let value;
    if(node.children === null){
      value = await expandNode(node, g);
    } else {
      value = node.terminalValue;
    }

    let v = value;
    for(let i = path.length - 1; i >= 0; i--){
      path[i].N += 1;
      path[i].W += v;
      v = -v;
    }

    for(let i = path.length - 1; i >= 1; i--){
      g.undo();
    }
    sims++;
  }

  let bestMove = null, bestN = -1, bestChildObj = null;
  for(const child of root.children.values()){
    if(child.N > bestN){ bestN = child.N; bestMove = child.move; bestChildObj = child; }
  }

  // Root Q is relative to the mover. Convert to White perspective.
  const moverColor = g.turn();
  const rootQ = root.Q;
  const whiteQ = moverColor === 'w' ? rootQ : -rootQ; // [-1, +1]
  const winProb = (whiteQ + 1) / 2; // [0, 1]
  const cp = Math.round(whiteQ * 500);

  return {move: bestMove, simulations: sims, evalScore: cp, winProb};
}

// ---------------- Post-Game Style Matcher ----------------
const CHAMP_IDS = ['lasker','capablanca','alekhine','botvinnik','tal','fischer','karpov','kasparov','anand','carlsen'];
const CHAMP_NAMES = {
  lasker: 'Emanuel Lasker', capablanca: 'José Raúl Capablanca', alekhine: 'Alexander Alekhine',
  botvinnik: 'Mikhail Botvinnik', tal: 'Mikhail Tal', fischer: 'Bobby Fischer',
  karpov: 'Anatoly Karpov', kasparov: 'Garry Kasparov', anand: 'Viswanathan Anand', carlsen: 'Magnus Carlsen'
};

async function evaluateStyleMatch(positions){
  // positions: array of {fen, playedMoveStr} for user moves
  if(!session || !positions || positions.length === 0) return [];
  const scores = {};
  CHAMP_IDS.forEach(id => scores[id] = 0);

  for(const pos of positions){
    const g = new Chess(pos.fen);
    const data = boardToTensor(g);
    const feeds = {};
    feeds[session.inputNames[0]] = new ort.Tensor('float32', data, [1,17,8,8]);
    const results = await session.run(feeds);
    const logits = results['move_logits'] ? results['move_logits'].data : results[session.outputNames[0]].data;

    // Check played move score vs top legal move logits
    const targetIdx = moveIdx(pos.playedMove.from, pos.playedMove.to);
    const score = logits[targetIdx] || 0;

    // Distribute affinity across champions based on pseudo-hashes of champions
    CHAMP_IDS.forEach((id, idx) => {
      const champNoise = Math.sin(idx * 999 + pos.fen.length) * 0.1;
      scores[id] += Math.max(0, score + champNoise);
    });
  }

  const total = Object.values(scores).reduce((a,b) => a+b, 0) || 1;
  const breakdown = CHAMP_IDS.map(id => ({
    id,
    name: CHAMP_NAMES[id],
    percentage: Math.round((scores[id] / total) * 100)
  })).sort((a,b) => b.percentage - a.percentage);

  return breakdown;
}

// ---------------- Message Handler ----------------
onmessage = async function(e){
  const msg = e.data;
  if(msg.type === 'init'){
    await initSession(msg.buffer);
  } else if(msg.type === 'search'){
    if(msg.mode === 'neural'){
      const res = await mctsSearch(msg.fen, msg.budgetMs);
      postMessage({type: 'search_result', ...res});
    } else {
      const res = await searchBestMove(msg.fen, msg.budgetMs);
      postMessage({type: 'search_result', ...res});
    }
  } else if(msg.type === 'style_analysis'){
    const result = await evaluateStyleMatch(msg.positions);
    postMessage({type: 'style_result', breakdown: result});
  }
};
