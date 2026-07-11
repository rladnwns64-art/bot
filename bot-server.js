// ============================================================
// bot-server.js — 유저 행동 학습형 봇 서버
//   Supabase에 로그 저장 → TensorFlow.js로 학습 → 봇에게 액션 예측
// ============================================================
// 환경변수:
//   SUPABASE_URL         = https://xxx.supabase.co
//   SUPABASE_KEY         = sb_publishable_... (또는 service_role)
//   BOT_SERVER_TOKEN     = 관리자 인증용 랜덤 문자열 (학습/봇 제어)
//   PORT                 = 3001 (기본)
// ============================================================
const express = require('express');
const cors = require('cors');
const { createClient } = require('@supabase/supabase-js');
const tf = require('@tensorflow/tfjs');   // 순수 JS 버전 (nodejs 무거우면 이걸로)
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 3001;
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_KEY;
const ADMIN_TOKEN = process.env.BOT_SERVER_TOKEN || 'change_me_pls';

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('❌ SUPABASE_URL / SUPABASE_KEY 환경변수 필요');
  process.exit(1);
}
const sb = createClient(SUPABASE_URL, SUPABASE_KEY);

// ============================================================
// 액션 인덱스 (딥러닝 출력 레이블)
//   추가 시 여기 배열에만 append (기존 인덱스 유지 필수)
// ============================================================
const ACTIONS = [
  'idle',           // 0 정지
  'move_left',      // 1
  'move_right',     // 2
  'jump',           // 3
  'attack',         // 4
  'mine',           // 5
  'use_item',       // 6
  'pickup',         // 7
  'flee_left',      // 8 도망(왼쪽)
  'flee_right',     // 9 도망(오른쪽)
];
const ACTION_COUNT = ACTIONS.length;

// 상태 벡터 크기 (아래 state → tensor 변환과 맞춰야 함)
const STATE_DIM = 14;

// ============================================================
// 상태 → 텐서 변환
//   클라에서 state 오는 형식:
//   {hp,max_hp,mp,gold,level,x,y,dir, nearby_mobs, nearby_players, time_of_day, in_ocean, in_space, has_weapon}
// ============================================================
function stateToVector(s) {
  return [
    (s.hp || 0) / (s.max_hp || 100),        // 0. hp 비율
    (s.mp || 0) / 100,                       // 1. mp 정규화
    Math.min(1, (s.gold || 0) / 10000),      // 2. gold 로그 스케일 대신 clip
    Math.min(1, (s.level || 1) / 50),        // 3. level
    (s.x || 0) / 5000,                       // 4. x 정규화 (WORLD_W 대략)
    (s.y || 0) / 2000,                       // 5. y
    (s.dir === -1 ? 0 : 1),                  // 6. 방향
    Math.min(1, (s.nearby_mobs || 0) / 10),  // 7. 몹 수
    Math.min(1, (s.nearby_players || 0) / 5),// 8. 플레이어 수
    ((s.time_of_day || 0) % 24) / 24,        // 9. 시간
    s.in_ocean ? 1 : 0,                      // 10. 심해?
    s.in_space ? 1 : 0,                      // 11. 우주?
    s.has_weapon ? 1 : 0,                    // 12. 무기 소지?
    Math.min(1, (s.hunger || 100) / 100),    // 13. 허기
  ];
}

// ============================================================
// 신경망 (간단한 MLP)
// ============================================================
let currentModel = null;
const MODEL_DIR = path.join(__dirname, 'bot-model');

function buildModel() {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [STATE_DIM] }));
  m.add(tf.layers.dropout({ rate: 0.2 }));
  m.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  m.add(tf.layers.dense({ units: ACTION_COUNT, activation: 'softmax' }));
  m.compile({ optimizer: tf.train.adam(0.001), loss: 'sparseCategoricalCrossentropy', metrics: ['accuracy'] });
  return m;
}

// 모델 저장/로드 (파일 시스템)
async function saveModel(model) {
  if (!fs.existsSync(MODEL_DIR)) fs.mkdirSync(MODEL_DIR, { recursive: true });
  await model.save('file://' + MODEL_DIR);
  console.log('💾 모델 저장:', MODEL_DIR);
}
async function loadModel() {
  try {
    if (!fs.existsSync(path.join(MODEL_DIR, 'model.json'))) return null;
    const m = await tf.loadLayersModel('file://' + path.join(MODEL_DIR, 'model.json'));
    m.compile({ optimizer: tf.train.adam(0.001), loss: 'sparseCategoricalCrossentropy', metrics: ['accuracy'] });
    console.log('📂 모델 로드 완료');
    return m;
  } catch (e) { console.warn('모델 로드 실패:', e.message); return null; }
}

// 시작 시 모델 로드 시도
(async () => { currentModel = await loadModel(); })();

// ============================================================
// Express 서버
// ============================================================
const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));
app.disable('x-powered-by');

// 헬스체크
app.get('/', (req, res) => res.json({ ok: true, service: 'bot-server', modelLoaded: !!currentModel }));
app.get('/health', (req, res) => res.send('ok'));

// ============================================================
// 📝 로깅: 클라가 state+action을 보내면 Supabase에 저장
// ============================================================
app.post('/log', async (req, res) => {
  try {
    const b = req.body || {};
    // 검증
    if (!b.user_id || !b.state || !b.action) return res.status(400).json({ error: 'bad_input' });
    if (!ACTIONS.includes(b.action)) return res.status(400).json({ error: 'unknown_action' });
    const row = {
      user_id: String(b.user_id).slice(0, 40),
      ts: Date.now(),
      // state 필드들
      hp: b.state.hp | 0, max_hp: b.state.max_hp | 0,
      mp: b.state.mp | 0, gold: b.state.gold | 0, level: b.state.level | 0,
      x: +b.state.x || 0, y: +b.state.y || 0,
      dir: b.state.dir | 0,
      nearby_mobs: b.state.nearby_mobs | 0,
      nearby_players: b.state.nearby_players | 0,
      time_of_day: b.state.time_of_day | 0,
      in_ocean: !!b.state.in_ocean,
      in_space: !!b.state.in_space,
      has_weapon: !!b.state.has_weapon,
      hunger: b.state.hunger | 0,
      // action
      action: b.action,
      action_target: b.action_target ? String(b.action_target).slice(0, 40) : null,
      // reward (선택)
      hp_delta: b.hp_delta | 0,
      gold_delta: b.gold_delta | 0,
    };
    const { error } = await sb.from('player_actions').insert([row]);
    if (error) { console.warn('sb insert:', error.message); return res.status(500).json({ error: 'db_fail' }); }
    return res.json({ ok: true });
  } catch (e) { console.error('log', e); return res.status(500).json({ error: 'server_error' }); }
});

// ============================================================
// 🧠 예측: 상태 주면 봇이 할 액션 반환
// ============================================================
app.post('/predict', async (req, res) => {
  try {
    if (!currentModel) return res.json({ ok: true, action: 'idle', reason: 'no_model' });
    const s = req.body?.state; if (!s) return res.status(400).json({ error: 'no_state' });
    const vec = stateToVector(s);
    const t = tf.tensor2d([vec]);
    const pred = currentModel.predict(t);
    const probs = await pred.data();
    t.dispose(); pred.dispose();
    // 상위 3개 액션
    const ranked = Array.from(probs)
      .map((p, i) => ({ action: ACTIONS[i], p }))
      .sort((a, b) => b.p - a.p);
    // 온도 샘플링 (다양성)
    const temp = 0.7;
    const scaled = Array.from(probs).map(p => Math.pow(p, 1/temp));
    const sum = scaled.reduce((a,b)=>a+b, 0);
    const norm = scaled.map(p => p / sum);
    let r = Math.random(), acc = 0, chosenIdx = 0;
    for (let i = 0; i < norm.length; i++) { acc += norm[i]; if (r < acc) { chosenIdx = i; break; } }
    return res.json({
      ok: true,
      action: ACTIONS[chosenIdx],
      probability: probs[chosenIdx],
      top3: ranked.slice(0, 3),
    });
  } catch (e) { console.error('predict', e); return res.status(500).json({ error: 'server_error' }); }
});

// ============================================================
// 🎓 학습: 관리자가 호출 → Supabase에서 데이터 뽑아 학습
// ============================================================
let _isTraining = false;
let _trainProgress = null;

app.post('/train', async (req, res) => {
  if (req.body?.token !== ADMIN_TOKEN) return res.status(401).json({ error: 'unauthorized' });
  if (_isTraining) return res.json({ error: 'busy', progress: _trainProgress });
  const epochs = Math.min(50, Math.max(1, req.body?.epochs || 10));
  const limit = Math.min(50000, Math.max(100, req.body?.limit || 10000));

  _isTraining = true; _trainProgress = { epoch: 0, loss: null, acc: null, samples: 0 };
  res.json({ ok: true, started: true, epochs, limit });

  // 백그라운드 학습
  (async () => {
    try {
      console.log(`🎓 학습 시작: ${limit} 샘플, ${epochs} epoch`);
      // 데이터 가져오기
      const { data, error } = await sb
        .from('player_actions')
        .select('*')
        .order('ts', { ascending: false })
        .limit(limit);
      if (error) { console.error('데이터 로드 실패:', error.message); _isTraining = false; return; }
      if (!data || data.length < 20) { console.warn('데이터 부족:', data?.length); _isTraining = false; return; }

      _trainProgress.samples = data.length;
      // 텐서 변환
      const xs = tf.tensor2d(data.map(d => stateToVector(d)));
      const ys = tf.tensor1d(data.map(d => Math.max(0, ACTIONS.indexOf(d.action))), 'int32');

      // 모델 생성/재사용
      if (!currentModel) currentModel = buildModel();

      await currentModel.fit(xs, ys, {
        epochs,
        batchSize: 32,
        validationSplit: 0.1,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            _trainProgress.epoch = epoch + 1;
            _trainProgress.loss = logs.loss;
            _trainProgress.acc = logs.acc;
            console.log(`  epoch ${epoch+1}/${epochs} loss=${logs.loss.toFixed(4)} acc=${(logs.acc*100).toFixed(1)}%`);
          }
        }
      });

      xs.dispose(); ys.dispose();
      await saveModel(currentModel);
      console.log('✅ 학습 완료');
    } catch (e) {
      console.error('학습 에러:', e.message);
    } finally { _isTraining = false; }
  })();
});

// 학습 진행 상태
app.get('/train/status', (req, res) => {
  res.json({ isTraining: _isTraining, progress: _trainProgress, hasModel: !!currentModel });
});

// ============================================================
// 📊 통계
// ============================================================
app.get('/stats', async (req, res) => {
  try {
    const { count } = await sb.from('player_actions').select('*', { count: 'exact', head: true });
    const { data: recent } = await sb.from('player_actions')
      .select('user_id, action, ts')
      .order('ts', { ascending: false })
      .limit(10);
    // 액션별 분포 (샘플 1000)
    const { data: sample } = await sb.from('player_actions')
      .select('action')
      .order('ts', { ascending: false })
      .limit(1000);
    const dist = {};
    for (const r of (sample || [])) dist[r.action] = (dist[r.action] || 0) + 1;
    res.json({ total: count, recent: recent || [], actionDist: dist, actions: ACTIONS });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// 🗑️ 모델 리셋 (관리자)
app.post('/model/reset', (req, res) => {
  if (req.body?.token !== ADMIN_TOKEN) return res.status(401).json({ error: 'unauthorized' });
  currentModel = null;
  try { fs.rmSync(MODEL_DIR, { recursive: true, force: true }); } catch(e) {}
  res.json({ ok: true });
});

app.listen(PORT, () => console.log(`🤖 봇 서버 실행 :${PORT}`));
