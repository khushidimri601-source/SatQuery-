const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { execFile } = require('child_process');
const multer = require('multer');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
require('dotenv').config();

const { dbQuery, dbGet, dbRun } = require('./db');
const intelligencePy = path.join(__dirname, 'intelligence.py');

const app = express();
const PORT = process.env.PORT || 3000;
const JWT_SECRET = process.env.JWT_SECRET || '';
if (JWT_SECRET.length < 32) throw new Error('JWT_SECRET must be set in .env and be at least 32 characters long.');
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN || '';
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowedMimes = new Set([
      'image/jpeg', 'image/pjpeg', 'image/jfif', 'image/png', 'image/webp',
      'image/tiff', 'image/geotiff', 'application/octet-stream'
    ]);
    const ext = path.extname(file.originalname || '').toLowerCase();
    const allowedExts = new Set(['.jpg', '.jpeg', '.jfif', '.pjp', '.pjpeg', '.png', '.webp', '.tif', '.tiff']);
    const ok = allowedMimes.has(file.mimetype) || allowedExts.has(ext);
    cb(ok ? null : new Error('Unsupported scene format. Supported formats: JPG, JPEG, JFIF, PNG, WEBP, GeoTIFF.'), ok);
  }
});

// Middleware
app.use(cors({
  origin: ALLOWED_ORIGIN ? ALLOWED_ORIGIN.split(',').map(v => v.trim()) : true,
  credentials: false
}));
app.use(express.json({ limit: '1mb' }));
app.disable('x-powered-by');
app.use((req,res,next)=>{
  res.setHeader('X-Content-Type-Options','nosniff');
  res.setHeader('X-Frame-Options','SAMEORIGIN');
  res.setHeader('Referrer-Policy','strict-origin-when-cross-origin');
  next();
});
app.use(express.static(path.join(__dirname)));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'claude.html'));
});

// ----------------------------------------------------
// Cross-platform Python detection (cached after first successful probe)
// ----------------------------------------------------
let PYTHON_CMD = null;
const PYTHON_CANDIDATES = process.platform === 'win32'
  ? ['py', 'python', 'python3']
  : ['python3', 'python'];

function detectPython(callback) {
  if (PYTHON_CMD) return callback(PYTHON_CMD);
  const tryNext = (i) => {
    if (i >= PYTHON_CANDIDATES.length) {
      console.error('❌ No working Python interpreter found. Tried:', PYTHON_CANDIDATES.join(', '));
      return callback(null);
    }
    const candidate = PYTHON_CANDIDATES[i];
    execFile(candidate, ['--version'], (error) => {
      if (!error) {
        PYTHON_CMD = candidate;
        console.log(`✅ Using Python interpreter: "${candidate}"`);
        callback(candidate);
      } else {
        tryNext(i + 1);
      }
    });
  };
  tryNext(0);
}

function runPython(script, args, callback) {
  detectPython((cmd) => {
    if (!cmd) {
      return callback(new Error('No Python interpreter found on this system. Install Python 3 and make sure "python3" or "python" is on your PATH, then restart the server.'));
    }
    execFile(cmd, [script, ...args], { maxBuffer: 30 * 1024 * 1024, timeout: 60000 }, callback);
  });
}

function tempFile(prefix, file) {
  const tmpDir = path.join(__dirname, '.tmp');
  if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir, { recursive: true });
  const ext = path.extname(file.originalname || '').toLowerCase() || '.png';
  const safeBase = path.basename(file.originalname || 'scene', ext).replace(/[^a-zA-Z0-9._-]/g, '_');
  const p = path.join(tmpDir, `${prefix}-${Date.now()}-${safeBase}${ext}`);
  fs.writeFileSync(p, file.buffer);
  return p;
}

// Auth + lightweight abuse protection. Analysis APIs require a real JWT; no guest/mocked sessions.
const rateBuckets = new Map();
function rateLimit(req, res, next) {
  const key = `${req.ip}:${req.path}`;
  const now = Date.now(); const windowMs = 60_000; const max = req.path.includes('/auth/') ? 12 : 90;
  const bucket = rateBuckets.get(key) || { start: now, count: 0 };
  if (now - bucket.start > windowMs) { bucket.start = now; bucket.count = 0; }
  bucket.count++; rateBuckets.set(key, bucket);
  if (bucket.count > max) return res.status(429).json({ error: 'Too many requests. Please wait a minute and try again.' });
  next();
}

const PUBLIC_API = new Set(['/api/health','/api/auth/login','/api/auth/register']);
const authenticateToken = (req, res, next) => {
  const route = req.originalUrl.split('?')[0];
  if (PUBLIC_API.has(route)) return next();
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.startsWith('Bearer ') ? authHeader.slice(7) : null;
  if (!token) return res.status(401).json({ error: 'Authentication required.' });
  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err || !user) return res.status(401).json({ error: 'Session expired or invalid. Please sign in again.' });
    req.user = user; next();
  });
};
app.use('/api', rateLimit, authenticateToken);

// ----------------------------------------------------
// 1. Healthcheck Endpoint
// ----------------------------------------------------
app.get('/api/health', (req, res) => {
  res.json({
    status: 'online',
    service: 'SatQuery AI Backend Engine',
    timestamp: new Date().toISOString()
  });
});

// ----------------------------------------------------
// 2. Authentication Endpoints
// ----------------------------------------------------
app.post('auth/register', async (req, res) => {
  try {
    const username = String(req.body?.username || '').trim();
    const password = String(req.body?.password || '');
    if (!/^[A-Za-z0-9_.-]{3,32}$/.test(username)) return res.status(400).json({ error: 'Username must be 3–32 characters: letters, numbers, _, ., or -.' });
    if (password.length < 8) return res.status(400).json({ error: 'Password must be at least 8 characters.' });
    const existing = await dbGet(`SELECT id FROM users WHERE username = ?`, [username]);
    if (existing) return res.status(409).json({ error: 'Username already exists. Sign in instead.' });
    const hash = await bcrypt.hash(password, 12);
    const result = await dbRun(`INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)`, [username, hash, 'Analyst']);
    res.status(201).json({ message: 'Account created. You can now sign in.', user: { id: result.id, username, role: 'Analyst' } });
  } catch (err) { console.error('Registration error:', err); res.status(500).json({ error: 'Unable to create account.' }); }
});

app.post('auth/login', async (req, res) => {
  try {
    const username = String(req.body?.username || '').trim();
    const password = String(req.body?.password || '');
    if (!username || !password) return res.status(400).json({ error: 'Username and password required.' });
    const user = await dbGet(`SELECT * FROM users WHERE username = ?`, [username]);
    if (!user || !(await bcrypt.compare(password, user.password_hash))) return res.status(401).json({ error: 'Invalid username or password.' });
    const token = jwt.sign({ id: user.id, username: user.username, role: user.role }, JWT_SECRET, { expiresIn: '8h', issuer: 'satquery' });
    await dbRun(`INSERT INTO activity_logs (user_id, action, details) VALUES (?, ?, ?)`, [user.id, 'USER_LOGIN', 'Successful login']);
    res.json({ message: 'Login successful', token, user: { id: user.id, username: user.username, role: user.role } });
  } catch (err) { console.error('Login error:', err); res.status(500).json({ error: 'Authentication service error.' }); }
});

// ----------------------------------------------------
// 3. AI Query Execution Endpoint (Invokes Python VLM Bridge)
// ----------------------------------------------------
app.post('/api/compare', upload.fields([
  { name: 'before', maxCount: 1 },
  { name: 'after', maxCount: 1 }
]), async (req, res) => {
  let beforePath, afterPath;
  try {
    const before = req.files?.before?.[0];
    const after = req.files?.after?.[0];
    const sensitivity = Math.max(1, Math.min(100, Number(req.body.sensitivity || 35)));

    if (!before || !after) {
      return res.status(400).json({ error: 'Both Before and After images are required.' });
    }

    beforePath = tempFile('satquery-before', before);
    afterPath = tempFile('satquery-after', after);

    const args = [
      '--before', beforePath,
      '--after', afterPath,
      '--sensitivity', String(sensitivity)
    ];

    runPython(path.join(__dirname, 'compare_images.py'), args, (error, stdout, stderr) => {
      if (beforePath) fs.unlink(beforePath, () => {});
      if (afterPath) fs.unlink(afterPath, () => {});

      if (error) {
        console.error('Comparison error:', (stderr || error.message || '').toString());
        return res.status(500).json({ error: error.message && error.message.includes('No Python interpreter') ? error.message : 'Image comparison failed. Install Python dependencies (pip install -r requirements.txt) and try again.', details: (stderr || error.message || '').toString().slice(0, 1000) });
      }

      try {
        const result = JSON.parse(stdout);
        res.json(result);
      } catch (e) {
        console.error('Invalid comparison output:', stdout);
        res.status(500).json({ error: 'Comparison engine returned invalid output.' });
      }
    });
  } catch (err) {
    if (beforePath) fs.unlink(beforePath, () => {});
    if (afterPath) fs.unlink(afterPath, () => {});
    console.error('Comparison endpoint error:', err);
    res.status(500).json({ error: 'Unable to compare images.' });
  }
});

app.post('/api/queries/run', upload.single('scene'), async (req, res) => {
  try {
    const { query, band_mode = 'optical', tile_id = 'TILE_S2A_2026_089', min_confidence = 0, center_lat = 26.9520, center_lng = 94.1700 } = req.body;

    if (!query || typeof query !== 'string' || query.length > 500) {
      return res.status(400).json({ error: 'Query must be a non-empty string of at most 500 characters.' });
    }

    const args = [
      '--query', query,
      '--mode', band_mode,
      '--min_conf', min_confidence.toString(),
      '--center_lat', String(center_lat),
      '--center_lng', String(center_lng)
    ];
    let uploadedPath;
    if (req.file) {
      uploadedPath = path.join(os.tmpdir(), `satquery-${Date.now()}-${req.file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_')}`);
      fs.writeFileSync(uploadedPath, req.file.buffer);
      args.push('--image', uploadedPath);
    }

    runPython(path.join(__dirname, 'ai_model_bridge.py'), args, async (error, stdout, stderr) => {
      if (uploadedPath) fs.unlink(uploadedPath, () => {});
      let aiResult;
      if (error) {
        console.error('❌ Python analysis failed:', (stderr || error.message || '').toString());
        return res.status(500).json({
          error: error.message && error.message.includes('No Python interpreter') ? error.message : 'Image analysis engine failed. Make sure Python and the required packages are installed (pip install -r requirements.txt).',
          details: (stderr || error.message || '').toString().slice(0, 1000)
        });
      }

      try {
        aiResult = JSON.parse(stdout);
      } catch (parseErr) {
        console.error('❌ Invalid Python analysis output:', stdout);
        return res.status(500).json({
          error: 'The analysis engine returned an invalid result.',
          details: String(parseErr.message || parseErr)
        });
      }

      // Save Query to SQLite Database
      const userId = req.user.id;
      const queryRun = await dbRun(
        `INSERT INTO queries (user_id, query_text, query_class, tile_id, band_mode, avg_confidence) VALUES (?, ?, ?, ?, ?, ?)`,
        [userId, query, aiResult.query_class, tile_id, band_mode, aiResult.avg_confidence]
      );
      const queryId = queryRun.id;

      // Save Detections to SQLite Database
      if (aiResult.features && aiResult.features.length > 0) {
        for (const feat of aiResult.features) {
          await dbRun(
            `INSERT INTO detections (query_id, feature_id, label, class_name, confidence, geometry_geojson) VALUES (?, ?, ?, ?, ?, ?)`,
            [
              queryId,
              feat.properties.id,
              feat.properties.label,
              feat.properties.class,
              feat.properties.confidence,
              JSON.stringify(feat.geometry)
            ]
          );
        }
      }

      // Log Activity
      await dbRun(
        `INSERT INTO activity_logs (user_id, action, details) VALUES (?, ?, ?)`,
        [userId, 'QUERY_RUN', `Executed query: "${query}" | Detections: ${aiResult.feature_count}`]
      );

      res.json({
        success: true,
        query_id: queryId,
        tile_id,
        band_mode,
        query: query,
        query_class: aiResult.query_class,
        class_label: aiResult.class_label,
        avg_confidence: aiResult.avg_confidence,
        feature_count: aiResult.feature_count,
        model_source: aiResult.model_source || 'cv-screening',
        image_received: Boolean(aiResult.image_received),
        image_metrics: aiResult.image_metrics || null,
        answer: aiResult.answer || '',
        disclaimer: aiResult.disclaimer || '',
        model: aiResult.model || null,
        uploaded_file: req.file ? {
          original_name: req.file.originalname,
          mime_type: req.file.mimetype,
          size: req.file.size
        } : null,
        features: aiResult.features,
        geojson: {
          type: "FeatureCollection",
          features: aiResult.features
        }
      });
    });

  } catch (err) {
    console.error('Query execution error:', err);
    res.status(500).json({ error: 'Server error processing satellite query' });
  }
});

// ----------------------------------------------------
// 4. Higher-level intelligence endpoints
// ----------------------------------------------------
// Real Prithvi capability/status endpoint. Heavy model loading only occurs during analysis when enabled.
app.get('/api/model/status', (req, res) => {
  res.json({
    prithvi_enabled: String(process.env.SATQUERY_PRITHVI_ENABLED || 'false').toLowerCase() === 'true',
    model: process.env.SATQUERY_PRITHVI_MODEL || 'prithvi_eo_v2_300',
    input: 'six-band HLS-style GeoTIFF: BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2',
    note: 'Prithvi is used as a pretrained EO backbone; task-specific flood/land-cover claims require a compatible downstream checkpoint.'
  });
});


app.post('/api/intelligence/quality', upload.single('scene'), (req,res)=>{
  if(!req.file) return res.status(400).json({error:'Scene image is required.'});
  const p=tempFile('satquery-quality',req.file);
  runPython(intelligencePy,['--quality',p],(error,stdout,stderr)=>{ fs.unlink(p,()=>{}); if(error) return res.status(500).json({error:stderr||error.message}); try{res.json(JSON.parse(stdout));}catch(e){res.status(500).json({error:'Invalid quality result.'});} });
});

app.post('/api/intelligence/report', express.json(), async (req,res)=>{
  try {
    const {query='',analysis=null,comparison=null,timeline=null}=req.body||{};
    const reportPath=path.join(os.tmpdir(),`satquery-report-${Date.now()}.json`);
    fs.writeFileSync(reportPath,JSON.stringify({query,analysis,comparison,timeline}));
    runPython(intelligencePy,['--report',reportPath],(error,stdout,stderr)=>{fs.unlink(reportPath,()=>{});if(error)return res.status(500).json({error:stderr||error.message});try{res.json(JSON.parse(stdout));}catch(e){res.status(500).json({error:'Invalid report result.'});}});
  } catch(e){res.status(500).json({error:'Report generation failed.'});}
});

app.post('/api/intelligence/timeline', upload.array('scenes',8), (req,res)=>{
  if(!req.files || req.files.length < 1) return res.status(400).json({error:'Upload at least one scene for timeline analysis.'});
  const paths=req.files.map(f=>tempFile('satquery-time',f)); const args=['--timeline',...paths,'--sensitivity',String(req.body.sensitivity||35)];
  runPython(intelligencePy,args,(error,stdout,stderr)=>{paths.forEach(p=>fs.unlink(p,()=>{}));if(error)return res.status(500).json({error:stderr||error.message});try{res.json(JSON.parse(stdout));}catch(e){res.status(500).json({error:'Invalid timeline result.'});}});
});

// ----------------------------------------------------
// 5. Query History Endpoint
// ----------------------------------------------------
app.get('/api/queries/history', async (req, res) => {
  try {
    const queries = await dbQuery(
      `SELECT q.*, u.username FROM queries q LEFT JOIN users u ON q.user_id = u.id ORDER BY q.created_at DESC LIMIT 50`
    );

    const fullHistory = await Promise.all(
      queries.map(async (q) => {
        const detections = await dbQuery(`SELECT * FROM detections WHERE query_id = ?`, [q.id]);
        return {
          id: q.id,
          username: q.username || 'Analyst',
          query_text: q.query_text,
          query_class: q.query_class,
          tile_id: q.tile_id,
          band_mode: q.band_mode,
          avg_confidence: q.avg_confidence,
          created_at: q.created_at,
          detections_count: detections.length,
          features: detections.map(d => ({
            type: "Feature",
            properties: {
              id: d.feature_id,
              label: d.label,
              class: d.class_name,
              confidence: d.confidence
            },
            geometry: JSON.parse(d.geometry_geojson)
          }))
        };
      })
    );

    res.json({ count: fullHistory.length, history: fullHistory });
  } catch (err) {
    console.error('Error fetching history:', err);
    res.status(500).json({ error: 'Failed to retrieve query history' });
  }
});

// ----------------------------------------------------
// 5. System Statistics Endpoint
// ----------------------------------------------------
app.get('/api/stats', async (req, res) => {
  try {
    const qCount = await dbGet(`SELECT COUNT(*) as total FROM queries`);
    const dCount = await dbGet(`SELECT COUNT(*) as total FROM detections`);
    const uCount = await dbGet(`SELECT COUNT(*) as total FROM users`);
    const avgConf = await dbGet(`SELECT AVG(avg_confidence) as avg FROM queries`);
    const recentQueries = await dbQuery(`SELECT query_text, avg_confidence, created_at FROM queries ORDER BY created_at DESC LIMIT 5`);

    res.json({
      total_queries: qCount ? qCount.total : 0,
      total_detections: dCount ? dCount.total : 0,
      active_users: uCount ? uCount.total : 0,
      system_avg_confidence: avgConf && avgConf.avg ? Math.round(avgConf.avg) : 0,
      recent_queries: recentQueries
    });
  } catch (err) {
    console.error('Error fetching stats:', err);
    res.status(500).json({ error: 'Failed to retrieve statistics' });
  }
});

// ----------------------------------------------------
// 6. Export Activity Logging Endpoint
// ----------------------------------------------------
app.post('/api/exports', async (req, res) => {
  try {
    const { format, feature_count } = req.body;
    const userId = req.user.id;

    await dbRun(
      `INSERT INTO activity_logs (user_id, action, details) VALUES (?, ?, ?)`,
      [userId, 'EXPORT_GEOJSON', `Exported ${feature_count || 0} features in format: ${format || 'GeoJSON'}`]
    );

    res.json({ success: true, message: 'Export logged successfully' });
  } catch (err) {
    res.status(500).json({ error: 'Failed to log export' });
  }
});

app.use((err, req, res, next) => {
  if (err && err.name === 'MulterError') return res.status(400).json({ error: `Upload error: ${err.message}` });
  if (err && /Unsupported scene format/.test(err.message || '')) return res.status(415).json({ error: err.message });
  console.error('Unhandled server error:', err);
  res.status(500).json({ error: 'Unexpected server error.' });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 SatQuery server running at http://localhost:${PORT}`);
});
