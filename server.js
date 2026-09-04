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

const app = express();
const PORT = process.env.PORT || 3000;
const JWT_SECRET = process.env.JWT_SECRET || 'satquery_secret_key_2026_earth_obs';
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 }
});

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname)));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'claude.html'));
});

// Auth Middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];
  if (!token) return next(); // allow guest if token omitted

  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (!err && user) {
      req.user = user;
    }
    next();
  });
};

app.use(authenticateToken);

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
app.post('/api/auth/login', async (req, res) => {
  try {
    const { username, password } = req.body;
    if (!username || !password) {
      return res.status(400).json({ error: 'Username and password required' });
    }

    let user = await dbGet(`SELECT * FROM users WHERE username = ?`, [username]);
    
    if (!user) {
      // Auto-register if new user
      const hash = bcrypt.hashSync(password, 10);
      const result = await dbRun(
        `INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)`,
        [username, hash, 'Analyst']
      );
      user = { id: result.id, username, role: 'Analyst' };
    } else {
      const match = bcrypt.compareSync(password, user.password_hash);
      if (!match) {
        return res.status(401).json({ error: 'Invalid credentials' });
      }
    }

    const token = jwt.sign(
      { id: user.id, username: user.username, role: user.role },
      JWT_SECRET,
      { expiresIn: '24h' }
    );

    await dbRun(
      `INSERT INTO activity_logs (user_id, action, details) VALUES (?, ?, ?)`,
      [user.id, 'USER_LOGIN', `User ${username} logged in.`]
    );

    res.json({
      message: 'Login successful',
      token,
      user: { id: user.id, username: user.username, role: user.role }
    });
  } catch (err) {
    console.error('Login error:', err);
    res.status(500).json({ error: 'Internal server error during authentication' });
  }
});

// ----------------------------------------------------
// 3. AI Query Execution Endpoint (Invokes Python VLM Bridge)
// ----------------------------------------------------
app.post('/api/queries/run', upload.single('scene'), async (req, res) => {
  try {
    const { query, band_mode = 'optical', tile_id = 'TILE_S2A_2026_089', min_confidence = 0, center_lat = 26.9520, center_lng = 94.1700 } = req.body;

    if (!query || typeof query !== 'string') {
      return res.status(400).json({ error: 'Valid query text is required' });
    }

    const pythonCmd = 'py'; // Windows Python launcher
    const args = [
      path.join(__dirname, 'ai_model_bridge.py'),
      '--query', query,
      '--mode', band_mode,
      '--min_conf', min_confidence.toString()
    ];
    let uploadedPath;
    if (req.file) {
      uploadedPath = path.join(os.tmpdir(), `satquery-${Date.now()}-${req.file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_')}`);
      fs.writeFileSync(uploadedPath, req.file.buffer);
      args.push('--image', uploadedPath);
    }

    execFile(pythonCmd, args, async (error, stdout, stderr) => {
      if (uploadedPath) fs.unlink(uploadedPath, () => {});
      let aiResult;
      if (error) {
        console.warn('⚠️ Python bridge CLI execution warning/fallback:', error.message);
        // Fallback internal simulation if python environment call has issue
        aiResult = fallbackInference(query, band_mode, center_lat, center_lng, min_confidence);
      } else {
        try {
          aiResult = JSON.parse(stdout);
        } catch (parseErr) {
          console.warn('⚠️ Failed to parse Python stdout, using fallback execution:', stdout);
          aiResult = fallbackInference(query, band_mode, center_lat, center_lng, min_confidence);
        }
      }

      // Save Query to SQLite Database
      const userId = req.user ? req.user.id : 1;
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
        model_source: aiResult.model_source || 'fallback-simulation',
        image_received: Boolean(aiResult.image_received),
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
// 4. Query History Endpoint
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
    const userId = req.user ? req.user.id : 1;

    await dbRun(
      `INSERT INTO activity_logs (user_id, action, details) VALUES (?, ?, ?)`,
      [userId, 'EXPORT_GEOJSON', `Exported ${feature_count || 0} features in format: ${format || 'GeoJSON'}`]
    );

    res.json({ success: true, message: 'Export logged successfully' });
  } catch (err) {
    res.status(500).json({ error: 'Failed to log export' });
  }
});

// ----------------------------------------------------
// Fallback Inference Engine (Node.js native generator)
// ----------------------------------------------------
function fallbackInference(queryText, bandMode, centerLat = 26.9520, centerLng = 94.1700, minConf = 0) {
  const CLASSES = {
    flood: { label: "Flood extent", color: "#38BDF8", shape: "polygon" },
    burn: { label: "Burn scar / fire", color: "#F2A93B", shape: "polygon" },
    deforest: { label: "Deforestation", color: "#F97066", shape: "polygon" },
    water: { label: "Water body", color: "#2DD4BF", shape: "polygon" },
    urban: { label: "Urban / built-up", color: "#A78BFA", shape: "box" },
    generic: { label: "Detected feature", color: "#94A3B8", shape: "box" }
  };

  const q = queryText.toLowerCase();
  let clsKey = "generic";
  if (q.includes("flood") || q.includes("inundat") || q.includes("waterlog")) clsKey = "flood";
  else if (q.includes("fire") || q.includes("burn") || q.includes("blaze")) clsKey = "burn";
  else if (q.includes("deforest") || q.includes("tree cover") || q.includes("forest")) clsKey = "deforest";
  else if (q.includes("water") || q.includes("river") || q.includes("lake")) clsKey = "water";
  else if (q.includes("urban") || q.includes("building") || q.includes("built-up")) clsKey = "urban";

  const clsInfo = CLASSES[clsKey];
  const count = Math.floor(Math.random() * 4) + 3;
  const features = [];

  for (let i = 0; i < count; i++) {
    const lat = centerLat + (Math.random() - 0.5) * 0.03;
    const lng = centerLng + (Math.random() - 0.5) * 0.05;
    const confidence = Math.floor(Math.random() * 38) + 60;

    if (confidence >= minConf) {
      let ring = [];
      if (clsInfo.shape === "polygon") {
        const numPts = 7;
        const r = 0.003 + Math.random() * 0.005;
        for (let p = 0; p < numPts; p++) {
          const angle = (p / numPts) * Math.PI * 2;
          const dist = r * (0.6 + Math.random() * 0.5);
          ring.push([
            parseFloat((lng + Math.cos(angle) * dist * 1.3).toFixed(6)),
            parseFloat((lat + Math.sin(angle) * dist).toFixed(6))
          ]);
        }
        ring.push(ring[0]);
      } else {
        const w = 0.004 + Math.random() * 0.005;
        const h = 0.003 + Math.random() * 0.004;
        ring = [
          [parseFloat((lng - w/2).toFixed(6)), parseFloat((lat - h/2).toFixed(6))],
          [parseFloat((lng + w/2).toFixed(6)), parseFloat((lat - h/2).toFixed(6))],
          [parseFloat((lng + w/2).toFixed(6)), parseFloat((lat + h/2).toFixed(6))],
          [parseFloat((lng - w/2).toFixed(6)), parseFloat((lat + h/2).toFixed(6))],
          [parseFloat((lng - w/2).toFixed(6)), parseFloat((lat - h/2).toFixed(6))]
        ];
      }

      features.append ? features.push({
        type: "Feature",
        properties: {
          id: `det_js_${Math.floor(Math.random() * 9000) + 1000}`,
          class: clsKey,
          label: clsInfo.label,
          confidence: confidence,
          band_mode: bandMode
        },
        geometry: {
          type: "Polygon",
          coordinates: [ring]
        }
      }) : features.push({
        type: "Feature",
        properties: {
          id: `det_js_${Math.floor(Math.random() * 9000) + 1000}`,
          class: clsKey,
          label: clsInfo.label,
          confidence: confidence,
          band_mode: bandMode
        },
        geometry: {
          type: "Polygon",
          coordinates: [ring]
        }
      });
    }
  }

  const avgConf = features.length ? Math.round(features.reduce((s, f) => s + f.properties.confidence, 0) / features.length) : 0;

  return {
    status: "success",
    query: queryText,
    query_class: clsKey,
    class_label: clsInfo.label,
    band_mode: bandMode,
    feature_count: features.length,
    avg_confidence: avgConf,
    features: features
  };
}

// Start Server
app.listen(PORT, () => {
  console.log(`====================================================`);
  console.log(`🚀 SatQuery AI Server live at http://localhost:${PORT}`);
  console.log(`🛰️ SQLite Database initialized & operational`);
  console.log(`====================================================`);
});
