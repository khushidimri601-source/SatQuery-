const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const bcrypt = require('bcryptjs');

// Persistent SQLite database for history across restarts
const DB_PATH = process.env.DB_PATH || path.join(__dirname, 'satquery.db');

const db = new sqlite3.Database(DB_PATH, (err) => {
  if (err) {
    console.error('❌ Failed to connect to SQLite database:', err.message);
  } else {
    console.log('✅ Connected to SQLite database mode:', DB_PATH);
  }
});


// Initialize database schema
db.serialize(() => {
  // 1. Users table
  db.run(`
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL,
      role TEXT DEFAULT 'analyst',
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  // 2. Queries table
  db.run(`
    CREATE TABLE IF NOT EXISTS queries (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER,
      query_text TEXT NOT NULL,
      query_class TEXT NOT NULL,
      tile_id TEXT NOT NULL,
      band_mode TEXT DEFAULT 'optical',
      avg_confidence INTEGER DEFAULT 0,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (user_id) REFERENCES users(id)
    )
  `);

  // 3. Detections table
  db.run(`
    CREATE TABLE IF NOT EXISTS detections (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      query_id INTEGER NOT NULL,
      feature_id TEXT NOT NULL,
      label TEXT NOT NULL,
      class_name TEXT NOT NULL,
      confidence INTEGER NOT NULL,
      geometry_geojson TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE CASCADE
    )
  `);

  // 4. Activity Logs & Exports table
  db.run(`
    CREATE TABLE IF NOT EXISTS activity_logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER,
      action TEXT NOT NULL,
      details TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (user_id) REFERENCES users(id)
    )
  `);

  // No hardcoded demo/guest accounts. Create an account through /api/auth/register.



});

// Database helper functions (Promise-based)
const dbQuery = (sql, params = []) => {
  return new Promise((resolve, reject) => {
    db.all(sql, params, (err, rows) => {
      if (err) reject(err);
      else resolve(rows);
    });
  });
};

const dbGet = (sql, params = []) => {
  return new Promise((resolve, reject) => {
    db.get(sql, params, (err, row) => {
      if (err) reject(err);
      else resolve(row);
    });
  });
};

const dbRun = (sql, params = []) => {
  return new Promise((resolve, reject) => {
    db.run(sql, params, function (err) {
      if (err) reject(err);
      else resolve({ id: this.lastID, changes: this.changes });
    });
  });
};

module.exports = { db, dbQuery, dbGet, dbRun };
