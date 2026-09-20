const sqlite3 = require('sqlite3').verbose();
const path = require('path');

// ----------------------------------------------------
// DATABASE PATH
// ----------------------------------------------------

const DB_PATH =
  process.env.DB_PATH ||
  path.join(__dirname, 'satquery.db');

console.log('📦 SQLite database path:', DB_PATH);

// ----------------------------------------------------
// DATABASE CONNECTION
// ----------------------------------------------------

const db = new sqlite3.Database(DB_PATH, (err) => {
  if (err) {
    console.error('❌ SQLite connection failed:', err.message);
  } else {
    console.log('✅ SQLite connected');
  }
});

// Enable foreign keys
db.run('PRAGMA foreign_keys = ON');

// ----------------------------------------------------
// DATABASE INITIALIZATION
// ----------------------------------------------------

function initializeDatabase() {
  return new Promise((resolve, reject) => {
    db.serialize(() => {

      db.run(`
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT UNIQUE NOT NULL,
          password_hash TEXT NOT NULL,
          role TEXT DEFAULT 'analyst',
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
      `, (err) => {
        if (err) return reject(err);
      });

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
      `, (err) => {
        if (err) return reject(err);
      });

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
          FOREIGN KEY (query_id)
            REFERENCES queries(id)
            ON DELETE CASCADE
        )
      `, (err) => {
        if (err) return reject(err);
      });

      db.run(`
        CREATE TABLE IF NOT EXISTS activity_logs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER,
          action TEXT NOT NULL,
          details TEXT,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (user_id)
            REFERENCES users(id)
        )
      `, (err) => {
        if (err) return reject(err);

        console.log('✅ Database tables initialized');
        resolve();
      });
    });
  });
}

// ----------------------------------------------------
// DATABASE HELPERS
// ----------------------------------------------------

function dbQuery(sql, params = []) {
  return new Promise((resolve, reject) => {
    db.all(sql, params, (err, rows) => {
      if (err) {
        console.error('❌ SQLite query error:', err.message);
        reject(err);
      } else {
        resolve(rows);
      }
    });
  });
}

function dbGet(sql, params = []) {
  return new Promise((resolve, reject) => {
    db.get(sql, params, (err, row) => {
      if (err) {
        console.error('❌ SQLite get error:', err.message);
        reject(err);
      } else {
        resolve(row);
      }
    });
  });
}

function dbRun(sql, params = []) {
  return new Promise((resolve, reject) => {
    db.run(sql, params, function (err) {
      if (err) {
        console.error('❌ SQLite write error:', err.message);
        reject(err);
      } else {
        resolve({
          id: this.lastID,
          changes: this.changes
        });
      }
    });
  });
}

// ----------------------------------------------------
// INITIALIZE
// ----------------------------------------------------

const dbReady = initializeDatabase()
  .then(() => {
    console.log('🚀 SatQuery database ready');
  })
  .catch((err) => {
    console.error(
      '❌ Database initialization failed:',
      err.message
    );
    throw err;
  });

// ----------------------------------------------------
// EXPORT
// ----------------------------------------------------

module.exports = {
  db,
  dbQuery,
  dbGet,
  dbRun,
  dbReady
};