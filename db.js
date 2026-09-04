const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const bcrypt = require('bcryptjs');

// Using In-Memory SQLite database to operate seamlessly without disk storage dependency
const DB_PATH = process.env.DB_PATH || ':memory:';

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

  // Seed default demo user 'analyst' / 'satquery' if not exists
  db.get(`SELECT * FROM users WHERE username = ?`, ['analyst'], (err, row) => {
    if (!err && !row) {
      const hash = bcrypt.hashSync('satquery', 10);
      db.run(
        `INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)`,
        ['analyst', hash, 'Lead EO Analyst'],
        (err) => {
          if (!err) console.log('👤 Default user created: username="analyst", password="satquery"');
        }
      );
    }
  });

  // Seed guest user if not exists
  db.get(`SELECT * FROM users WHERE username = ?`, ['Guest'], (err, row) => {
    if (!err && !row) {
      const hash = bcrypt.hashSync('guestpass', 10);
      db.run(
        `INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)`,
        ['Guest', hash, 'Guest Analyst'],
        (err) => {
          if (!err) console.log('👤 Guest user registered in SQLite DB.');
        }
      );
    }
  });
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

module.exports = {
  db,
  dbQuery,
  dbGet,
  dbRun
};
